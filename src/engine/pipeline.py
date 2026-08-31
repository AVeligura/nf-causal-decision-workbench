from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from typing import Any

import numpy as np
import pandas as pd

from domain import (
    AnalysisResult,
    CausalQuery,
    DatasetSpec,
    EffectResult,
    EvidenceBundle,
    GeneratorConfig,
    RunManifest,
)
from study.dgp import generate_dataset, value_config_from_generator

from .decision import evaluate_decisions
from .estimation import estimate_effect
from .evidence import alpha_cut, reference_evidence, score_graph
from .graphs import reference_graphs
from .identification import identify_effect
from .passport import build_causal_structure_passport
from .stability import assess_effect_stability, summarize_alpha_trajectory


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _data_hash(data: pd.DataFrame) -> str:
    values = pd.util.hash_pandas_object(data, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def run_analysis(
    config: GeneratorConfig,
    *,
    data: pd.DataFrame | None = None,
    dataset_spec: DatasetSpec | None = None,
    evidence_bundle: EvidenceBundle | None = None,
    compute_cate: bool = True,
    project_id: str = "reference-project",
    replay_of: str | None = None,
) -> AnalysisResult:
    generated_diagnostics: dict[str, Any] = {}
    if data is None:
        generated = generate_dataset(config)
        data = generated.data
        config = generated.config
        generated_diagnostics = generated.diagnostics
        dataset_spec = dataset_spec or DatasetSpec()
    elif dataset_spec is None:
        dataset_spec = DatasetSpec() if config.mode != "import" else DatasetSpec(
            kind="imported",
            source="Imported dataset",
            doi=None,
            license=None,
            selected_features=tuple(),
            checksum_sha256="unknown",
            rows=len(data),
            columns=len(data.columns),
            truth_available=False,
        )
    graphs = reference_graphs()
    evidence = evidence_bundle or reference_evidence(
            reliability_multiplier=config.evidence_reliability / 0.90,
            conflict_strength=config.evidence_conflict,
        )
    scores = tuple(score_graph(graph, evidence) for graph in graphs)
    alphas = set(float(alpha) for alpha in config.alpha_grid)
    if config.scenario == "evidence_conflict":
        alphas.update(score.mu for score in scores)
    alpha_cuts = tuple(alpha_cut(graphs, scores, alpha) for alpha in sorted(alphas, reverse=True))

    queries = (
        CausalQuery(query_id="financial_recovery_cr", outcome="Y_CR", estimand="ATE"),
        CausalQuery(query_id="financial_recovery_cfo", outcome="Y_CFO", estimand="ATE"),
    )
    effects = []
    estimation_cache: dict[tuple[Any, ...], EffectResult] = {}
    for query in queries:
        for graph in graphs:
            identification = identify_effect(
                graph,
                query,
                scenario=config.scenario,
                imported=config.mode == "import",
            )
            cache_key = (
                query.outcome,
                query.estimand,
                identification.status,
                identification.functional,
                identification.adjustment_set,
            )
            if cache_key in estimation_cache:
                cached = estimation_cache[cache_key]
                effect = cached.model_copy(
                    update={"graph_id": graph.graph_id, "query_id": query.query_id}
                )
            else:
                effect = estimate_effect(
                    data,
                    identification,
                    outcome=query.outcome,
                    estimand=query.estimand,
                    folds=config.crossfit_folds,
                    cate_trees=config.cate_trees,
                    seed=config.seed + int(graph.graph_id[-1]),
                    compute_cate=compute_cate,
                )
                estimation_cache[cache_key] = effect
            effects.append(effect)
    effect_tuple = tuple(effects)
    stability = (
        assess_effect_stability(
            alpha_cuts,
            effect_tuple,
            outcome="Y_CR",
            practical_threshold=0.10,
        ),
        assess_effect_stability(
            alpha_cuts,
            effect_tuple,
            outcome="Y_CFO",
            practical_threshold=0.015,
        ),
    )
    value_config = value_config_from_generator(config)
    decision_started = time.perf_counter()
    decisions = tuple(
        evaluate_decisions(
            effect_tuple,
            cut.graph_ids,
            alpha=cut.alpha,
            value_config=value_config,
            seed=config.pilot_seed if config.pilot_seed is not None else config.seed,
        )
        for cut in alpha_cuts
        if not cut.empty
    )
    trajectory_summary = summarize_alpha_trajectory(
        alpha_cuts,
        decisions,
        stability,
        value_config=value_config,
    )
    full_procedure_decision_seconds = time.perf_counter() - decision_started

    config_hash = _stable_hash(config.model_dump(mode="json"))
    data_hash = _data_hash(data)
    evidence_hash = _stable_hash(evidence.as_dict())
    analysis_input_hash = _stable_hash(
        {
            "config_hash": config_hash,
            "data_hash": data_hash,
            "evidence_hash": evidence_hash,
        }
    )
    manifest = RunManifest(
        run_id=f"run-{uuid.uuid4().hex[:12]}",
        project_id=project_id,
        config_hash=config_hash,
        data_hash=data_hash,
        state="replayed" if replay_of else "completed",
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        replay_of=replay_of,
    )
    limitations = [
        "Устойчивость внутри Γ не доказывает присутствие истинной структуры в пространстве кандидатов",
        "μΓ(G) не является вероятностью и не используется как вес причинного эффекта",
        "Статистические интервалы условны относительно графа и идентифицирующего функционала",
    ]
    if config.scenario == "weak_overlap":
        limitations.append(
            "Слабое overlap ограничивает эмпирическую поддержку альтернативных режимов"
        )
    if config.scenario == "outside_gamma":
        limitations.append("Истинная структура содержит ненаблюдаемый U и отсутствует в Γ")
    passport = build_causal_structure_passport(
        manifest=manifest,
        config=config,
        queries=queries,
        dataset_spec=dataset_spec,
        evidence=evidence,
        evidence_hash=evidence_hash,
        analysis_input_hash=analysis_input_hash,
        graphs=graphs,
        graph_scores=scores,
        alpha_cuts=alpha_cuts,
        effects=effect_tuple,
        stability=stability,
        decisions=decisions,
        trajectory_summary=trajectory_summary,
        assumptions_and_limitations=tuple(limitations),
    )
    return AnalysisResult(
        manifest=manifest,
        config=config,
        dataset_spec=dataset_spec,
        graph_scores=scores,
        alpha_cuts=alpha_cuts,
        effects=effect_tuple,
        stability=stability,
        decisions=decisions,
        trajectory_summary=trajectory_summary,
        passport=passport,
        diagnostics={
            "data": generated_diagnostics,
            "rows": len(data),
            "oracle_truth_available": dataset_spec.truth_available,
            "full_procedure_decision_seconds": full_procedure_decision_seconds,
            "evidence_hash": evidence_hash,
            "analysis_input_hash": analysis_input_hash,
        },
    )
