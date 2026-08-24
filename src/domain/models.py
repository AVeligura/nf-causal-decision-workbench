from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class IdentificationStatus(StrEnum):
    IDENTIFIED = "identified"
    PARTIALLY_IDENTIFIED = "partially_identified"
    NOT_IDENTIFIED = "not_identified"
    STRUCTURAL_ZERO = "structural_zero"


class CausalQuery(FrozenModel):
    query_id: str = "financial_recovery_cr"
    treatment: str = "T"
    baseline: str = "base_regime"
    outcome: Literal["Y_CR", "Y_CFO"] = "Y_CR"
    estimand: Literal["ATE", "ATT", "CATE"] = "ATE"
    horizon_quarters: int = 4
    population: str = "Средние промышленные предприятия с повышенной долговой нагрузкой"
    context: str = "Полусинтетическая совокупность предприятие–квартал"
    treatment_version: str = "full_package"


class DatasetSpec(FrozenModel):
    kind: Literal["reference", "imported"] = "reference"
    source: str = "UCI Polish Companies Bankruptcy"
    doi: str | None = "10.24432/C5F600"
    license: str | None = "CC BY 4.0"
    unit: str = "enterprise-quarter"
    selected_features: tuple[str, ...] = (
        "A1",
        "A2",
        "A4",
        "A5",
        "A21",
        "A27",
        "A29",
        "A44",
    )
    checksum_sha256: str = "afbfbed015d20f8421c32c62db37367c018eb6e92b00ea62a23354af8f84c44e"
    source_file_name: str | None = None
    file_format: str | None = None
    rows: int | None = None
    columns: int | None = None
    variable_mapping: dict[str, str] = Field(default_factory=dict)
    missingness: dict[str, float] = Field(default_factory=dict)
    imported_at: datetime | None = None
    user_description: str | None = None
    truth_available: bool = True


class GeneratorConfig(FrozenModel):
    profile_name: str = "reference"
    mode: Literal["reference", "laboratory", "import"] = "reference"
    scenario: Literal[
        "reference",
        "evidence_conflict",
        "weak_overlap",
        "version_mixing",
        "informative_loss",
        "outside_gamma",
    ] = "reference"
    preset_scenario: Literal[
        "reference",
        "evidence_conflict",
        "weak_overlap",
        "version_mixing",
        "informative_loss",
        "outside_gamma",
    ] | None = "reference"
    customized: bool = False
    value_regime: Literal["favorable", "boundary", "unfavorable"] = "favorable"
    true_graph_id: Literal["G1", "G2", "G3", "G4"] | None = "G1"
    sample_size: int = Field(1500, ge=300, le=5000)
    seed: int = 20260814
    pilot_seed: int | None = None
    truth_pilot_seed: int | None = None
    effect_scale: float = Field(1.0, ge=0.5, le=1.5)
    heterogeneity: float = Field(0.8, ge=0.0, le=1.5)
    nonlinearity: float = Field(0.8, ge=0.0, le=1.5)
    noise_scale: float = Field(1.0, ge=0.5, le=2.0)
    assignment_strength: float = Field(1.2, ge=0.5, le=2.5)
    propensity_lower: float = Field(0.15, ge=0.03, le=0.30)
    propensity_upper: float = Field(0.85, ge=0.70, le=0.97)
    partial_share: float = Field(0.10, ge=0.0, le=0.40)
    refusal_share: float = Field(0.0, ge=0.0, le=0.30)
    missing_share: float = Field(0.05, ge=0.0, le=0.35)
    hidden_confounding: float = Field(0.0, ge=0.0, le=1.5)
    evidence_reliability: float = Field(0.90, ge=0.50, le=0.99)
    evidence_conflict: float = Field(0.0, ge=0.0, le=1.0)
    pilot_share: float = Field(0.20, ge=0.10, le=0.40)
    value_multiplier: float = Field(1.0, ge=0.5, le=1.5)
    sales_loss_scale: float = Field(0.006, ge=0.0, le=0.030)
    zombie_risk_scale: float = Field(0.020, ge=0.0, le=0.100)
    program_cost_a1: float = Field(0.004, ge=0.0, le=0.030)
    program_cost_a2: float = Field(0.008, ge=0.0, le=0.060)
    cr_weight: float = Field(0.05, ge=0.0, le=0.50)
    financing_weight: float = Field(1.0, ge=0.0, le=3.0)
    arrears_weight: float = Field(0.60, ge=0.0, le=2.0)
    sales_loss_weight: float = Field(0.25, ge=0.0, le=2.0)
    zombie_weight: float = Field(0.03, ge=0.0, le=1.0)
    pilot_information_cost: float = Field(0.001, ge=0.0, le=0.020)
    conditional_regret_threshold: float = Field(0.005, ge=0.0, le=0.050)
    alpha: float = Field(0.60, ge=0.0, le=1.0)
    alpha_grid: tuple[float, ...] = (0.92, 0.90, 0.81, 0.80, 0.67, 0.60, 0.43, 0.40)
    cate_trees: int = Field(300, ge=20, le=1000)
    crossfit_folds: int = Field(5, ge=2, le=10)

    @model_validator(mode="after")
    def validate_joint_ranges(self) -> GeneratorConfig:
        if self.propensity_lower >= self.propensity_upper:
            raise ValueError("Нижняя граница propensity score должна быть меньше верхней")
        if self.partial_share + self.refusal_share > 0.70:
            raise ValueError("Суммарная доля неполного исполнения и отказов слишком велика")
        if self.program_cost_a1 > self.program_cost_a2:
            raise ValueError("Стоимость пилота не должна превышать стоимость полного внедрения")
        return self


class EvidenceItem(FrozenModel):
    evidence_id: str
    assertion_id: str
    assertion: str
    assertion_kind: Literal["edge", "path", "rule", "soft_rule"] = "edge"
    target_graph: str | None = None
    edge: tuple[str, str] | None = None
    path: tuple[str, str] | None = None
    rule_id: str | None = None
    expected_present: bool | None = None
    support: float = Field(ge=0.0, le=1.0)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    applicability: float = Field(1.0, ge=0.0, le=1.0)
    provenance: str
    evidence_type: Literal[
        "panel",
        "algorithmic",
        "expert",
        "regulatory",
        "interventional",
        "external",
    ]
    period: str = "reference"
    environment: str = "industrial_enterprises"
    context: str = "enterprise_financial_recovery"
    dependent_group: str
    constraint: Literal["hard", "soft"] = "soft"
    version: str = "1.0"
    comment: str = ""
    conflict: bool = False


class EvidenceBundle(FrozenModel):
    version: str
    context: str
    items: tuple[EvidenceItem, ...]


class GraphSpec(FrozenModel):
    graph_id: str
    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    graph_class: Literal["DAG"] = "DAG"
    version: str = "1.0"
    description: str = ""


class GraphScore(FrozenModel):
    graph_id: str
    mu: float
    valid: bool = True
    local_scores: dict[str, float] = Field(default_factory=dict)
    corrected_scores: dict[str, float] = Field(default_factory=dict)
    conflicts: dict[str, float] = Field(default_factory=dict)
    effective_weights: dict[str, float] = Field(default_factory=dict)
    coverage: float = 1.0
    provenance: dict[str, list[str]] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class AlphaCut(FrozenModel):
    alpha: float
    graph_ids: tuple[str, ...]
    core_edges: tuple[tuple[str, str], ...]
    alternative_edges: tuple[tuple[str, str], ...]
    empty: bool = False


class IdentificationResult(FrozenModel):
    graph_id: str
    query_id: str
    status: IdentificationStatus
    functional: str | None = None
    adjustment_set: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    identification_bounds: tuple[float, float] | None = None
    structural_origin: bool = False


class EffectResult(FrozenModel):
    graph_id: str
    query_id: str
    outcome: str
    estimand: str
    status: IdentificationStatus
    estimate: float | None = None
    interval: tuple[float, float] | None = None
    identified_bounds: tuple[float, float] | None = None
    bound_intervals: tuple[tuple[float, float], tuple[float, float]] | None = None
    standard_error: float | None = None
    functional: str | None = None
    adjustment_set: tuple[str, ...] = ()
    cate_profiles: dict[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class StabilityPoint(FrozenModel):
    alpha: float
    graph_ids: tuple[str, ...]
    uniformly_identified: bool
    functional_stable: bool
    sign_stable: bool
    threshold_stable: bool
    structural_spread: float | None = None
    first_changed_by: str | None = None
    warnings: tuple[str, ...] = ()


class StabilityProfile(FrozenModel):
    outcome: str
    points: tuple[StabilityPoint, ...]
    first_identification_loss_alpha: float | None = None
    first_sign_loss_alpha: float | None = None
    first_threshold_loss_alpha: float | None = None


class ActionSpec(FrozenModel):
    action_id: Literal["a0", "a1", "a2"]
    name: str
    coverage: float
    program_cost: float


class ValueModelConfig(FrozenModel):
    cr_weight: float = 0.05
    financing_weight: float = 1.0
    arrears_weight: float = 0.60
    sales_loss_weight: float = 0.25
    zombie_weight: float = 0.03
    pilot_information_cost: float = 0.001
    conditional_regret_threshold: float = 0.005
    cr_practical_threshold: float = 0.10
    cfo_practical_threshold: float = 0.015
    pilot_virtual_samples: int = 50
    pilot_share: float = Field(0.20, ge=0.0, le=1.0)
    program_cost_a1: float = 0.004
    program_cost_a2: float = 0.008
    financing_reduction_ratio: float = 0.35
    arrears_reduction_ratio: float = 0.22
    sales_loss_per_full_coverage: float = 0.006
    zombie_risk_per_full_coverage: float = 0.020
    multiplier: float = 1.0


class DecisionResult(FrozenModel):
    alpha: float
    status: Literal["robust", "conditionally_robust", "pilot", "abstain"]
    selected_action: str | None
    values: dict[str, dict[str, float | tuple[float, float] | None]]
    regrets: dict[str, dict[str, float | None]]
    maximum_regret: dict[str, float | None]
    worst_graph: dict[str, str | None]
    pilot_information_value: float = 0.0
    pilot_expected_regret_reduction: float = 0.0
    pilot_immediate_value: float = 0.0
    pilot_expected_continuation_value: float = 0.0
    pilot_gross_evi: float = 0.0
    pilot_information_cost: float = 0.0
    pilot_net_evi: float = 0.0
    pilot_r0: float = 0.0
    pilot_r1: float = 0.0
    pilot_adaptive_r1: float = 0.0
    pilot_common_evidence_method: str = ""
    pilot_admissible: bool = False
    pilot_information_value_se: float | None = None
    pilot_seed: int | None = None
    pilot_virtual_samples: int = 0
    reason: str = ""


class AlphaTrajectorySummary(FrozenModel):
    status: Literal["robust", "conditionally_robust", "switching", "pilot", "abstain"]
    selected_action: str | None
    stable_alpha_range: tuple[float, float] | None = None
    structural_condition: tuple[str, ...] = ()
    first_identification_loss_alpha: float | None = None
    first_identification_loss_graph: str | None = None
    first_sign_loss_alpha: float | None = None
    first_sign_loss_graph: str | None = None
    first_threshold_loss_alpha: float | None = None
    first_threshold_loss_graph: str | None = None
    first_action_change_alpha: float | None = None
    first_action_change_graph: str | None = None
    action_sequence: tuple[tuple[float, str | None], ...] = ()
    trajectory_maximum_regret: dict[str, float | None] = Field(default_factory=dict)
    operational_decision: DecisionResult
    reason: str = ""


class RunManifest(FrozenModel):
    run_id: str
    project_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    state: Literal[
        "draft", "valid", "stale", "running", "completed", "failed", "cancelled", "replayed"
    ] = "completed"
    config_hash: str
    data_hash: str
    app_version: str = "0.3.1"
    environment: dict[str, str] = Field(default_factory=dict)
    replay_of: str | None = None


class CausalDecisionPassport(FrozenModel):
    passport_version: str = "1.0"
    review_status: Literal["preliminary", "confirmed", "revised"] = "preliminary"
    manifest: RunManifest
    causal_queries: tuple[CausalQuery, ...]
    dataset: DatasetSpec
    evidence: dict[str, Any]
    structural_space: dict[str, Any]
    identification_and_estimation: dict[str, Any]
    uncertainty_profile: dict[str, Any]
    alpha_stability: dict[str, Any]
    decision: dict[str, Any]
    assumptions_and_limitations: tuple[str, ...]
    validation: dict[str, Any]
    audit_trail: tuple[dict[str, Any], ...] = ()


class AnalysisResult(FrozenModel):
    manifest: RunManifest
    config: GeneratorConfig
    dataset_spec: DatasetSpec
    graph_scores: tuple[GraphScore, ...]
    alpha_cuts: tuple[AlphaCut, ...]
    effects: tuple[EffectResult, ...]
    stability: tuple[StabilityProfile, ...]
    decisions: tuple[DecisionResult, ...]
    trajectory_summary: AlphaTrajectorySummary
    passport: CausalDecisionPassport
    diagnostics: dict[str, Any] = Field(default_factory=dict)
