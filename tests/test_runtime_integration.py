from __future__ import annotations

from pathlib import Path

from domain import GeneratorConfig
from engine import run_analysis
from runtime import DataImporter, ExportManager, RunRepository
from study import generate_dataset


def test_reference_pipeline_save_and_replay(tmp_path: Path):
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    result = run_analysis(generated.config, data=generated.data, compute_cate=False)
    repository = RunRepository(tmp_path / "repository")
    run_dir = repository.save(result, generated.data)
    assert (run_dir / "passport" / "CausalDecisionPassport.json").exists()
    assert (run_dir / "passport" / "CausalDecisionPassport.pdf").exists()
    assert repository.verify_checksums(result.manifest.run_id) == (True, ())
    replay = repository.replay(result.manifest.run_id)
    assert replay.matched, replay.differences


def test_import_export_csv_xlsx_parquet(tmp_path: Path):
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    for suffix in ("csv", "xlsx", "parquet"):
        path = ExportManager.export_table(generated.data, tmp_path / f"data.{suffix}")
        imported = DataImporter.read(path)
        assert len(imported) == len(generated.data)
        valid, warnings = DataImporter.validate(imported)
        assert valid, warnings


def test_importer_suggests_and_applies_alias_mapping():
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    renamed = generated.data.rename(columns={"T": "treatment", "Y_CR": "coverage_ratio"})
    mapping = DataImporter.suggest_mapping(renamed)
    assert mapping["T"] == "treatment"
    assert mapping["Y_CR"] == "coverage_ratio"
    mapped = DataImporter.map_columns(renamed, mapping)
    valid, warnings = DataImporter.validate(mapped)
    assert valid, warnings


def test_effect_scale_is_a_real_parameter():
    low_config = GeneratorConfig(mode="laboratory", sample_size=400, effect_scale=0.5, seed=99)
    high_config = GeneratorConfig(mode="laboratory", sample_size=400, effect_scale=1.5, seed=99)
    low = generate_dataset(low_config)
    high = generate_dataset(high_config)
    assert high.truth.true_ate["Y_CR"] > low.truth.true_ate["Y_CR"] * 2.0
    assert high.truth.true_ate["Y_CFO"] > low.truth.true_ate["Y_CFO"] * 2.0


def test_alpha_change_reuses_effects():
    generated = generate_dataset(GeneratorConfig(mode="laboratory", sample_size=300))
    result = run_analysis(generated.config, data=generated.data, compute_cate=False)
    signatures = {(e.graph_id, e.outcome): (e.estimate, e.interval) for e in result.effects}
    cut_060 = next(cut for cut in result.alpha_cuts if cut.alpha == 0.60)
    cut_080 = next(cut for cut in result.alpha_cuts if cut.alpha == 0.80)
    assert cut_060.graph_ids != cut_080.graph_ids
    assert signatures == {(e.graph_id, e.outcome): (e.estimate, e.interval) for e in result.effects}
