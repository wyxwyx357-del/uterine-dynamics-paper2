import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "08_analyze_clinical_pregnancy.py"
spec = importlib.util.spec_from_file_location("pregnancy_analysis", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(n=240):
    rng = np.random.default_rng(20260922)
    table = pd.DataFrame({
        "case_id": pd.Series([str(i) for i in range(n)], dtype="string"),
        "female_age": rng.normal(32, 4, n),
        "female_bmi": rng.normal(22, 2, n),
        "infertility_years": rng.uniform(1, 8, n),
        "transfer_embryo_count": rng.integers(1, 3, n),
        "endometrial_thickness_mm": rng.normal(10, 1, n),
        "infertility_type": np.where(np.arange(n) % 2, "B", "A"),
        "embryo_type": np.where(np.arange(n) % 3, "blastocyst", "cleavage"),
        "endometrial_type": np.where(np.arange(n) % 4, "2", "3"),
        "cycle_type": np.where(np.arange(n) % 5, "natural", "artificial"),
    })
    outcome = rng.binomial(1, .45, n)
    table["clinical_pregnancy"] = outcome
    table["clinical_pregnancy_parseable_01"] = True
    table["eligible_pregnancy_association"] = True
    for i, (feature, column) in enumerate(module.FEATURES.items()):
        table[column] = rng.normal(1 + i, .3, n)
        table[f"eligible_pregnancy_{feature}"] = True
    counts = {"base_rows": n, "pregnancy_eligible": n,
              **{f"pregnancy_{feature}_eligible": n for feature in module.FEATURES}}
    manifest = {"eligible_feature_ids": list(module.FEATURES), "primary": "F01",
                "secondary": ["F07", "F09", "F15"], "decision_basis": "PAPER1_ONLY",
                "cohort_counts": counts}
    return table, manifest


def test_normal_models_and_matched_unadjusted_cohort():
    table, manifest = fixture()
    results, flow, diagnostics = module.analyze(table, manifest)
    assert len(results) == 12
    assert results.status.eq("EVALUABLE").all()
    assert results.loc[results.model.eq("ADJUSTED"), "n"].eq(240).all()
    assert np.isfinite(results.or_per_iqr).all()
    assert np.isfinite(results.ci_lower).all()
    assert np.isfinite(results.raw_p).all()
    assert flow.adjusted_n.eq(240).all()
    assert diagnostics["category_coding"]["infertility_type"]["reference"] == "A"


def test_duplicate_patient_and_bad_outcome_rejected():
    table, manifest = fixture()
    table.loc[1, "case_id"] = table.loc[0, "case_id"]
    with pytest.raises(ValueError, match="duplicate case_id"):
        module.validate_master(table, manifest)
    table, manifest = fixture()
    table.loc[0, "clinical_pregnancy"] = 2
    with pytest.raises(ValueError, match="expected 0/1"):
        module.validate_master(table, manifest)


def test_pregnancy_flags_must_match_outcome_and_feature():
    table, manifest = fixture()
    table.loc[0, "clinical_pregnancy"] = np.nan
    with pytest.raises(ValueError, match="parseability flag"):
        module.validate_master(table, manifest)
    table, manifest = fixture()
    table.loc[0, "eligible_pregnancy_F01"] = False
    with pytest.raises(ValueError, match="eligibility disagrees"):
        module.validate_master(table, manifest)


def test_covariate_missing_removes_only_adjusted_and_matched_rows():
    table, manifest = fixture()
    table.loc[0, "female_bmi"] = np.nan
    results, flow, _ = module.analyze(table, manifest)
    f01 = results.loc[results.feature_id.eq("F01")].set_index("model")
    assert f01.loc["UNADJUSTED_FULL", "n"] == 240
    assert f01.loc["UNADJUSTED_MATCHED", "n"] == 239
    assert f01.loc["ADJUSTED", "n"] == 239
    assert flow.loc[0, "missing_female_bmi_n"] == 1


def test_f15_missing_does_not_remove_other_features():
    table, manifest = fixture()
    column = module.FEATURES["F15"]
    table.loc[:9, column] = np.nan
    table.loc[:9, "eligible_pregnancy_F15"] = False
    manifest["cohort_counts"]["pregnancy_F15_eligible"] = 230
    results, flow, _ = module.analyze(table, manifest)
    assert results.loc[results.feature_id.eq("F01"), "n"].eq(240).all()
    assert results.loc[results.feature_id.eq("F15"), "n"].eq(230).all()
    assert flow.loc[flow.feature_id.eq("F15"), "feature_unavailable_among_pregnancy_base_n"].item() == 10


def test_category_coding_has_fixed_reference_and_columns():
    table, manifest = fixture()
    coding = module.category_coding(table)
    x = module.design(table, "F01", 1.0, coding, adjusted=True)
    assert coding["embryo_type"]["reference"] == "blastocyst"
    assert "embryo_type=cleavage" in x.columns
    assert "embryo_type=blastocyst" not in x.columns
    assert x["infertility_type=B"].sum() == 120


def test_complete_separation_has_no_estimate():
    table, manifest = fixture()
    table["clinical_pregnancy"] = (table[module.FEATURES["F01"]] > 1).astype(int)
    coding = module.category_coding(table)
    row, diagnostic = module.fit_model(table, "F01", .5, coding, "UNADJUSTED_FULL")
    assert row["status"] in {"COMPLETE_SEPARATION", "QUASI_SEPARATION"}
    assert np.isnan(row["or_per_iqr"])
    assert diagnostic["status"] == row["status"]


def test_nonconvergence_has_no_estimate(monkeypatch):
    table, manifest = fixture()
    class FailedFit:
        mle_retvals = {"converged": False}
    monkeypatch.setattr(module.sm.Logit, "fit", lambda self, **kwargs: FailedFit())
    row, _ = module.fit_model(table, "F01", .5, module.category_coding(table), "UNADJUSTED_FULL")
    assert row["status"] == "NONCONVERGED"
    assert np.isnan(row["raw_p"])


def test_sparse_category_recorded_and_rank_deficiency_blocks_estimate():
    table, manifest = fixture()
    table["cycle_type"] = "natural"
    table.loc[:2, "cycle_type"] = "artificial"
    coding = module.category_coding(table)
    row, diagnostic = module.fit_model(table, "F01", .5, coding, "ADJUSTED")
    assert diagnostic["sparse_categories"]["cycle_type"]["artificial"] == 3
    assert row["status"] in {"EVALUABLE", "QUASI_SEPARATION", "COMPLETE_SEPARATION"}
    assert diagnostic["category_outcome_counts"]["cycle_type"]["artificial"]["positive"] >= 0

    table["female_bmi"] = table["female_age"]
    row, diagnostic = module.fit_model(table, "F01", .5, coding, "ADJUSTED")
    assert row["status"] == "RANK_DEFICIENT"
    assert diagnostic["design_rank"] < diagnostic["design_columns_n"]


def test_holm_applies_only_to_three_adjusted_secondary_results():
    rows = pd.DataFrame({"feature_id": ["F01", "F07", "F09", "F15", "F07"],
                         "family": ["PRIMARY", "SECONDARY", "SECONDARY", "SECONDARY", "SECONDARY"],
                         "model": ["ADJUSTED", "ADJUSTED", "ADJUSTED", "ADJUSTED", "UNADJUSTED_FULL"],
                         "raw_p": [.001, .01, .03, np.nan, .0001],
                         "holm_adjusted_p": [np.nan] * 5})
    result = module.holm_secondary(rows)
    assert result.loc[1, "holm_adjusted_p"] == pytest.approx(.03)
    assert result.loc[2, "holm_adjusted_p"] == pytest.approx(.06)
    assert np.isnan(result.loc[3, "holm_adjusted_p"])
    assert np.isnan(result.loc[4, "holm_adjusted_p"])


def test_synthetic_command_line_outputs_and_hashes(tmp_path):
    table, manifest = fixture()
    master = tmp_path / "paper2_analysis_master.csv"
    table.to_csv(master, index=False)
    gate = tmp_path / "paper2_feature_gate.csv"
    gate.write_text("frozen synthetic gate\n", encoding="utf-8")
    manifest["input_paths"] = {"frozen_gate": str(gate)}
    manifest["frozen_gate_sha256"] = module.sha256(gate)
    manifest["output_sha256"] = {"master_csv": module.sha256(master)}
    source_manifest = tmp_path / "manifest.json"
    source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "pregnancy_output"
    subprocess.run([sys.executable, str(SCRIPT), "--master", str(master),
                    "--master-manifest", str(source_manifest), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    results = pd.read_csv(output / "pregnancy_associations.csv")
    assert len(results) == 12
    assert results.status.eq("EVALUABLE").all()
    recorded = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert recorded["input_sha256"]["master"] == module.sha256(master)
    assert recorded["output_sha256"]["associations"] == module.sha256(
        output / "pregnancy_associations.csv")
