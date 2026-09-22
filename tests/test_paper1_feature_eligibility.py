from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "04_audit_paper1_feature_eligibility.py"
)
spec = importlib.util.spec_from_file_location("paper2_gate", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def perturbation():
    rows = []
    for r in mod.CONTRACT.itertuples(index=False):
        rows.append(
            {
                "feature_code": r.feature_id,
                "feature": r.feature_name,
                "n_cases": 100,
                "primary_icc": 0.95,
                "ci_lower": 0.91,
                "ci_upper": 0.98,
                "median_cv_pct": 2.0,
                "p95_cv_pct": 5.0,
                "cv_lt20_fraction": 1.0,
            }
        )
    return pd.DataFrame(rows)


def duration():
    rows = []
    for target in (75, 50, 25):
        for r in mod.CONTRACT.itertuples(index=False):
            rows.append(
                {
                    "feature_code": r.feature_id,
                    "feature_name": r.feature_name,
                    "target_percent": target,
                    "reference_percent": 100,
                    "n_pairs": 90,
                    "icc_a1": 0.90,
                    "icc_ci_lower": 0.85,
                    "icc_ci_upper": 0.95,
                    "spearman_rho": 0.90,
                    "median_absolute_relative_error_pct": 5.0,
                    "p95_absolute_relative_error_pct": 15.0,
                }
            )
    return pd.DataFrame(rows)


def qc():
    rows = []
    for stratum in ("all", "pending_grade3"):
        for r in mod.CONTRACT.itertuples(index=False):
            rows.append(
                {
                    "stratum": stratum,
                    "feature": r.feature_name,
                    "statistic": r.statistic,
                    "n_original_finite": 80,
                    "n_paired_finite": 79,
                    "finite_to_nan_n": 1,
                    "availability_retention_fraction": 0.9875,
                    "spearman": 0.95,
                    "median_srd_pct": 2.0,
                    "p95_srd_pct": 10.0,
                }
            )
    return pd.DataFrame(rows)


def temporal():
    rows = []
    for stratum in ("all", "pending_grade3"):
        for r in mod.CONTRACT.itertuples(index=False):
            rows.append(
                {
                    "stratum": stratum,
                    "feature_id": r.feature_id,
                    "feature": r.feature_name,
                    "family": r.family,
                    "domain": r.domain,
                    "statistic": r.statistic,
                    "n_cases": 80,
                    "n_temporal_spearman_available": 79,
                    "median_temporal_spearman": 0.90,
                    "p05_temporal_spearman": 0.70,
                    "median_valid_position_retention_fraction": 0.98,
                    "finite_to_nan_bins_total": 1,
                    "cases_with_finite_to_nan_bins": 1,
                    "median_case_median_bin_srd_pct": 3.0,
                    "p95_case_median_bin_srd_pct": 12.0,
                    "median_case_p95_bin_srd_pct": 8.0,
                    "median_aggregate_srd_pct": 2.0,
                }
            )
    return pd.DataFrame(rows)


def spatial():
    rows = []
    for stratum in ("all", "pending_grade3"):
        for r in mod.CONTRACT.itertuples(index=False):
            rows.append(
                {
                    "stratum": stratum,
                    "feature_id": r.feature_id,
                    "feature": r.feature_name,
                    "family": r.family,
                    "domain": r.domain,
                    "statistic": r.statistic,
                    "n_cases": 80,
                    "n_spatial_spearman_available": 79,
                    "median_spatial_spearman": 0.90,
                    "p05_spatial_spearman": 0.70,
                    "median_valid_position_retention_fraction": 0.98,
                    "finite_to_nan_bins_total": 1,
                    "cases_with_finite_to_nan_bins": 1,
                    "median_case_median_bin_srd_pct": 3.0,
                    "p95_case_median_bin_srd_pct": 12.0,
                    "median_case_p95_bin_srd_pct": 8.0,
                    "median_aggregate_srd_pct": 2.0,
                }
            )
    return pd.DataFrame(rows)


def build():
    return mod.build_evidence(
        perturbation=perturbation(),
        duration=duration(),
        qc=qc(),
        temporal=temporal(),
        spatial=spatial(),
    )


def test_complete_evidence_does_not_auto_select():
    evidence = build()
    assert len(evidence) == 20
    assert set(evidence.paper1_evidence_status) == {
        "EVIDENCE_READY_FOR_PRESPECIFIED_DECISION"
    }
    gate = mod.apply_decisions(evidence, None)
    assert set(gate.paper2_decision) == {"UNFROZEN"}
    assert gate.paper2_eligible.isna().all()
    assert not evidence.automatic_robustness_pass_created.any()


def test_missing_duration_feature_is_rejected():
    d = duration()
    d = d[~((d.feature_code == "F20") & (d.target_percent == 25))]
    with pytest.raises(ValueError, match="F20"):
        mod.build_evidence(
            perturbation=perturbation(),
            duration=d,
            qc=qc(),
            temporal=temporal(),
            spatial=spatial(),
        )


def test_paper1_only_decisions_apply(tmp_path):
    evidence = build()
    table = mod.decision_template()
    table["decision"] = "EXCLUDE"
    table["role"] = "NONE"
    table["rationale"] = "Paper 1 evidence review"
    table.loc[
        table.feature_id == "F01", ["decision", "role"]
    ] = ["INCLUDE", "PRIMARY"]
    table.loc[
        table.feature_id == "F07", ["decision", "role"]
    ] = ["INCLUDE", "SECONDARY"]
    path = tmp_path / "decisions.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")

    decisions = mod.load_decisions(path)
    gate = mod.apply_decisions(evidence, decisions)
    assert (
        gate.loc[gate.feature_id == "F01", "paper2_role"].item()
        == "PRIMARY"
    )
    assert gate.loc[gate.feature_id == "F07", "paper2_eligible"].item()
    assert int(gate.paper2_eligible.sum()) == 2


def test_outcome_style_columns_in_decision_file_are_rejected(tmp_path):
    table = mod.decision_template()
    table["decision"] = "EXCLUDE"
    table["role"] = "NONE"
    table["rationale"] = "Paper 1 only"
    table.loc[
        table.feature_id == "F01", ["decision", "role"]
    ] = ["INCLUDE", "PRIMARY"]
    table["pregnancy_auc"] = 0.60
    path = tmp_path / "bad.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")

    with pytest.raises(ValueError, match="not allowed"):
        mod.load_decisions(path)


def test_contract_metadata_mismatch_is_rejected():
    table = temporal()
    table.loc[
        (table.stratum == "pending_grade3")
        & (table.feature_id == "F01"),
        "family",
    ] = "curvature"
    with pytest.raises(ValueError, match="frozen F01-F20 contract"):
        mod.build_evidence(
            perturbation=perturbation(),
            duration=duration(),
            qc=qc(),
            temporal=table,
            spatial=spatial(),
        )
