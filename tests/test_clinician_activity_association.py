import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "06_analyze_clinician_activity_association.py"
spec = importlib.util.spec_from_file_location("activity_association", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    table = pd.DataFrame({
        "case_id": ["1", "2", "3", "4", "5"],
        "peristalsis_forward_count": [0, 0, 1, 2, 3],
        "peristalsis_reverse_count": [0, 1, 0, 2, 1],
    })
    for feature, column in module.FEATURES.items():
        table[column] = [1, 2, 3, 4, np.nan if feature == "F15" else 5]
        table[f"eligible_activity_{feature}"] = [True] * 4 + [feature != "F15"]
    manifest = {
        "eligible_feature_ids": list(module.FEATURES), "primary": "F01",
        "secondary": ["F07", "F09", "F15"], "decision_basis": "PAPER1_ONLY",
        "cohort_counts": {"base_rows": 5, **{
            f"activity_{feature}_eligible": int(table[f"eligible_activity_{feature}"].sum())
            for feature in module.FEATURES}},
    }
    return table, manifest


def test_frozen_contract_and_duplicate_cases_rejected():
    table, manifest = fixture()
    module.validate_master(table, manifest)
    manifest["eligible_feature_ids"] = ["F01", "F07"]
    with pytest.raises(ValueError, match="frozen feature contract"):
        module.validate_master(table, manifest)
    _, manifest = fixture()
    table.loc[1, "case_id"] = "1"
    with pytest.raises(ValueError, match="duplicate case_id"):
        module.validate_master(table, manifest)


def test_negative_or_fractional_counts_rejected_but_zero_retained():
    table, manifest = fixture()
    module.validate_master(table, manifest)
    table.loc[0, "peristalsis_forward_count"] = -1
    with pytest.raises(ValueError, match="nonnegative integer"):
        module.validate_master(table, manifest)
    table["peristalsis_forward_count"] = table["peristalsis_forward_count"].astype(float)
    table.loc[0, "peristalsis_forward_count"] = 0.5
    with pytest.raises(ValueError, match="nonnegative integer"):
        module.validate_master(table, manifest)


def test_feature_specific_cohorts_and_zero_counts(monkeypatch):
    table, manifest = fixture()
    module.validate_master(table, manifest)
    monkeypatch.setattr(module, "PERMUTATIONS", 99)
    monkeypatch.setattr(module, "BOOTSTRAPS", 100)
    result = module.analyze(table)
    assert len(result) == 8
    assert result.loc[(result.feature_id == "F01") & (result.activity_direction == "forward"), "n"].item() == 5
    assert result.loc[(result.feature_id == "F15") & (result.activity_direction == "forward"), "n"].item() == 4
    assert result.loc[(result.feature_id == "F01") & (result.activity_direction == "forward"), "zero_count_n"].item() == 2
    assert result.groupby("family").size().to_dict() == {"PRIMARY": 2, "SECONDARY": 6}


def test_constant_count_not_evaluable_and_holm_family_size_preserved():
    result = module.association(np.array([1., 2., 3.]), np.array([0., 0., 0.]), 0)
    assert result["status"] == "NOT_EVALUABLE"
    assert np.isnan(result["raw_p"])
    adjusted = module.holm_with_planned_family(pd.Series([0.01, np.nan]))
    assert adjusted[0] == pytest.approx(0.02)
    assert np.isnan(adjusted[1])


def test_deterministic_permutation_and_bootstrap(monkeypatch):
    monkeypatch.setattr(module, "PERMUTATIONS", 99)
    monkeypatch.setattr(module, "BOOTSTRAPS", 100)
    x = np.array([1., 2., 3., 4., 5.])
    y = np.array([0., 0., 1., 2., 1.])
    assert module.association(x, y, 0) == module.association(x, y, 0)
