import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_explore_spatiotemporal_activity.py"
spec = importlib.util.spec_from_file_location("profile_activity", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture(n=6):
    cases = [f"CASE_{i}_20260801" for i in range(1, n + 1)]
    master = pd.DataFrame({
        "case_id": pd.Series([str(i) for i in range(1, n + 1)], dtype="string"),
        "paper1_filename_date8": ["20260801"] * n,
        "paper1_feature_match_status": ["MATCHED"] * n,
        "F01_rsr_abs_median": [2.0] * n,
        "eligible_activity_F01": [True] * (n-1) + [False],
        "peristalsis_forward_count": [0, 0, 1, 2, 3, 2][:n],
        "peristalsis_reverse_count": [0, 1, 0, 2, 1, 3][:n],
    })
    profile = pd.DataFrame([
        {"case_id": case, "feature_id": "F01", "feature": "rsr_abs_median",
         "statistic": "median", "bin_index": j+1, "original": 2.0 + (j-2) * 0.1 * i,
         "original_valid_positions": 12}
        for i, case in enumerate(cases, 1) for j in range(5)
    ])
    summary = pd.DataFrame({"case_id": cases, "feature_id": ["F01"] * n,
                            "feature": ["rsr_abs_median"] * n,
                            "aggregate_original": [2.0] * n})
    source = {"bins": 5, "clinical_labels_used": False, "cases": cases,
              "normalized_position_definition": "frozen section coordinate"}
    return master, profile, summary, source


def test_original_profile_relative_mad_and_all_four_exploratory_tests(monkeypatch):
    master, profile, summary, source = fixture()
    temporal = module.original_profile_features(profile, summary, source, master, "temporal")
    spatial = module.original_profile_features(profile, summary, source, master, "spatial")
    assert temporal.temporal_relative_mad.iloc[0] == pytest.approx(0.05)
    assert spatial.spatial_relative_mad.iloc[0] == pytest.approx(0.05)
    assert temporal.temporal_complete_bins.eq(5).all()
    combined = module.assemble(master, temporal, spatial)
    monkeypatch.setattr(module.core, "PERMUTATIONS", 49)
    monkeypatch.setattr(module.core, "BOOTSTRAPS", 100)
    result = module.analyze(combined)
    assert len(result) == 4
    assert result.n.tolist() == [5, 5, 5, 5]
    assert result.zero_count_n.tolist() == [2, 2, 2, 2]
    assert result.holm_adjusted_p.ge(result.raw_p - 1e-12).all()
    assert result.family.eq("EXPLORATORY_SPATIOTEMPORAL_4").all()
    assert result.equals(module.analyze(combined))


def test_profile_missing_bin_or_zero_median_remains_missing():
    master, profile, summary, source = fixture()
    profile.loc[0, ["original", "original_valid_positions"]] = [np.nan, 0]
    # Five valid zeros produce undefined relative MAD (not a false numeric zero).
    profile.loc[profile.case_id.eq("CASE_2_20260801"), "original"] = 0.0
    t = module.original_profile_features(profile, summary, source, master, "temporal")
    assert t.temporal_complete_bins.tolist()[:2] == [4, 5]
    assert np.isnan(t.temporal_relative_mad.iloc[0])
    assert np.isnan(t.temporal_relative_mad.iloc[1])


def test_manifest_and_profile_structure_validation():
    master, profile, summary, source = fixture()
    altered = profile.copy()
    altered.loc[1, "bin_index"] = 1
    with pytest.raises(ValueError, match="duplicate case/bin"):
        module.original_profile_features(altered, summary, source, master, "spatial")
    source["bins"] = 4
    with pytest.raises(ValueError, match="five-bin source manifest"):
        module.original_profile_features(profile, summary, source, master, "temporal")


def test_wrong_exam_date_or_aggregate_rejected():
    master, profile, summary, source = fixture()
    wrong = master.copy()
    wrong.loc[0, "paper1_filename_date8"] = "20260901"
    with pytest.raises(ValueError, match="filename date disagrees"):
        module.original_profile_features(profile, summary, source, wrong, "temporal")
    bad = summary.copy()
    bad.loc[0, "aggregate_original"] = 3.0
    with pytest.raises(ValueError, match="aggregate differs"):
        module.original_profile_features(profile, bad, source, master, "spatial")


def test_negative_bin_and_missing_profile_case_rejected():
    master, profile, summary, source = fixture()
    bad = profile.copy()
    bad.loc[0, "original"] = -0.1
    with pytest.raises(ValueError, match="bin value/count inconsistency"):
        module.original_profile_features(bad, summary, source, master, "temporal")
    missing = profile.loc[~profile.case_id.eq("CASE_1_20260801")]
    with pytest.raises(ValueError, match="source manifest cases differ"):
        module.original_profile_features(missing, summary, source, master, "spatial")


def test_partial_source_subset_does_not_require_all_master_patients():
    master, profile, summary, source = fixture()
    source["cases"] = source["cases"][:-1]
    profile = profile.loc[profile.case_id.isin(source["cases"])]
    summary = summary.loc[summary.case_id.isin(source["cases"])]
    t = module.original_profile_features(profile, summary, source, master, "temporal")
    combined = module.assemble(master, t, t.rename(columns={"temporal_complete_bins": "spatial_complete_bins", "temporal_relative_mad": "spatial_relative_mad"}))
    assert len(combined) == 6
    assert not combined.temporal_source_available.iloc[-1]
    assert not combined.spatial_eligible.iloc[-1]
