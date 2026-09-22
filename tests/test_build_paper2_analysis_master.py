import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "05_build_paper2_analysis_master.py"
spec = importlib.util.spec_from_file_location("paper2_master", SCRIPT)
master_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master_module)


def inputs():
    gate_rows = []
    for i in range(1, 21):
        feature_id = f"F{i:02d}"
        role = "PRIMARY" if feature_id == "F01" else "SECONDARY" if feature_id in {
            "F07", "F09", "F15"
        } else "NONE"
        gate_rows.append({
            "feature_id": feature_id,
            "feature_name": master_module.FEATURE_NAMES.get(feature_id, f"other_{feature_id}"),
            "paper2_decision": "EXCLUDE" if role == "NONE" else "INCLUDE",
            "paper2_role": role,
            "paper2_eligible": role != "NONE",
            "paper2_decision_basis": "PAPER1_ONLY",
        })
    gate = pd.DataFrame(gate_rows)
    base = pd.DataFrame({
        "case_id": ["101", "102", "103"],
        "video_case_date_status": ["CONFIRMED"] * 3,
        "patient_match_status": ["MATCH_UNIQUE"] * 3,
        "qc_match_status": ["MATCH_UNIQUE"] * 3,
        "match_status": ["MATCH"] * 3,
        "video_vs_image_date": ["MATCH"] * 3,
        "video_vs_matched_exam_date": ["MATCH"] * 3,
        "video_vs_patient_id_date": ["MATCH"] * 3,
        "video_real_exam_date8": ["20240101", "20240102", "20240103"],
        "matched_exam_date8": ["20240101", "20240102", "20240103"],
        "video_filenames": ["A_101_20240101.mp4", "B_102_20240102.mp4",
                            "C_103_20240103.mp4"],
        "final_audit_status": ["PASS_THREE_WAY"] * 3,
        "date_audit_pass": [True] * 3,
        "eligible_activity_association": [True, True, False],
        "eligible_pregnancy_association": [True, False, True],
        "clinical_pregnancy_parseable_01": [True, False, True],
        "clinical_pregnancy_binary_qc": [1, None, 0],
        "peristalsis_forward_count": [0, 0, None],
        "peristalsis_reverse_count": [0, 1, None],
        "female_age": [30, 31, 32],
        "female_bmi": [21, 22, 23],
        "infertility_years\nsource": [2, 3, 4],
        "transfer_embryo_count": [1, 1, 1],
        "endometrial_thickness_mm": [10, 10, 10],
        "infertility_type\nsource": ["primary"] * 3,
        "embryo_type": ["A"] * 3,
        "endometrial_type": ["A"] * 3,
        "cycle_type": ["A"] * 3,
    })
    feature_rows = []
    for case_id, curvature in (("A_101_20240101", 4.0), ("B_102_20240102", None)):
        row = {"case_id": case_id, **{name: 1.0 for name in gate.feature_name}}
        row[master_module.FEATURE_NAMES["F15"]] = curvature
        feature_rows.append(row)
    video_audit = pd.DataFrame({
        "video_filename": ["A_101_20240101.mp4", "B_102_20240102.mp4"],
        "name_date": ["20240101", "20240102"],
        "ocr_date": ["20240101", "20240102"],
        "ocr_status": ["UNIQUE_OCR_DATE"] * 2,
    })
    return base, pd.DataFrame(feature_rows), gate, video_audit


def test_duplicate_case_id_rejected():
    base, features, gate, video_audit = inputs()
    base.loc[1, "case_id"] = "101"
    with pytest.raises(ValueError, match="duplicate case_id"):
        master_module.build_master(base, features, gate, video_audit)


def test_duplicate_feature_case_id_rejected():
    base, features, gate, video_audit = inputs()
    features.loc[1, "case_id"] = "B_101_20240102"
    with pytest.raises(ValueError, match="duplicate case_id"):
        master_module.build_master(base, features, gate, video_audit)


def test_wrong_frozen_gate_rejected():
    base, features, gate, video_audit = inputs()
    gate.loc[gate.feature_id.eq("F15"), "paper2_eligible"] = False
    with pytest.raises(ValueError, match="frozen gate"):
        master_module.build_master(base, features, gate, video_audit)


def test_zero_activity_and_missing_pregnancy_preserved():
    base, features, gate, video_audit = inputs()
    master, _ = master_module.build_master(base, features, gate, video_audit)
    first = master.set_index("case_id").loc["101"]
    second = master.set_index("case_id").loc["102"]
    assert first.peristalsis_forward_count == 0
    assert first.peristalsis_reverse_count == 0
    assert first.eligible_activity_association
    assert pd.isna(second.clinical_pregnancy)
    assert not second.eligible_pregnancy_association


def test_f15_missing_does_not_remove_f01_activity_eligibility():
    base, features, gate, video_audit = inputs()
    master, _ = master_module.build_master(base, features, gate, video_audit)
    second = master.set_index("case_id").loc["102"]
    assert not second.F15_available
    assert second.eligible_activity_F01
    assert second.eligible_activity_F07
    assert second.eligible_activity_F09
    assert not second.eligible_activity_F15


def test_feature_specific_eligibility_and_no_extra_cases():
    base, features, gate, video_audit = inputs()
    master, _ = master_module.build_master(base, features, gate, video_audit)
    assert len(master) == 3
    assert set(master.case_id) == set(base.case_id)
    assert master.eligible_activity_F01.sum() == 2
    assert master.eligible_activity_F15.sum() == 1
    assert master.eligible_pregnancy_F01.sum() == 1
    assert master.eligible_pregnancy_F15.sum() == 1
    assert master.set_index("case_id").loc["103", "paper1_feature_match_status"] == "NO_PAPER1_FEATURE"


def test_excluded_features_do_not_enter_master():
    base, features, gate, video_audit = inputs()
    master, _ = master_module.build_master(base, features, gate, video_audit)
    for name in gate.loc[gate.paper2_decision.eq("EXCLUDE"), "feature_name"]:
        assert name not in master.columns
    assert sum(col.startswith("F01_") and not col.endswith("available") for col in master) == 1


def test_filename_date_mismatch_is_accepted_when_video_ocr_confirms_exam_date():
    base, features, gate, video_audit = inputs()
    features.loc[0, "case_id"] = "A_101_20240201"
    base.loc[0, "video_filenames"] = "A_101_20240201.mp4"
    video_audit.loc[0, "video_filename"] = "A_101_20240201.mp4"
    video_audit.loc[0, "name_date"] = "20240201"
    master, _ = master_module.build_master(base, features, gate, video_audit)
    first = master.set_index("case_id").loc["101"]
    assert len(master) == len(base)
    assert first.paper1_feature_match_status == "MATCHED"
    assert first.paper1_filename_date_differs_from_exam_date
    assert first.F01_available
    assert first.eligible_activity_F01


def test_unconfirmed_or_discordant_video_exam_date_rejected():
    base, features, gate, video_audit = inputs()
    video_audit.loc[0, "ocr_date"] = "20240201"
    with pytest.raises(ValueError, match="confirmed exam date disagrees"):
        master_module.build_master(base, features, gate, video_audit)


def test_wrong_video_identity_rejected():
    base, features, gate, video_audit = inputs()
    base.loc[0, "video_filenames"] = "another_video.mp4"
    with pytest.raises(ValueError, match="video identity disagrees"):
        master_module.build_master(base, features, gate, video_audit)
