"""Synthetic regression tests: raw activity cells must survive stages 02, 03 and 05."""
import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest
from openpyxl import load_workbook

from scripts.activity_count_io import read_excel_preserving_activity


ROOT = Path(__file__).resolve().parents[1]


def load_script(filename):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("token", ["NULL", "#N/A"])
def test_raw_excel_text_and_other_fields_keep_normal_na_parsing(tmp_path, token):
    path = tmp_path / "source.xlsx"
    pd.DataFrame({
        "peristalsis_forward_count": [token, 0],
        "peristalsis_reverse_count": [1, None],
        "unrelated": ["NULL", None],
    }).to_excel(path, sheet_name="source", index=False)
    table = read_excel_preserving_activity(path, "source")
    assert table.loc[0, "peristalsis_forward_count"] == token
    assert table.loc[1, "peristalsis_forward_count"] == 0
    assert pd.isna(table.loc[0, "unrelated"])


def test_native_excel_error_cell_is_preserved(tmp_path):
    path = tmp_path / "error.xlsx"
    pd.DataFrame({"peristalsis_forward_count": [0]}).to_excel(
        path, sheet_name="source", index=False)
    book = load_workbook(path)
    book["source"]["A2"] = "#N/A"
    book["source"]["A2"].data_type = "e"
    book.save(path)
    table = read_excel_preserving_activity(path, "source")
    assert table.loc[0, "peristalsis_forward_count"] == "#N/A"


@pytest.mark.parametrize("token", ["NULL", "#N/A"])
def test_tokens_survive_02_03_05_excel_path_and_fail_validation(
    tmp_path, monkeypatch, token
):
    script02 = load_script("02_build_video_clinical_match.py")
    script03 = load_script("03_audit_patient_data.py")
    script05 = load_script("05_build_paper2_analysis_master.py")

    clinical = tmp_path / "clinical.xlsx"
    patient = pd.DataFrame({
        "case_id": ["101"], "patient_name": ["synthetic"],
        "patient_id": ["20240101"], "peristalsis_forward_count": [token],
        "peristalsis_reverse_count": [0],
    })
    qc = pd.DataFrame({
        "case_id": ["101"], "image_datetime": ["20240101"],
        "matched_exam_datetime": ["20240101"],
    })
    with pd.ExcelWriter(clinical) as writer:
        patient.to_excel(writer, sheet_name=script02.PATIENT_SHEET, index=False)
        qc.to_excel(writer, sheet_name=script02.QC_SHEET, index=False)
    patient_loaded, _ = script02.read_clinical(clinical)
    assert patient_loaded.loc[0, "peristalsis_forward_count"] == token

    matched = tmp_path / "matched.xlsx"
    patient_loaded["final_audit_status"] = "PASS_THREE_WAY"
    with pd.ExcelWriter(matched) as writer:
        patient_loaded.to_excel(writer, sheet_name=script03.INPUT_SHEET, index=False)
    audited = tmp_path / "audited.xlsx"
    monkeypatch.setattr(sys, "argv", [
        "03_audit_patient_data.py", "--input", str(matched), "--output", str(audited)])
    script03.main()

    table = script05.read_excel_preserving_activity(
        audited, script05.PATIENT_SHEET, dtype={"case_id": "string"})
    assert table.loc[0, "peristalsis_forward_count"] == token
    assert table.loc[0, "peristalsis_reverse_count"] == 0
    with pytest.raises(ValueError, match="peristalsis_forward_count.*nonnegative integer"):
        script05.parse_activity_count_column(
            table["peristalsis_forward_count"], "peristalsis_forward_count")
