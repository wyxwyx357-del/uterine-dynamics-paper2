#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the patient-level Paper 2 master from frozen Paper 1 measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PATIENT_SHEET = "02_患者级状态"
FEATURE_IDS = ("F01", "F07", "F09", "F15")
FEATURE_NAMES = {
    "F01": "rsr_abs_median",
    "F07": "cavity_width_strain_rate_abs_median",
    "F09": "longitudinal_wall_strain_rate_abs_median",
    "F15": "wall_curvature_change_rate_mm_inv_s_abs_median",
}
STATUS_COLUMNS = (
    "video_case_date_status", "patient_match_status", "qc_match_status",
    "match_status", "video_vs_image_date", "video_vs_matched_exam_date",
    "video_vs_patient_id_date", "video_real_exam_date8", "matched_exam_date8",
    "final_audit_status", "date_audit_pass",
    "eligible_activity_association", "eligible_pregnancy_association",
    "clinical_pregnancy_parseable_01",
)
CLINICAL_COLUMNS = (
    "peristalsis_forward_count", "peristalsis_reverse_count",
    "female_age", "female_bmi", "infertility_years",
    "transfer_embryo_count", "endometrial_thickness_mm",
    "infertility_type", "embryo_type", "endometrial_type", "cycle_type",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(table: pd.DataFrame, columns, source: str) -> None:
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def unique_case_ids(table: pd.DataFrame, source: str) -> None:
    if table["case_id"].isna().any() or table["case_id"].eq("").any():
        raise ValueError(f"{source}: blank case_id")
    duplicate = table.loc[table["case_id"].duplicated(), "case_id"]
    if not duplicate.empty:
        raise ValueError(f"{source}: duplicate case_id {duplicate.tolist()}")


def boolean_column(values: pd.Series, source: str) -> pd.Series:
    parsed = values.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if parsed.isna().any():
        raise ValueError(f"{source}: expected True/False without missing values")
    return parsed.astype(bool)


def validate_gate(gate: pd.DataFrame) -> None:
    require_columns(
        gate,
        ("feature_id", "feature_name", "paper2_decision", "paper2_role",
         "paper2_eligible", "paper2_decision_basis"),
        "frozen gate",
    )
    if len(gate) != 20 or set(gate.feature_id) != {
        f"F{i:02d}" for i in range(1, 21)
    } or gate.feature_id.duplicated().any():
        raise ValueError("frozen gate: expected exactly F01-F20 once each")
    if not gate.paper2_decision_basis.eq("PAPER1_ONLY").all():
        raise ValueError("frozen gate: decision basis must be PAPER1_ONLY")
    eligible = boolean_column(gate.paper2_eligible, "frozen gate eligibility")
    expected = {
        "F01": ("INCLUDE", "PRIMARY", True),
        "F07": ("INCLUDE", "SECONDARY", True),
        "F09": ("INCLUDE", "SECONDARY", True),
        "F15": ("INCLUDE", "SECONDARY", True),
    }
    for index, row in gate.iterrows():
        wanted = expected.get(row.feature_id, ("EXCLUDE", "NONE", False))
        actual = (row.paper2_decision, row.paper2_role, bool(eligible.loc[index]))
        if actual != wanted:
            raise ValueError(f"frozen gate: {row.feature_id} has {actual}, expected {wanted}")
    for feature_id, name in FEATURE_NAMES.items():
        actual = gate.loc[gate.feature_id.eq(feature_id), "feature_name"].iloc[0]
        if actual != name:
            raise ValueError(f"frozen gate: {feature_id} feature definition mismatch")


def resolve_clinical_columns(base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    renamed = base.copy()
    mapping = {}
    for name in CLINICAL_COLUMNS:
        hits = [c for c in renamed.columns if str(c).split("\n", 1)[0] == name]
        if len(hits) != 1:
            raise ValueError(f"patient audit: expected one source column for {name}, got {hits}")
        mapping[name] = hits[0]
    renamed = renamed.rename(columns={source: name for name, source in mapping.items()})
    return renamed, mapping


def build_master(
    patient_audit: pd.DataFrame, patient_features: pd.DataFrame,
    gate: pd.DataFrame, video_audit: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    validate_gate(gate)
    base, clinical_mapping = resolve_clinical_columns(patient_audit)
    require_columns(base, ("case_id", "video_filenames", "clinical_pregnancy_binary_qc")
                    + STATUS_COLUMNS, "patient audit")
    base = base.copy()
    base["case_id"] = base.case_id.astype("string").str.strip().str.replace(
        r"\.0$", "", regex=True
    )
    if not base.case_id.str.fullmatch(r"\d+").fillna(False).all():
        raise ValueError("patient audit: case_id must be numeric IDs from script 03")
    unique_case_ids(base, "patient audit")
    base["matched_exam_date8"] = base.matched_exam_date8.astype("string").str.strip().str.replace(
        r"\.0$", "", regex=True
    )
    if not base.matched_exam_date8.str.fullmatch(r"20\d{6}").fillna(False).all():
        raise ValueError("patient audit: matched examination date must be YYYYMMDD")
    base["video_real_exam_date8"] = base.video_real_exam_date8.astype("string").str.strip().str.replace(
        r"\.0$", "", regex=True
    )

    audit = video_audit.copy()
    require_columns(audit, ("video_filename", "name_date", "ocr_date", "ocr_status"),
                    "video date audit")
    audit["source_case_id"] = audit.video_filename.map(lambda name: Path(str(name)).stem)
    if audit.source_case_id.duplicated().any():
        raise ValueError("video date audit: duplicate video filename stem")

    features = patient_features.copy()
    require_columns(features, ("case_id",) + tuple(gate.feature_name), "Paper 1 features")
    source_ids = features.case_id.astype("string").str.extract(r"^.+_(\d+)_(20\d{6})$")
    if source_ids.isna().any().any():
        raise ValueError("Paper 1 features: case_id must contain numeric ID and filename date")
    features["source_case_id"] = features.case_id
    features["case_id"] = source_ids[0]
    features["paper1_filename_date8"] = source_ids[1]
    unique_case_ids(features, "Paper 1 features")
    if not set(features.case_id).issubset(set(base.case_id)):
        raise ValueError("Paper 1 features: cases absent from patient audit")
    features = features.merge(
        audit[["source_case_id", "video_filename", "name_date", "ocr_date", "ocr_status"]],
        on="source_case_id", how="left", validate="one_to_one"
    )
    if features.video_filename.isna().any():
        raise ValueError("Paper 1 features: source video absent from video date audit")
    if not features.name_date.eq(features.paper1_filename_date8).all():
        raise ValueError("Paper 1 features: filename date disagrees with video date audit")
    if not features.ocr_status.isin({"UNIQUE_OCR_DATE", "KEYWORD_RESOLVED_DATE", "OK"}).all():
        raise ValueError("Paper 1 features: video exam date is not OCR-confirmed")
    base_by_id = base.set_index("case_id")
    for row in features.itertuples():
        patient = base_by_id.loc[row.case_id]
        if row.video_filename not in str(patient.video_filenames).split(" | "):
            raise ValueError(f"Paper 1 features: video identity disagrees for case_id {row.case_id}")
        if row.ocr_date != patient.video_real_exam_date8 or row.ocr_date != patient.matched_exam_date8:
            raise ValueError(f"Paper 1 features: confirmed exam date disagrees for case_id {row.case_id}")
        if patient.final_audit_status != "PASS_THREE_WAY":
            raise ValueError(f"Paper 1 features: case_id {row.case_id} is not three-way matched")

    selected = [FEATURE_NAMES[feature_id] for feature_id in FEATURE_IDS]
    features["paper1_video_exam_date8"] = features.ocr_date
    features = features.loc[:, ["case_id", "paper1_filename_date8",
                                "paper1_video_exam_date8", *selected]].rename(
        columns={FEATURE_NAMES[feature_id]: f"{feature_id}_{FEATURE_NAMES[feature_id]}"
                 for feature_id in FEATURE_IDS}
    )
    base["clinical_pregnancy"] = pd.to_numeric(
        base.clinical_pregnancy_binary_qc, errors="coerce"
    )
    if not base.clinical_pregnancy.dropna().isin([0, 1]).all():
        raise ValueError("patient audit: parsed clinical pregnancy must be 0/1")
    for name in ("date_audit_pass", "eligible_activity_association",
                 "eligible_pregnancy_association", "clinical_pregnancy_parseable_01"):
        base[name] = boolean_column(base[name], f"patient audit {name}")
    if not base.clinical_pregnancy.notna().equals(base.clinical_pregnancy_parseable_01):
        raise ValueError("patient audit: pregnancy parseability flag disagrees with parsed 0/1")
    if base.loc[base.eligible_pregnancy_association, "clinical_pregnancy"].isna().any():
        raise ValueError("patient audit: pregnancy eligibility includes missing outcome")
    activity_numeric = base[["peristalsis_forward_count", "peristalsis_reverse_count"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if activity_numeric.loc[base.eligible_activity_association].isna().all(axis=1).any():
        raise ValueError("patient audit: activity eligibility includes no recorded activity")
    base[["peristalsis_forward_count", "peristalsis_reverse_count"]] = activity_numeric

    output_columns = ["case_id", *STATUS_COLUMNS, *CLINICAL_COLUMNS, "clinical_pregnancy"]
    master = base.loc[:, output_columns].merge(
        features, on="case_id", how="left", validate="one_to_one", indicator=True
    )
    if len(master) != len(base) or set(master.case_id) != set(base.case_id):
        raise ValueError("merge changed the patient-level base cohort")
    master["paper1_feature_match_status"] = master.pop("_merge").map(
        {"both": "MATCHED", "left_only": "NO_PAPER1_FEATURE"}
    ).astype("string")
    master["paper1_filename_date_differs_from_exam_date"] = (
        master.paper1_filename_date8.notna()
        & master.paper1_filename_date8.ne(master.paper1_video_exam_date8)
    ).fillna(False)
    for feature_id in FEATURE_IDS:
        name = f"{feature_id}_{FEATURE_NAMES[feature_id]}"
        values = pd.to_numeric(master[name], errors="coerce")
        master[name] = values.where(np.isfinite(values))
        master[f"{feature_id}_available"] = master[name].notna()
        master[f"eligible_activity_{feature_id}"] = (
            master.eligible_activity_association & master[f"{feature_id}_available"]
        )
        master[f"eligible_pregnancy_{feature_id}"] = (
            master.eligible_pregnancy_association & master[f"{feature_id}_available"]
        )
    unique_case_ids(master, "analysis master")
    return master, clinical_mapping


def counts_table(master: pd.DataFrame) -> pd.DataFrame:
    counts = {
        "base_rows": len(master),
        "unique_case_ids": master.case_id.nunique(),
        "paper1_feature_matched": int(master.paper1_feature_match_status.eq("MATCHED").sum()),
        "paper1_filename_date_mismatch": int(
            master.paper1_filename_date_differs_from_exam_date.sum()
        ),
        "paper1_feature_source_missing": int(
            master.paper1_feature_match_status.eq("NO_PAPER1_FEATURE").sum()
        ),
        "activity_eligible": int(master.eligible_activity_association.sum()),
        "pregnancy_eligible": int(master.eligible_pregnancy_association.sum()),
    }
    for feature_id in FEATURE_IDS:
        counts[f"{feature_id}_available"] = int(master[f"{feature_id}_available"].sum())
        counts[f"activity_{feature_id}_eligible"] = int(
            master[f"eligible_activity_{feature_id}"].sum()
        )
        counts[f"pregnancy_{feature_id}_eligible"] = int(
            master[f"eligible_pregnancy_{feature_id}"].sum()
        )
    return pd.DataFrame(counts.items(), columns=["item", "n"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient-audit", type=Path, required=True)
    parser.add_argument("--video-date-audit", type=Path, required=True)
    parser.add_argument("--paper1-features", type=Path, required=True)
    parser.add_argument("--paper1-source-manifest", type=Path, required=True)
    parser.add_argument("--frozen-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads(args.paper1_source_manifest.read_text(encoding="utf-8"))
    expected_hash = source_manifest["output_files"][args.paper1_features.name]
    if sha256(args.paper1_features) != expected_hash:
        raise ValueError("Paper 1 feature table SHA256 differs from source manifest")
    patient_audit = pd.read_excel(
        args.patient_audit, sheet_name=PATIENT_SHEET, engine="openpyxl", dtype={"case_id": "string"}
    )
    patient_features = pd.read_csv(args.paper1_features, encoding="utf-8-sig", dtype={"case_id": "string"})
    if len(patient_features) != source_manifest["final_main_analysis_count"]:
        raise ValueError("Paper 1 feature row count differs from source manifest")
    gate = pd.read_csv(args.frozen_gate, encoding="utf-8-sig", dtype=str)
    video_audit = pd.read_csv(args.video_date_audit, encoding="utf-8-sig", dtype="string")
    master, clinical_mapping = build_master(patient_audit, patient_features, gate, video_audit)
    counts = counts_table(master)

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    output.mkdir(parents=True)
    master_path = output / "paper2_analysis_master.csv"
    xlsx_path = output / "paper2_analysis_master.xlsx"
    counts_path = output / "cohort_counts.csv"
    manifest_path = output / "manifest.json"
    report_path = output / "REPORT.md"
    master.to_csv(master_path, index=False, encoding="utf-8-sig", na_rep="NA")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="analysis_master", index=False)
        counts.to_excel(writer, sheet_name="cohort_counts", index=False)
    counts.to_csv(counts_path, index=False, encoding="utf-8-sig")
    inputs = {
        "patient_audit": args.patient_audit.resolve(),
        "video_date_audit": args.video_date_audit.resolve(),
        "paper1_features": args.paper1_features.resolve(),
        "paper1_source_manifest": args.paper1_source_manifest.resolve(),
        "frozen_gate": args.frozen_gate.resolve(),
    }
    output_files = {"master_csv": master_path, "master_xlsx": xlsx_path, "cohort_counts": counts_path}
    manifest = {
        "status": "paper2_analysis_master_created",
        "input_paths": {key: str(path) for key, path in inputs.items()},
        "input_sha256": {key: sha256(path) for key, path in inputs.items()},
        "frozen_gate_sha256": sha256(args.frozen_gate),
        "paper1_patient_level_feature_source_sha256": sha256(args.paper1_features),
        "output_sha256": {key: sha256(path) for key, path in output_files.items()},
        "clinical_source_columns": clinical_mapping,
        "eligible_feature_ids": list(FEATURE_IDS),
        "primary": "F01",
        "secondary": ["F07", "F09", "F15"],
        "decision_basis": "PAPER1_ONLY",
        "outcome_selection_used": False,
        "statistical_testing_performed": False,
        "cohort_counts": dict(zip(counts.item, counts.n.astype(int))),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    n = manifest["cohort_counts"]
    report_path.write_text(
        "# Paper 2 patient-level analysis master\n\n"
        f"Base: {n['base_rows']} rows and {n['unique_case_ids']} unique case IDs. "
        f"Paper 1 video-verified feature matches: {n['paper1_feature_matched']}; "
        f"no Paper 1 source row: {n['paper1_feature_source_missing']}. "
        f"Filename dates different from OCR-confirmed exam dates: "
        f"{n['paper1_filename_date_mismatch']}; these were retained after "
        "video identity and confirmed date checks.\n\n"
        f"Inherited activity eligibility: {n['activity_eligible']}; "
        f"inherited pregnancy eligibility: {n['pregnancy_eligible']}.\n\n"
        + "| Feature | Available | Activity eligible | Pregnancy eligible |\n"
        + "|---|---:|---:|---:|\n"
        + "".join(
            f"| {feature_id} | {n[f'{feature_id}_available']} | "
            f"{n[f'activity_{feature_id}_eligible']} | "
            f"{n[f'pregnancy_{feature_id}_eligible']} |\n"
            for feature_id in FEATURE_IDS
        )
        + f"\nF15 missing: {len(master) - n['F15_available']}; "
        "missing measurements were not imputed or used to remove base rows. "
        "Feature-specific eligibility uses only the corresponding measurement. "
        "Zero clinician-recorded forward/reverse activity counts remain valid observations. "
        "Clinical pregnancy is the existing parsed 0/1 field; unparseable values remain missing.\n\n"
        "No clinical association, statistical test, regression, AUC or outcome-based feature selection was performed.\n",
        encoding="utf-8",
    )
    print(counts.to_string(index=False))
    print(f"\nCOMPLETE: {output}")


if __name__ == "__main__":
    main()
