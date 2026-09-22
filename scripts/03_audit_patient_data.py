#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from activity_count_io import read_excel_preserving_activity
except ModuleNotFoundError as exc:
    if exc.name != "activity_count_io":
        raise
    from scripts.activity_count_io import read_excel_preserving_activity

INPUT_SHEET = "01_全部病例三方核对"
PREDICTION_CANDIDATE_FIELDS = [
    "clinical_pregnancy",
    "female_age",
    "female_bmi",
    "endometrial_thickness_mm",
    "transfer_embryo_count",
    "embryo_type",
]


def clean_case_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def missing(s: pd.Series) -> pd.Series:
    t = s.astype("string").str.strip()
    return s.isna() | t.eq("") | t.str.lower().isin({"na", "n/a", "nan", "none", "<na>"})


def binary_pregnancy(value):
    if value is None or pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    pos = {"1", "1.0", "阳性", "是", "yes", "y", "positive", "pregnant"}
    neg = {"0", "0.0", "阴性", "否", "no", "n", "negative", "not pregnant"}
    if s in pos:
        return 1.0
    if s in neg:
        return 0.0
    try:
        f = float(s)
        return f if f in (0.0, 1.0) else np.nan
    except Exception:
        return np.nan


def summarize(df: pd.DataFrame, col: str) -> dict:
    m = missing(df[col])
    num = pd.to_numeric(df[col], errors="coerce")
    return {
        "variable": col,
        "n_total": len(df),
        "n_nonmissing": int((~m).sum()),
        "n_missing": int(m.sum()),
        "missing_pct": round(float(m.mean() * 100), 2),
        "n_unique_nonmissing": int(df.loc[~m, col].astype(str).nunique()),
        "numeric_n": int(num.notna().sum()),
        "min": float(num.min()) if num.notna().any() else np.nan,
        "median": float(num.median()) if num.notna().any() else np.nan,
        "max": float(num.max()) if num.notna().any() else np.nan,
        "n_zero": int((num == 0).sum()) if num.notna().any() else np.nan,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(src, engine="openpyxl")
    if INPUT_SHEET not in xls.sheet_names:
        raise ValueError(f"missing sheet: {INPUT_SHEET}")

    df = read_excel_preserving_activity(src, sheet_name=INPUT_SHEET)
    for c in ("case_id", "final_audit_status"):
        if c not in df.columns:
            raise ValueError(f"missing required column: {c}")

    df = df.copy()
    df["case_id"] = df["case_id"].map(clean_case_id)
    counts = df["case_id"].value_counts(dropna=False)
    df["n_rows_same_case_id"] = df["case_id"].map(counts)
    df["case_id_duplicate_flag"] = np.where(df["n_rows_same_case_id"] > 1, "DUPLICATE", "UNIQUE")
    df["date_audit_pass"] = df["final_audit_status"].eq("PASS_THREE_WAY")

    for c in ("peristalsis_forward_count", "peristalsis_reverse_count", "clinical_pregnancy"):
        if c not in df.columns:
            df[c] = pd.NA

    fwd_m = missing(df["peristalsis_forward_count"])
    rev_m = missing(df["peristalsis_reverse_count"])
    df["forward_activity_available"] = ~fwd_m
    df["reverse_activity_available"] = ~rev_m
    df["any_activity_available"] = (~fwd_m) | (~rev_m)
    df["both_activity_available"] = (~fwd_m) & (~rev_m)

    df["clinical_pregnancy_binary_qc"] = df["clinical_pregnancy"].map(binary_pregnancy)
    df["clinical_pregnancy_parseable_01"] = df["clinical_pregnancy_binary_qc"].notna()

    base_ok = (
        df["date_audit_pass"]
        & df["case_id"].ne("")
        & df["case_id_duplicate_flag"].eq("UNIQUE")
    )
    df["eligible_activity_association"] = base_ok & df["any_activity_available"]
    df["eligible_activity_both_directions"] = base_ok & df["both_activity_available"]
    df["eligible_pregnancy_association"] = base_ok & df["clinical_pregnancy_parseable_01"]

    pred_fields = [c for c in PREDICTION_CANDIDATE_FIELDS if c in df.columns]
    complete = pd.Series(True, index=df.index)
    for c in pred_fields:
        complete &= df["clinical_pregnancy_parseable_01"] if c == "clinical_pregnancy" else ~missing(df[c])
    df["prediction_candidate_complete_case"] = base_ok & complete

    clinical_cols = [
        c for c in [
            "clinical_pregnancy", "biochemical_pregnancy",
            "peristalsis_forward_count", "peristalsis_reverse_count",
            "female_age", "female_bmi", "endometrial_thickness_mm",
            "cycle_type", "treatment_protocol", "transfer_embryo_count",
            "embryo_type", "embryo_detail"
        ] if c in df.columns
    ]
    variable_summary = pd.DataFrame([summarize(df, c) for c in clinical_cols])

    activity_cohort = df.loc[df["eligible_activity_association"]].copy()
    both_cohort = df.loc[df["eligible_activity_both_directions"]].copy()
    pregnancy_cohort = df.loc[df["eligible_pregnancy_association"]].copy()
    prediction_cohort = df.loc[df["prediction_candidate_complete_case"]].copy()
    duplicate_cases = df.loc[df["case_id_duplicate_flag"].eq("DUPLICATE")].copy()

    summary = pd.DataFrame([
        ("input_rows", len(df)),
        ("unique_case_id", df["case_id"].replace("", pd.NA).nunique()),
        ("PASS_THREE_WAY", int(df["date_audit_pass"].sum())),
        ("forward_activity_available", int(df["forward_activity_available"].sum())),
        ("reverse_activity_available", int(df["reverse_activity_available"].sum())),
        ("both_activity_available", int(df["both_activity_available"].sum())),
        ("clinical_pregnancy_parseable_01", int(df["clinical_pregnancy_parseable_01"].sum())),
        ("eligible_activity_association", len(activity_cohort)),
        ("eligible_pregnancy_association", len(pregnancy_cohort)),
        ("prediction_candidate_complete_case", len(prediction_cohort)),
    ], columns=["item", "n"])

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="01_总体汇总", index=False)
        df.to_excel(writer, sheet_name="02_患者级状态", index=False)
        activity_cohort.to_excel(writer, sheet_name="03_活动关联候选队列", index=False)
        both_cohort.to_excel(writer, sheet_name="04_双方向活动完整", index=False)
        pregnancy_cohort.to_excel(writer, sheet_name="05_妊娠关联候选队列", index=False)
        prediction_cohort.to_excel(writer, sheet_name="06_预测候选完整病例", index=False)
        variable_summary.to_excel(writer, sheet_name="07_临床变量完整性", index=False)
        duplicate_cases.to_excel(writer, sheet_name="08_case重复审查", index=False)

    print(summary.to_string(index=False))
    print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
