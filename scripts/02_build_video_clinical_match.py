#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PATIENT_SHEET = "患者信息提取"
QC_SHEET = "匹配质控"
CASE_RE = re.compile(r"(?:^|[_-])(\d+)[_-](20\d{6})(?:\D|$)")


def clean_case_id(x) -> str:
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except Exception:
        pass
    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def date8(x) -> str:
    if x is None or pd.isna(x) or str(x).strip() == "":
        return ""
    s = str(x).strip()
    m = re.search(r"(?<!\d)(20\d{6})(?!\d)", s)
    if m:
        try:
            return pd.to_datetime(m.group(1), format="%Y%m%d").strftime("%Y%m%d")
        except Exception:
            pass
    d = pd.to_datetime(x, errors="coerce")
    return "" if pd.isna(d) else d.strftime("%Y%m%d")


def show_date(x: str) -> str:
    return f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 else ""


def by_prefix(df: pd.DataFrame, prefix: str) -> str:
    hits = [c for c in df.columns if str(c).split("\n", 1)[0] == prefix]
    if len(hits) != 1:
        raise ValueError(f"column prefix {prefix!r}: expected 1 match, got {hits}")
    return hits[0]


def parse_video_identity(row):
    for c in ("patient_folder", "video_filename", "video_relpath"):
        m = CASE_RE.search(str(row.get(c, "") or ""))
        if m:
            return clean_case_id(m.group(1)), m.group(2), c
    return "", date8(row.get("name_date", "")), "UNRESOLVED"


def real_video_date(row):
    od = date8(row.get("ocr_date", ""))
    status = str(row.get("ocr_status", "") or "")
    if od and status in {"UNIQUE_OCR_DATE", "KEYWORD_RESOLVED_DATE", "OK"}:
        return od, "CONFIRMED"
    if od:
        return od, "REVIEW_OCR_STATUS"
    return "", "REVIEW_NO_VIDEO_DATE"


def read_clinical(path: Path):
    patient = pd.read_excel(path, sheet_name=PATIENT_SHEET, engine="openpyxl")
    qc = pd.read_excel(path, sheet_name=QC_SHEET, engine="openpyxl")

    rename_patient = {}
    for prefix, short in [
        ("case_id", "case_id"),
        ("patient_name", "patient_name"),
        ("patient_id", "patient_id"),
        ("clinical_pregnancy_raw", "clinical_pregnancy"),
        ("biochemical_pregnancy_raw", "biochemical_pregnancy"),
        ("female_age", "female_age"),
        ("female_bmi", "female_bmi"),
        ("cycle_type", "cycle_type"),
        ("treatment_protocol", "treatment_protocol"),
        ("transfer_embryo_count", "transfer_embryo_count"),
        ("embryo_detail", "embryo_detail"),
        ("embryo_type", "embryo_type"),
        ("endometrial_thickness_mm", "endometrial_thickness_mm"),
        ("endometrial_type", "endometrial_type"),
        ("peristalsis_forward_count", "peristalsis_forward_count"),
        ("peristalsis_reverse_count", "peristalsis_reverse_count"),
        ("diag_all", "diag_all"),
    ]:
        try:
            rename_patient[by_prefix(patient, prefix)] = short
        except ValueError:
            if prefix in {"case_id", "patient_name", "patient_id"}:
                raise

    patient = patient.rename(columns=rename_patient)
    qc = qc.rename(columns={
        "case_id": "case_id",
        "image_patient_id": "image_patient_id",
        "image_datetime": "image_datetime",
        "match_status": "match_status",
        "candidate_count": "candidate_count",
        "matched_exam_datetime": "matched_exam_datetime",
        "matched_main_name": "matched_main_name",
        "matched_clinical_name": "matched_clinical_name",
    })

    patient["case_id"] = patient["case_id"].map(clean_case_id)
    qc["case_id"] = qc["case_id"].map(clean_case_id)
    patient["patient_id_date8"] = patient["patient_id"].map(date8)
    qc["image_date8"] = qc["image_datetime"].map(date8)
    qc["matched_exam_date8"] = qc["matched_exam_datetime"].map(date8)
    return patient, qc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-audit", required=True)
    p.add_argument("--clinical-workbook", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    v = pd.read_csv(args.video_audit, encoding="utf-8-sig", dtype=str).fillna("")
    parsed = v.apply(parse_video_identity, axis=1, result_type="expand")
    parsed.columns = ["case_id", "filename_date8", "case_id_source"]
    v = pd.concat([v, parsed], axis=1)
    rd = v.apply(real_video_date, axis=1, result_type="expand")
    rd.columns = ["video_real_exam_date8", "video_real_date_status"]
    v = pd.concat([v, rd], axis=1)

    # one row per case_id; conflicting dates are review, not auto-resolved
    case_rows = []
    for cid, g in v.groupby("case_id", dropna=False):
        ds = sorted(set(x for x in g["video_real_exam_date8"] if x))
        confirmed = sorted(set(
            x for x in g.loc[g["video_real_date_status"].eq("CONFIRMED"), "video_real_exam_date8"] if x
        ))
        if cid and len(ds) == 1 and confirmed == ds:
            status, final = "CONFIRMED", ds[0]
        elif not cid:
            status, final = "REVIEW_NO_CASE_ID", ""
        elif len(ds) == 0:
            status, final = "REVIEW_NO_VIDEO_DATE", ""
        elif len(ds) > 1:
            status, final = "REVIEW_CONFLICTING_VIDEO_DATES", ""
        else:
            status, final = "REVIEW_VIDEO_DATE_NOT_FULLY_CONFIRMED", ds[0]
        case_rows.append({
            "case_id": clean_case_id(cid),
            "video_real_exam_date8": final,
            "video_real_exam_date": show_date(final),
            "video_case_date_status": status,
            "n_videos": len(g),
            "video_filenames": " | ".join(g["video_filename"].tolist()),
        })
    case_video = pd.DataFrame(case_rows)

    patient, qc = read_clinical(Path(args.clinical_workbook))
    pcount = patient["case_id"].value_counts()
    qcount = qc["case_id"].value_counts()

    rows = []
    for _, r in case_video.iterrows():
        cid = r["case_id"]
        base = r.to_dict()
        ps = patient.loc[patient["case_id"].eq(cid)]
        qs = qc.loc[qc["case_id"].eq(cid)]
        base["patient_match_status"] = (
            "MATCH_UNIQUE" if len(ps) == 1 else
            "NO_MATCH_PATIENT_SHEET" if len(ps) == 0 else
            "REVIEW_MULTIPLE_PATIENT_ROWS"
        )
        base["qc_match_status"] = (
            "MATCH_UNIQUE" if len(qs) == 1 else
            "NO_MATCH_QC_SHEET" if len(qs) == 0 else
            "REVIEW_MULTIPLE_QC_ROWS"
        )
        if len(ps) == 1:
            for c, x in ps.iloc[0].items():
                if c != "case_id":
                    base[c] = x
        if len(qs) == 1:
            for c, x in qs.iloc[0].items():
                if c != "case_id":
                    base[c] = x

        vv = str(base.get("video_real_exam_date8", "") or "")
        ii = str(base.get("image_date8", "") or "")
        cc = str(base.get("matched_exam_date8", "") or "")
        pp = str(base.get("patient_id_date8", "") or "")

        base["video_vs_image_date"] = "MATCH" if vv and ii and vv == ii else "MISMATCH" if vv and ii else "NA"
        base["video_vs_matched_exam_date"] = "MATCH" if vv and cc and vv == cc else "MISMATCH" if vv and cc else "NA"
        base["video_vs_patient_id_date"] = "MATCH" if vv and pp and vv == pp else "MISMATCH" if vv and pp else "NA"

        if base["video_case_date_status"] != "CONFIRMED":
            final = "REVIEW_VIDEO_DATE"
        elif base["patient_match_status"] != "MATCH_UNIQUE":
            final = base["patient_match_status"]
        elif base["qc_match_status"] != "MATCH_UNIQUE":
            final = base["qc_match_status"]
        elif not cc:
            final = "REVIEW_NO_MATCHED_EXAM_DATE"
        elif vv != cc:
            final = "FAIL_VIDEO_VS_CLINICAL_DATE"
        elif ii and vv != ii:
            final = "REVIEW_IMAGE_DATETIME_DIFFERS"
        elif pp and vv != pp:
            final = "REVIEW_PATIENT_ID_DATE_DIFFERS"
        else:
            final = "PASS_THREE_WAY"
        base["final_audit_status"] = final
        base["image_date"] = show_date(ii)
        base["matched_exam_date"] = show_date(cc)
        base["patient_id_date"] = show_date(pp)
        rows.append(base)

    audit = pd.DataFrame(rows)
    passed = audit.loc[audit["final_audit_status"].eq("PASS_THREE_WAY")]
    review = audit.loc[~audit["final_audit_status"].eq("PASS_THREE_WAY")]
    summary = audit["final_audit_status"].value_counts().rename_axis("status").reset_index(name="n")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        audit.to_excel(w, sheet_name="01_全部病例三方核对", index=False)
        passed.to_excel(w, sheet_name="02_PASS病例", index=False)
        review.to_excel(w, sheet_name="03_待人工复核", index=False)
        case_video.to_excel(w, sheet_name="04_视频真实日期", index=False)
        summary.to_excel(w, sheet_name="05_汇总", index=False)

    audit.to_csv(out.with_name(out.stem + "_三方核对.csv"), index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\n[DONE] {out}")


if __name__ == "__main__":
    main()
