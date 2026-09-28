"""Read-only sensitivity checks for the exploratory Paper 2 Doppler associations.

Prints aggregate results only. Patient identifiers and rows are never exported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = {
    "F09": "F09_longitudinal_wall_strain_rate_abs_median",
    "F15": "F15_wall_curvature_change_rate_mm_inv_s_abs_median",
}
DOPPLER = {"VI": "flow_vi", "VFI": "flow_vfi"}


def ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float).copy()


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def partial_rank(x: np.ndarray, y: np.ndarray, age: np.ndarray, thickness: np.ndarray) -> float:
    controls = np.column_stack((np.ones(len(x)), ranks(age), ranks(thickness)))
    rx = ranks(x) - controls @ np.linalg.lstsq(controls, ranks(x), rcond=None)[0]
    ry = ranks(y) - controls @ np.linalg.lstsq(controls, ranks(y), rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def read_linked(master_path: Path, audit_path: Path) -> pd.DataFrame:
    master = pd.read_csv(master_path, dtype={"case_id": "string"})
    excel = pd.ExcelFile(audit_path)
    patient_sheet = next(name for name in excel.sheet_names if name.startswith("02_"))
    audit = pd.read_excel(audit_path, sheet_name=patient_sheet, dtype={"case_id": "string"})
    doppler_columns = {name: next(col for col in audit if str(col).startswith(prefix))
                       for name, prefix in DOPPLER.items()}
    audit = audit[["case_id", "matched_exam_date8", *doppler_columns.values()]].copy()
    audit = audit.rename(columns={column: name for name, column in doppler_columns.items()})
    for table in (master, audit):
        table["case_id"] = (table.case_id.astype("string").str.strip()
                            .str.replace(r"\.0$", "", regex=True))
        if table.case_id.isna().any() or table.case_id.duplicated().any():
            raise ValueError("blank or duplicate case_id")
    joined = master.merge(audit, on="case_id", how="left", validate="one_to_one",
                          suffixes=("_master", "_clinical"))
    if joined.matched_exam_date8_clinical.isna().any():
        raise ValueError("unmatched clinical patient")
    mismatched = (joined.matched_exam_date8_master.astype("string")
                  != joined.matched_exam_date8_clinical.astype("string"))
    if mismatched.fillna(True).any():
        raise ValueError("patient examination date mismatch")
    return joined


def assess(table: pd.DataFrame, feature: str, doppler: str) -> dict:
    fields = [FEATURES[feature], doppler, "female_age", "endometrial_thickness_mm"]
    values = table[fields].apply(pd.to_numeric, errors="coerce")
    values = values.loc[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)]
    x, y, age, thick = (values[col].to_numpy(dtype=float) for col in fields)
    if len(x) < 10 or np.unique(x).size < 2 or np.unique(y).size < 2:
        raise ValueError(f"insufficient variation for {feature}-{doppler}")
    nonzero = y > 0
    tails = ((x >= np.quantile(x, 0.01)) & (x <= np.quantile(x, 0.99))
             & (y >= np.quantile(y, 0.01)) & (y <= np.quantile(y, 0.99)))
    leave_one_out = [spearman(np.delete(x, i), np.delete(y, i)) for i in range(len(x))]
    quartiles = pd.qcut(pd.Series(x).rank(method="first"), q=4, labels=False)
    medians = [float(np.median(y[quartiles.to_numpy() == q])) for q in range(4)]
    return {
        "feature": feature,
        "doppler": doppler,
        "n": len(x),
        "rho": spearman(x, y),
        "zero_doppler_n": int((~nonzero).sum()),
        "nonzero_n": int(nonzero.sum()),
        "nonzero_rho": spearman(x[nonzero], y[nonzero]),
        "one_percent_trim_n": int(tails.sum()),
        "one_percent_trim_rho": spearman(x[tails], y[tails]),
        "leave_one_out_rho_min": min(leave_one_out),
        "leave_one_out_rho_max": max(leave_one_out),
        "age_thickness_partial_rank_rho": partial_rank(x, y, age, thick),
        "doppler_median_by_feature_quartile": medians,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    linked = read_linked(args.master, args.audit)
    results = [assess(linked, feature, doppler)
               for feature in FEATURES for doppler in DOPPLER]
    for first, second, label in ((FEATURES["F09"], FEATURES["F15"], "F09_F15"),
                                 ("VI", "VFI", "VI_VFI")):
        values = linked[[first, second]].apply(pd.to_numeric, errors="coerce").dropna()
        print(json.dumps({"pair": label, "n": len(values),
                          "rho": spearman(values[first].to_numpy(), values[second].to_numpy())}))
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    years = linked.matched_exam_date8_master.astype("string").str.slice(0, 4)
    for year in sorted(years.dropna().unique()):
        subset = linked.loc[years.eq(year)]
        for feature in FEATURES:
            for doppler in DOPPLER:
                values = subset[[FEATURES[feature], doppler]].apply(
                    pd.to_numeric, errors="coerce"
                ).dropna()
                x, y = (values[col].to_numpy(dtype=float)
                        for col in (FEATURES[feature], doppler))
                if len(x) >= 10 and np.unique(x).size > 1 and np.unique(y).size > 1:
                    print(json.dumps({"year": year, "feature": feature,
                                      "doppler": doppler, "n": len(x),
                                      "rho": spearman(x, y)}))


if __name__ == "__main__":
    main()
