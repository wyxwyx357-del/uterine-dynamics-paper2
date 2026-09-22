#!/usr/bin/env python
"""Frozen-feature associations with clinical pregnancy; no prediction analysis."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import linprog
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning, PerfectSeparationError, PerfectSeparationWarning,
)


FEATURES = {
    "F01": "F01_rsr_abs_median",
    "F07": "F07_cavity_width_strain_rate_abs_median",
    "F09": "F09_longitudinal_wall_strain_rate_abs_median",
    "F15": "F15_wall_curvature_change_rate_mm_inv_s_abs_median",
}
NUMERIC_COVARIATES = (
    "female_age", "female_bmi", "infertility_years", "transfer_embryo_count",
    "endometrial_thickness_mm",
)
CATEGORICAL_COVARIATES = (
    "infertility_type", "embryo_type", "endometrial_type", "cycle_type",
)
MODELS = ("UNADJUSTED_FULL", "UNADJUSTED_MATCHED", "ADJUSTED")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flags(values: pd.Series, name: str) -> pd.Series:
    result = values.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if result.isna().any():
        raise ValueError(f"{name}: expected nonmissing Boolean values")
    return result.astype(bool)


def numeric(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce").astype(float)
    if ((values.notna() & parsed.isna()) | np.isinf(parsed)).any():
        raise ValueError(f"{name}: invalid numeric value")
    return parsed


def validate_master(master: pd.DataFrame, manifest: dict) -> dict[str, pd.Series]:
    if (manifest.get("eligible_feature_ids") != list(FEATURES)
            or manifest.get("primary") != "F01"
            or manifest.get("secondary") != ["F07", "F09", "F15"]
            or manifest.get("decision_basis") != "PAPER1_ONLY"):
        raise ValueError("master manifest: frozen feature contract mismatch")
    required = {"case_id", "clinical_pregnancy", "clinical_pregnancy_parseable_01",
                "eligible_pregnancy_association", *FEATURES.values(),
                *NUMERIC_COVARIATES, *CATEGORICAL_COVARIATES,
                *(f"eligible_pregnancy_{f}" for f in FEATURES)}
    if missing := required - set(master.columns):
        raise ValueError(f"master: missing columns {sorted(missing)}")
    case_id = master.case_id.astype("string").str.strip()
    if case_id.isna().any() or case_id.eq("").any() or case_id.duplicated().any():
        raise ValueError("master: blank or duplicate case_id")
    counts = manifest.get("cohort_counts", {})
    if len(master) != counts.get("base_rows"):
        raise ValueError("master: base cohort differs from manifest")
    outcome = numeric(master.clinical_pregnancy, "clinical_pregnancy")
    if not outcome.dropna().isin({0, 1}).all():
        raise ValueError("clinical_pregnancy: expected 0/1 or missing")
    parseable = flags(master.clinical_pregnancy_parseable_01, "clinical_pregnancy_parseable_01")
    if not parseable.equals(outcome.notna()):
        raise ValueError("clinical_pregnancy: parseability flag disagrees with outcome")
    eligible_base = flags(master.eligible_pregnancy_association, "eligible_pregnancy_association")
    if (eligible_base & outcome.isna()).any() or int(eligible_base.sum()) != counts.get("pregnancy_eligible"):
        raise ValueError("master: pregnancy base eligibility disagrees with outcome or manifest")
    for name in NUMERIC_COVARIATES:
        numeric(master[name], name)
    for name in CATEGORICAL_COVARIATES:
        text = master[name].astype("string").str.strip()
        if text.eq("").fillna(False).any():
            raise ValueError(f"{name}: blank category")
    result = {}
    for feature, column in FEATURES.items():
        measurement = numeric(master[column], column)
        eligible = flags(master[f"eligible_pregnancy_{feature}"], f"eligible_pregnancy_{feature}")
        if not eligible.equals(eligible_base & measurement.notna()):
            raise ValueError(f"{feature}: pregnancy eligibility disagrees with feature availability")
        if int(eligible.sum()) != counts.get(f"pregnancy_{feature}_eligible"):
            raise ValueError(f"{feature}: pregnancy eligibility count differs from manifest")
        result[feature] = eligible
    return result


def category_coding(master: pd.DataFrame) -> dict:
    coding = {}
    for name in CATEGORICAL_COVARIATES:
        levels = sorted(master[name].astype("string").str.strip().dropna().unique().tolist())
        if not levels:
            raise ValueError(f"{name}: no observed category")
        coding[name] = {"levels": levels, "reference": levels[0],
                        "dummy_columns": [f"{name}={level}" for level in levels[1:]]}
    return coding


def design(data: pd.DataFrame, feature: str, scale: float, coding: dict,
           adjusted: bool) -> pd.DataFrame:
    measurement = numeric(data[FEATURES[feature]], FEATURES[feature])
    result = pd.DataFrame({"intercept": np.ones(len(data)),
                           "feature_per_iqr": (measurement.to_numpy() - measurement.mean()) / scale},
                          index=data.index)
    if adjusted:
        for name in NUMERIC_COVARIATES:
            values = numeric(data[name], name)
            sd = float(values.std(ddof=0))
            result[name] = ((values - values.mean()) / sd if sd > 0 else values * 0).to_numpy()
        for name in CATEGORICAL_COVARIATES:
            values = data[name].astype("string").str.strip()
            for level in coding[name]["levels"][1:]:
                result[f"{name}={level}"] = values.eq(level).to_numpy(dtype=float)
    return result.astype(float)


def separation_kind(x: np.ndarray, y: np.ndarray) -> str | None:
    """Find a nonzero weak separating direction, including quasi separation."""
    signed = x * (2 * y - 1)[:, None]
    solution = linprog(-signed.sum(axis=0), A_ub=-signed,
                       b_ub=np.zeros(len(y)), bounds=[(-1, 1)] * x.shape[1],
                       method="highs")
    if not solution.success:
        return "SEPARATION_CHECK_FAILED"
    margins = signed @ solution.x
    if margins.sum() > 1e-7 and margins.min() >= -1e-7:
        return "COMPLETE_SEPARATION" if margins.min() > 1e-7 else "QUASI_SEPARATION"
    return None


def fit_model(data: pd.DataFrame, feature: str, scale: float, coding: dict,
              model_name: str) -> tuple[dict, dict]:
    y = numeric(data.clinical_pregnancy, "clinical_pregnancy").to_numpy(dtype=float)
    adjusted = model_name == "ADJUSTED"
    row = {"feature_id": feature, "family": "PRIMARY" if feature == "F01" else "SECONDARY",
           "model": model_name, "n": len(data), "pregnancy_positive_n": int((y == 1).sum()),
           "pregnancy_negative_n": int((y == 0).sum()), "feature_iqr": scale,
           "or_per_iqr": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
           "raw_p": np.nan, "holm_adjusted_p": np.nan, "status": "NOT_EVALUABLE"}
    diagnostic = {"feature_id": feature, "model": model_name, "n": len(data),
                  "status": "NOT_EVALUABLE", "converged": False,
                  "design_columns": [], "design_rank": 0, "design_columns_n": 0,
                  "condition_number": np.nan, "sparse_categories": {}, "category_outcome_counts": {}}
    if len(data) == 0 or len(set(y)) < 2 or not np.isfinite(scale) or scale <= 0:
        row["status"] = diagnostic["status"] = "INSUFFICIENT_OUTCOME_OR_FEATURE_VARIATION"
        return row, diagnostic
    x = design(data, feature, scale, coding, adjusted)
    diagnostic["design_columns"] = x.columns.tolist()
    diagnostic["design_columns_n"] = len(x.columns)
    diagnostic["design_rank"] = int(np.linalg.matrix_rank(x.to_numpy()))
    diagnostic["condition_number"] = float(np.linalg.cond(x.to_numpy()))
    if adjusted:
        if any(len(coding[name]["levels"]) < 2 for name in CATEGORICAL_COVARIATES):
            row["status"] = diagnostic["status"] = "CONSTANT_CATEGORICAL_COVARIATE"
            return row, diagnostic
        for name in CATEGORICAL_COVARIATES:
            counts = data.groupby(data[name].astype("string"), dropna=False).size().to_dict()
            diagnostic["sparse_categories"][name] = {str(k): int(v) for k, v in counts.items() if v < 5}
            diagnostic["category_outcome_counts"][name] = {
                str(level): {"positive": int(((data[name].astype("string") == level) & (y == 1)).sum()),
                             "negative": int(((data[name].astype("string") == level) & (y == 0)).sum())}
                for level in coding[name]["levels"]}
    if diagnostic["design_rank"] < len(x.columns):
        row["status"] = diagnostic["status"] = "RANK_DEFICIENT"
        return row, diagnostic
    problem = separation_kind(x.to_numpy(), y)
    if problem:
        row["status"] = diagnostic["status"] = problem
        return row, diagnostic
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = sm.Logit(y, x).fit(disp=False, maxiter=200)
        if (not fit.mle_retvals.get("converged", False)
                or any(issubclass(w.category, (ConvergenceWarning, PerfectSeparationWarning)) for w in caught)):
            row["status"] = diagnostic["status"] = "NONCONVERGED"
            return row, diagnostic
        beta = float(fit.params["feature_per_iqr"])
        lower, upper = fit.conf_int().loc["feature_per_iqr"]
        p = float(fit.pvalues["feature_per_iqr"])
        if not np.isfinite([beta, lower, upper, p]).all():
            row["status"] = diagnostic["status"] = "NONFINITE_ESTIMATE"
            return row, diagnostic
        row.update({"or_per_iqr": float(np.exp(beta)), "ci_lower": float(np.exp(lower)),
                    "ci_upper": float(np.exp(upper)), "raw_p": p, "status": "EVALUABLE"})
        if not np.isfinite([row["or_per_iqr"], row["ci_lower"], row["ci_upper"]]).all():
            row.update({"or_per_iqr": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                        "raw_p": np.nan, "status": "NONFINITE_ESTIMATE"})
        diagnostic["status"] = row["status"]
        diagnostic["converged"] = True
    except (ValueError, np.linalg.LinAlgError, OverflowError, PerfectSeparationError) as exc:
        row["status"] = diagnostic["status"] = "FIT_FAILED"
        diagnostic["failure_detail"] = str(exc)
    return row, diagnostic


def holm_secondary(results: pd.DataFrame) -> pd.DataFrame:
    selected = results.model.eq("ADJUSTED") & results.family.eq("SECONDARY")
    p = results.loc[selected, "raw_p"].fillna(1).to_numpy(dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[position]))
        adjusted[position] = running
    adjusted[results.loc[selected, "raw_p"].isna().to_numpy()] = np.nan
    results.loc[selected, "holm_adjusted_p"] = adjusted
    return results


def analyze(master: pd.DataFrame, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eligibility = validate_master(master, manifest)
    coding = category_coding(master)
    rows, diagnostics, flows = [], [], []
    for feature, column in FEATURES.items():
        eligible = master.loc[eligibility[feature]].copy()
        covariate_missing = pd.DataFrame(index=eligible.index)
        for name in NUMERIC_COVARIATES:
            covariate_missing[name] = numeric(eligible[name], name).isna()
        for name in CATEGORICAL_COVARIATES:
            covariate_missing[name] = eligible[name].isna()
        complete = eligible.loc[~covariate_missing.any(axis=1)].copy()
        values = numeric(eligible[column], column)
        scale = float(values.quantile(.75) - values.quantile(.25))
        flows.append({"feature_id": feature, "base_n": len(master),
                      "base_outcome_positive_n": int((master.clinical_pregnancy == 1).sum()),
                      "base_outcome_negative_n": int((master.clinical_pregnancy == 0).sum()),
                      "base_outcome_missing_n": int(master.clinical_pregnancy.isna().sum()),
                      "pregnancy_base_eligible_n": int(flags(master.eligible_pregnancy_association,
                                                               "eligible_pregnancy_association").sum()),
                      "feature_eligible_n": len(eligible),
                      "feature_unavailable_among_pregnancy_base_n": int(flags(master.eligible_pregnancy_association,
                                                                               "eligible_pregnancy_association").sum()) - len(eligible),
                      "eligible_positive_n": int((eligible.clinical_pregnancy == 1).sum()),
                      "eligible_negative_n": int((eligible.clinical_pregnancy == 0).sum()),
                      "covariate_incomplete_n": int(covariate_missing.any(axis=1).sum()),
                      "adjusted_n": len(complete),
                      "adjusted_positive_n": int((complete.clinical_pregnancy == 1).sum()),
                      "adjusted_negative_n": int((complete.clinical_pregnancy == 0).sum()),
                      **{f"missing_{name}_n": int(covariate_missing[name].sum())
                         for name in (*NUMERIC_COVARIATES, *CATEGORICAL_COVARIATES)}})
        for model_name in MODELS:
            data = eligible if model_name == "UNADJUSTED_FULL" else complete
            row, diagnostic = fit_model(data, feature, scale, coding, model_name)
            rows.append(row)
            diagnostics.append(diagnostic)
    return holm_secondary(pd.DataFrame(rows)), pd.DataFrame(flows), {
        "category_coding": coding, "models": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    source_manifest = json.loads(args.master_manifest.read_text(encoding="utf-8"))
    if sha256(args.master) != source_manifest.get("output_sha256", {}).get("master_csv"):
        raise ValueError("master CSV SHA256 differs from script 05 manifest")
    gate = Path(source_manifest["input_paths"]["frozen_gate"])
    if sha256(gate) != source_manifest["frozen_gate_sha256"]:
        raise ValueError("frozen feature gate SHA256 differs from script 05 manifest")
    master = pd.read_csv(args.master, dtype={"case_id": "string"}, low_memory=False)
    results, flow, diagnostics = analyze(master, source_manifest)
    args.output.mkdir(parents=True)
    paths = {"cohort": args.output / "pregnancy_cohort_flow.csv",
             "associations": args.output / "pregnancy_associations.csv",
             "diagnostics": args.output / "model_diagnostics.json"}
    flow.to_csv(paths["cohort"], index=False, encoding="utf-8-sig")
    results.to_csv(paths["associations"], index=False, encoding="utf-8-sig", na_rep="NA")
    paths["diagnostics"].write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                          cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    output_manifest = {"status": "clinical_pregnancy_association_complete",
                       "code_revision": revision,
                       "input_paths": {"master": str(args.master.resolve()),
                                       "master_manifest": str(args.master_manifest.resolve()),
                                       "frozen_gate": str(gate.resolve())},
                       "input_sha256": {"master": sha256(args.master),
                                        "master_manifest": sha256(args.master_manifest),
                                        "frozen_gate": sha256(gate)},
                       "output_sha256": {name: sha256(path) for name, path in paths.items()},
                       "primary": "F01 adjusted association", "secondary": ["F07", "F09", "F15"],
                       "outcome": "clinical_pregnancy 0/1", "model": "logistic regression",
                       "feature_scale": "OR per feature-specific eligible-cohort IQR; same scale in all models",
                       "numeric_covariates": NUMERIC_COVARIATES,
                       "categorical_covariates": CATEGORICAL_COVARIATES,
                       "missing_data": "complete cases for adjusted and matched unadjusted models",
                       "secondary_multiplicity": "Holm for three adjusted secondary feature P values",
                       "prediction_performed": False}
    (args.output / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Paper 2 clinical pregnancy associations", "",
             "Logistic association only; ORs are per feature-specific eligible-cohort IQR. No causal or prediction claim.",
             "The adjusted F01 model is the primary inference. Failed models have no OR or P value.",
             "", "| Feature | Model | N | Positive | Negative | OR | 95% CI | Raw P | Holm P | Status |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|---|"]
    for row in results.itertuples():
        ci = f"[{row.ci_lower:.3g}, {row.ci_upper:.3g}]" if np.isfinite(row.ci_lower) else "NA"
        odds = f"{row.or_per_iqr:.3g}" if np.isfinite(row.or_per_iqr) else "NA"
        raw = f"{row.raw_p:.4g}" if np.isfinite(row.raw_p) else "NA"
        holm = f"{row.holm_adjusted_p:.4g}" if np.isfinite(row.holm_adjusted_p) else "NA"
        lines.append(f"| {row.feature_id} | {row.model} | {row.n} | {row.pregnancy_positive_n} | "
                     f"{row.pregnancy_negative_n} | {odds} | {ci} | {raw} | {holm} | {row.status} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(results[["feature_id", "model", "n", "status", "or_per_iqr", "raw_p", "holm_adjusted_p"]].to_string(index=False))
    print(f"COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
