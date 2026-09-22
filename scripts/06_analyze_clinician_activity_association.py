#!/usr/bin/env python
"""Prespecified, unadjusted Paper 2 imaging/activity associations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


FEATURES = {
    "F01": "F01_rsr_abs_median",
    "F07": "F07_cavity_width_strain_rate_abs_median",
    "F09": "F09_longitudinal_wall_strain_rate_abs_median",
    "F15": "F15_wall_curvature_change_rate_mm_inv_s_abs_median",
}
DIRECTIONS = {
    "forward": "peristalsis_forward_count",
    "reverse": "peristalsis_reverse_count",
}
COMPARISONS = tuple((feature, direction) for feature in FEATURES for direction in DIRECTIONS)
PERMUTATIONS = 9999
BOOTSTRAPS = 5000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flags(series: pd.Series, name: str) -> pd.Series:
    values = series.astype("string").str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    if values.isna().any():
        raise ValueError(f"{name}: expected nonmissing Boolean values")
    return values.astype(bool)


def validate_master(master: pd.DataFrame, manifest: dict) -> None:
    if (manifest.get("eligible_feature_ids") != list(FEATURES)
            or manifest.get("primary") != "F01"
            or manifest.get("secondary") != ["F07", "F09", "F15"]
            or manifest.get("decision_basis") != "PAPER1_ONLY"):
        raise ValueError("master manifest: frozen feature contract mismatch")
    required = {"case_id", *FEATURES.values(), *DIRECTIONS.values(),
                *(f"eligible_activity_{feature}" for feature in FEATURES)}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"master: missing columns {sorted(missing)}")
    if master.case_id.isna().any() or master.case_id.eq("").any() or master.case_id.duplicated().any():
        raise ValueError("master: blank or duplicate case_id")
    counts = manifest.get("cohort_counts", {})
    if len(master) != counts.get("base_rows"):
        raise ValueError("master: base cohort differs from source manifest")
    for feature, column in FEATURES.items():
        eligible = flags(master[f"eligible_activity_{feature}"], f"eligible_activity_{feature}")
        measurement = pd.to_numeric(master[column], errors="coerce")
        if eligible.sum() != counts.get(f"activity_{feature}_eligible"):
            raise ValueError(f"{feature}: eligibility count differs from source manifest")
        if (eligible & ~np.isfinite(measurement)).any():
            raise ValueError(f"{feature}: eligible patient has nonfinite measurement")
    for direction, column in DIRECTIONS.items():
        values = pd.to_numeric(master[column], errors="coerce")
        bad = values.notna() & (~np.isfinite(values) | (values < 0) | (values != np.floor(values)))
        if bad.any():
            raise ValueError(f"{direction}: activity count must be a nonnegative integer")
        unparsed = master[column].notna() & values.isna()
        if unparsed.any():
            raise ValueError(f"{direction}: unparseable activity count")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def association(x: np.ndarray, y: np.ndarray, index: int) -> dict:
    n = len(x)
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return {"status": "NOT_EVALUABLE", "rho": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "raw_p": np.nan, "bootstrap_valid": 0}
    rx, ry = rankdata(x), rankdata(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    rho = float(np.dot(rx, ry) / denominator)
    rng = np.random.default_rng(20260922 + index)
    extreme = 0
    for _ in range(PERMUTATIONS):
        permuted = float(np.dot(rx[rng.permutation(n)], ry) / denominator)
        extreme += abs(permuted) >= abs(rho) - 1e-12
    raw_p = (extreme + 1) / (PERMUTATIONS + 1)

    rng = np.random.default_rng(20270922 + index)
    estimates = []
    for _ in range(BOOTSTRAPS):
        sample = rng.integers(0, n, n)
        if np.unique(x[sample]).size > 1 and np.unique(y[sample]).size > 1:
            estimates.append(spearman(x[sample], y[sample]))
    lower, upper = (np.percentile(estimates, [2.5, 97.5]) if estimates
                    else (np.nan, np.nan))
    return {"status": "EVALUABLE", "rho": rho, "ci_lower": lower,
            "ci_upper": upper, "raw_p": raw_p, "bootstrap_valid": len(estimates)}


def holm_with_planned_family(values: pd.Series) -> np.ndarray:
    """Preserve planned family size when a test cannot be evaluated."""
    p = values.fillna(1).to_numpy(dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[position]))
        adjusted[position] = running
    adjusted[values.isna().to_numpy()] = np.nan
    return adjusted


def analyze(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (feature, direction) in enumerate(COMPARISONS):
        eligibility = flags(master[f"eligible_activity_{feature}"], f"eligible_activity_{feature}")
        x = pd.to_numeric(master[FEATURES[feature]], errors="coerce")
        y = pd.to_numeric(master[DIRECTIONS[direction]], errors="coerce")
        selected = eligibility & np.isfinite(x) & np.isfinite(y)
        result = association(x[selected].to_numpy(dtype=float),
                             y[selected].to_numpy(dtype=float), index)
        rows.append({"family": "PRIMARY" if feature == "F01" else "SECONDARY",
                     "feature_id": feature, "measurement": FEATURES[feature],
                     "activity_direction": direction, "activity_source_column": DIRECTIONS[direction],
                     "n": int(selected.sum()), "eligible_flag_n": int(eligibility.sum()),
                     "zero_count_n": int((y[selected] == 0).sum()), **result})
    results = pd.DataFrame(rows)
    for family in ("PRIMARY", "SECONDARY"):
        positions = results.family.eq(family)
        results.loc[positions, "holm_adjusted_p"] = holm_with_planned_family(
            results.loc[positions, "raw_p"]
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--master-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError(f"output already exists: {args.output}")
    manifest = json.loads(args.master_manifest.read_text(encoding="utf-8"))
    if sha256(args.master) != manifest.get("output_sha256", {}).get("master_csv"):
        raise ValueError("master CSV SHA256 differs from source manifest")
    gate_path = Path(manifest["input_paths"]["frozen_gate"])
    if sha256(gate_path) != manifest["frozen_gate_sha256"]:
        raise ValueError("frozen gate SHA256 differs from source manifest")
    master = pd.read_csv(args.master, dtype={"case_id": "string"}, low_memory=False)
    validate_master(master, manifest)
    results = analyze(master)
    args.output.mkdir(parents=True)
    result_path = args.output / "paper2_activity_associations.csv"
    results.to_csv(result_path, index=False)
    output_manifest = {
        "status": "paper2_activity_association_complete",
        "input_paths": {"master": str(args.master), "master_manifest": str(args.master_manifest),
                        "frozen_gate": str(gate_path)},
        "input_sha256": {"master": sha256(args.master),
                         "master_manifest": sha256(args.master_manifest),
                         "frozen_gate": sha256(gate_path)},
        "output_sha256": {"associations": sha256(result_path)},
        "primary": ["F01-forward", "F01-reverse"],
        "secondary": [f"{f}-{d}" for f, d in COMPARISONS if f != "F01"],
        "method": "Spearman average-rank correlation",
        "raw_p": "two-sided pairing permutation, plus-one Monte Carlo correction",
        "permutations": PERMUTATIONS, "permutation_seed_base": 20260922,
        "ci": "95% paired-patient bootstrap percentile",
        "bootstraps": BOOTSTRAPS, "bootstrap_seed_base": 20270922,
        "multiplicity": "Holm separately in planned primary (2) and secondary (6) families",
        "clinical_covariate_adjustment": False, "pregnancy_outcome_used": False,
        "prediction_performed": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = ["# Paper 2 clinician-recorded activity associations", "",
             "Prespecified unadjusted Spearman associations; these are not agreement, causal, or prediction analyses.",
             "Primary (2) and secondary (6) families have separate Holm correction.",
             "Each comparison uses its own feature-specific eligible patients; zero counts are retained.", "",
             "| Family | Comparison | N | rho | 95% CI | raw P | Holm P | Status |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for row in results.itertuples():
        ci = f"[{row.ci_lower:.3f}, {row.ci_upper:.3f}]" if np.isfinite(row.ci_lower) else "NA"
        lines.append(f"| {row.family} | {row.feature_id}-{row.activity_direction} | {row.n} | "
                     f"{row.rho:.3f} | {ci} | {row.raw_p:.4g} | {row.holm_adjusted_p:.4g} | {row.status} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
