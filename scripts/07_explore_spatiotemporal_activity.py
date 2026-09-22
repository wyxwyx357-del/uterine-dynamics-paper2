#!/usr/bin/env python
"""Exploratory F01 profile heterogeneity vs clinician counts; leaves script 06 unchanged.

Consumes Paper 1's ORIGINAL five-bin normalized-time/normalized-position profiles,
not QC shadow profiles or population-level stability summaries. No retracking.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


_CORE_PATH = Path(__file__).with_name("06_analyze_clinician_activity_association.py")
_SPEC = importlib.util.spec_from_file_location("paper2_frozen_activity", _CORE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load frozen statistical implementation: {_CORE_PATH}")
core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(core)

FEATURE_ID = "F01"
FEATURE_NAME = "rsr_abs_median"
MASTER_FEATURE = "F01_rsr_abs_median"
BINS = 5
DOMAINS = ("temporal", "spatial")
INDICES = tuple((domain, direction) for domain in DOMAINS for direction in core.DIRECTIONS)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"case_id": "string"}, low_memory=False)


def validate_source_manifest(manifest: dict, domain: str) -> set[str]:
    if manifest.get("bins") != BINS or manifest.get("clinical_labels_used") is not False:
        raise ValueError(f"{domain}: require original outcome-blind Paper 1 five-bin source manifest")
    definition_key = "normalized_position_definition" if domain == "spatial" else None
    if definition_key and not manifest.get(definition_key):
        raise ValueError("spatial: missing frozen normalized-position definition")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases or any(not isinstance(x, str) or not x for x in cases):
        raise ValueError(f"{domain}: missing source case list")
    if len(cases) != len(set(cases)):
        raise ValueError(f"{domain}: repeated source cases")
    return set(cases)


def patient_ids(source_case: pd.Series, domain: str) -> pd.DataFrame:
    # Profile source is Paper 1's original VIDEO_STEM, not its OCR date.
    fields = source_case.astype("string").str.extract(r"^.+_(\d+)_(20\d{6})$")
    if fields.isna().any().any():
        raise ValueError(f"{domain}: Paper 1 case_id lacks ID/date suffix")
    return fields.set_axis(["case_id", "paper1_filename_date8"], axis=1)


def numeric(series: pd.Series, name: str, *, allow_missing: bool = False) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").astype(float)
    if ((series.notna() & result.isna()) | np.isinf(result)).any():
        raise ValueError(f"{name}: invalid numeric value")
    if not allow_missing and result.isna().any():
        raise ValueError(f"{name}: missing numeric value")
    return result


def original_profile_features(
    profile: pd.DataFrame,
    summary: pd.DataFrame,
    source_manifest: dict,
    master: pd.DataFrame,
    domain: str,
) -> pd.DataFrame:
    """Compute relative MAD of all five ORIGINAL F01 bin medians, one row/case.

    Undefined if any original bin is unavailable or the five-bin median is zero.
    This conservative complete-bin rule is fixed before examining count results.
    """
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain: {domain}")
    allowed = validate_source_manifest(source_manifest, domain)
    need_profile = {"case_id", "feature_id", "feature", "statistic", "bin_index",
                    "original", "original_valid_positions"}
    need_summary = {"case_id", "feature_id", "feature", "aggregate_original"}
    for table, required, name in ((profile, need_profile, "profile"), (summary, need_summary, "summary")):
        if missing := (required - set(table.columns)):
            raise ValueError(f"{domain} {name}: missing {sorted(missing)}")
        if table.case_id.isna().any() or not set(table.case_id.astype(str)).issubset(allowed):
            raise ValueError(f"{domain} {name}: cases not in source manifest")
    if set(profile.case_id.astype(str)) != allowed or set(summary.case_id.astype(str)) != allowed:
        raise ValueError(f"{domain}: source manifest cases differ from profile/summary")

    p = profile.loc[profile.feature_id.eq(FEATURE_ID)].copy()
    s = summary.loc[summary.feature_id.eq(FEATURE_ID)].copy()
    if len(p) != BINS * len(allowed) or len(s) != len(allowed):
        raise ValueError(f"{domain}: require one F01 summary and five F01 bins per source case")
    if not p.feature.eq(FEATURE_NAME).all() or not s.feature.eq(FEATURE_NAME).all():
        raise ValueError(f"{domain}: F01 source measurement definition mismatch")
    if not p.statistic.eq("median").all():
        raise ValueError(f"{domain}: F01 bins must be median statistics")
    p["bin_index"] = numeric(p.bin_index, f"{domain} bin_index")
    if not p.bin_index.isin(range(1, BINS + 1)).all():
        raise ValueError(f"{domain}: require original bins 1 to 5")
    if p.duplicated(["case_id", "bin_index"]).any() or s.case_id.duplicated().any():
        raise ValueError(f"{domain}: duplicate case/bin or summary")
    p["original"] = numeric(p.original, f"{domain} original", allow_missing=True)
    p["original_valid_positions"] = numeric(p.original_valid_positions, f"{domain} count")
    count = p.original_valid_positions
    if ((count < 0) | (count != np.floor(count))).any():
        raise ValueError(f"{domain}: original valid-position counts must be nonnegative integers")
    if ((count.eq(0) & p.original.notna()) | (count.gt(0) & p.original.isna())
            | p.original.dropna().lt(0).reindex(p.index, fill_value=False)).any():
        raise ValueError(f"{domain}: original bin value/count inconsistency")
    s["aggregate_original"] = numeric(s.aggregate_original, f"{domain} summary", allow_missing=True)
    s["case_id"] = s.case_id.astype("string")
    p["case_id"] = p.case_id.astype("string")

    # Join through BOTH numeric patient ID and original video filename date.
    ids = patient_ids(s.case_id, domain).reset_index(drop=True)
    s = s.reset_index(drop=True)
    s = s.rename(columns={"case_id": "source_case_id"})
    s["case_id"] = ids["case_id"].astype("string")
    s["paper1_filename_date8"] = ids["paper1_filename_date8"].astype("string")
    if s.case_id.duplicated().any():
        raise ValueError(f"{domain}: repeated patient IDs across source videos")
    matched = s.merge(
        master[["case_id", "paper1_filename_date8", MASTER_FEATURE,
                "paper1_feature_match_status"]],
        on="case_id", how="left", validate="one_to_one", indicator=True,
        suffixes=("_profile", "_master"),
    )
    if not matched._merge.eq("both").all() or not matched.paper1_feature_match_status.eq("MATCHED").all():
        raise ValueError(f"{domain}: profile source patient lacks verified Paper 1 master match")
    filename_date = matched.paper1_filename_date8_master.astype("string").str.replace(r"\.0$", "", regex=True)
    if not matched.paper1_filename_date8_profile.eq(filename_date).all():
        raise ValueError(f"{domain}: original video filename date disagrees with master")
    official = numeric(matched[MASTER_FEATURE], f"{domain} master F01", allow_missing=True)
    original = matched.aggregate_original
    same = ((official.isna() & original.isna()) |
            (official.notna() & original.notna() & np.isclose(official, original, rtol=1e-8, atol=1e-10)))
    if not same.all():
        raise ValueError(f"{domain}: original F01 aggregate differs from frozen master")

    records = []
    for source_case, group in p.groupby("case_id", sort=False):
        group = group.sort_values("bin_index")
        if group.bin_index.tolist() != list(range(1, BINS + 1)):
            raise ValueError(f"{domain}: five contiguous ordered bins required")
        bins = group.original.to_numpy(dtype=float)
        complete = bool(np.isfinite(bins).all())
        center = float(np.median(bins)) if complete else np.nan
        score = (float(np.median(np.abs(bins - center)) / center)
                 if complete and center > 0 else np.nan)
        records.append({"source_case_id": source_case,
                        f"{domain}_complete_bins": int(np.isfinite(bins).sum()),
                        f"{domain}_relative_mad": score})
    derived = pd.DataFrame(records)
    matched = matched.merge(derived, on="source_case_id", validate="one_to_one")
    return matched[["case_id", f"{domain}_complete_bins", f"{domain}_relative_mad"]]


def assemble(master: pd.DataFrame, temporal: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    result = master[["case_id", "eligible_activity_F01", *core.DIRECTIONS.values()]].copy()
    for domain, derived in (("temporal", temporal), ("spatial", spatial)):
        if derived.case_id.duplicated().any():
            raise ValueError(f"{domain}: duplicate patient")
        result = result.merge(derived, on="case_id", how="left", validate="one_to_one")
        result[f"{domain}_source_available"] = result[f"{domain}_complete_bins"].notna()
        result[f"{domain}_eligible"] = (core.flags(result.eligible_activity_F01, "eligible_activity_F01")
                                        & np.isfinite(result[f"{domain}_relative_mad"]))
    return result


def analyze(exploratory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for index, (domain, direction) in enumerate(INDICES):
        x = numeric(exploratory[f"{domain}_relative_mad"], domain, allow_missing=True)
        y = numeric(exploratory[core.DIRECTIONS[direction]], direction, allow_missing=True)
        eligible = core.flags(exploratory.eligible_activity_F01, "eligible_activity_F01")
        use = eligible & np.isfinite(x) & np.isfinite(y)
        result = core.association(x[use].to_numpy(float), y[use].to_numpy(float), 100 + index)
        rows.append({"family": "EXPLORATORY_SPATIOTEMPORAL_4",
                     "feature": f"{domain}_relative_mad", "activity_direction": direction,
                     "n": int(use.sum()), "zero_count_n": int(y[use].eq(0).sum()),
                     "eligible_f01_n": int(eligible.sum()), "profile_available_n": int(exploratory[f"{domain}_source_available"].sum()),
                     "complete_five_bin_n": int(exploratory[f"{domain}_complete_bins"].eq(BINS).sum()),
                     **result})
    table = pd.DataFrame(rows)
    table["holm_adjusted_p"] = core.holm_with_planned_family(table.raw_p)
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", type=Path, required=True)
    ap.add_argument("--master-manifest", type=Path, required=True)
    for domain in DOMAINS:
        ap.add_argument(f"--{domain}-profiles", type=Path, required=True)
        ap.add_argument(f"--{domain}-summary", type=Path, required=True)
        ap.add_argument(f"--{domain}-source-manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output exists, choose a new directory: {args.output}")
    source_paths = {"master": args.master, "master_manifest": args.master_manifest}
    manifest = json.loads(args.master_manifest.read_text(encoding="utf-8"))
    if core.sha256(args.master) != manifest.get("output_sha256", {}).get("master_csv"):
        raise ValueError("master SHA256 differs from script 05 manifest")
    gate = Path(manifest["input_paths"]["frozen_gate"])
    if core.sha256(gate) != manifest["frozen_gate_sha256"]:
        raise ValueError("frozen feature gate SHA256 differs from script 05 manifest")
    master = read_csv(args.master)
    core.validate_master(master, manifest)
    pieces = {}
    for domain in DOMAINS:
        profiles = getattr(args, f"{domain}_profiles")
        summary = getattr(args, f"{domain}_summary")
        source = getattr(args, f"{domain}_source_manifest")
        source_paths.update({f"{domain}_profiles": profiles, f"{domain}_summary": summary,
                             f"{domain}_source_manifest": source})
        pieces[domain] = original_profile_features(
            read_csv(profiles), read_csv(summary),
            json.loads(source.read_text(encoding="utf-8")), master, domain,
        )
    exploratory = assemble(master, pieces["temporal"], pieces["spatial"])
    result = analyze(exploratory)
    args.output.mkdir(parents=True, exist_ok=False)
    features_path = args.output / "exploratory_patient_features.csv"
    assoc_path = args.output / "exploratory_activity_associations.csv"
    exploratory.to_csv(features_path, index=False, encoding="utf-8-sig", na_rep="NA")
    result.to_csv(assoc_path, index=False, encoding="utf-8-sig", na_rep="NA")
    output_manifest = {
        "status": "exploratory_spatiotemporal_associations_complete",
        "source_paths": {k: str(v.resolve()) for k, v in source_paths.items()},
        "source_sha256": {k: core.sha256(v) for k, v in source_paths.items()},
        "frozen_gate_sha256": core.sha256(gate),
        "output_sha256": {"features": core.sha256(features_path), "associations": core.sha256(assoc_path)},
        "source_feature": FEATURE_ID,
        "profile_source": "Paper 1 original 5-bin normalized profiles; not Grade-3 shadow or robustness rho/SRD",
        "metric": "median(abs(original 5 bin medians - median(original bins))) / median(original bins)",
        "requirement": "five finite original bins, positive median, nonzero valid-position count per bin",
        "feature_gate_role": "NEW_EXPLORATORY_NOT_PAPER1_GATE_QUALIFIED",
        "analysis_family": list(f"{d}-{v}" for d, v in INDICES),
        "statistical_method": "same Spearman average ranks and permutation/bootstrap implementation as script 06",
        "permutations": core.PERMUTATIONS, "bootstraps": core.BOOTSTRAPS,
        "permutation_seeds": [20260922 + 100 + i for i in range(4)],
        "bootstrap_seeds": [20270922 + 100 + i for i in range(4)],
        "multiplicity": "Holm over all 4 prespecified exploratory tests",
        "clinical_covariates_adjusted": False,
        "pregnancy_outcome_used": False, "changes_to_script_06": False,
    }
    (args.output / "manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Exploratory original-profile heterogeneity and clinician activity", "",
             "These new F01-derived metrics are NOT Paper 1 robustness-qualified; do not replace the frozen F01 primary analyses.",
             "Relative MAD across all five original normalized-time or normalized-position bin medians, defined only when all bins are finite and the median is positive.",
             "Spearman, two-sided paired permutation P and 95% paired bootstrap CI as script 06; Holm across four exploratory tests.",
             "Per-feature eligible F01 patients only; recorded zero counts included. No activity direction, propagation, wave frequency or pregnancy inference.", "",
             "| Metric | Direction | N | rho | 95% CI | raw P | Holm P | Status |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for row in result.itertuples():
        ci = f"[{row.ci_lower:.3f}, {row.ci_upper:.3f}]" if np.isfinite(row.ci_lower) else "NA"
        lines.append(f"| {row.feature} | {row.activity_direction} | {row.n} | {row.rho:.3f} | "
                     f"{ci} | {row.raw_p:.4g} | {row.holm_adjusted_p:.4g} | {row.status} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    print(f"COMPLETE: {args.output}")


if __name__ == "__main__":
    main()
