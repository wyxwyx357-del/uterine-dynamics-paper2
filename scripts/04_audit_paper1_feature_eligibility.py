#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the Paper 1 -> Paper 2 feature evidence gate for frozen F01-F20.

This script does not select features from Paper 2 clinical associations, pregnancy
outcomes, p-values, odds ratios, or AUC. It only consolidates already-generated
Paper 1 robustness evidence. Paper 1 analyses explicitly do not define automatic
stable/unstable thresholds for truncation, QC, temporal, or spatial sensitivity,
so this script does not invent any.

Without --decision-file the output is evidence-only and every feature remains
UNFROZEN. A final decision file may encode a separately prespecified Paper-1-only
PRIMARY/SECONDARY/EXCLUDE decision after review of the evidence table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_CONTRACT = (
    ("F01", "rsr_abs_median", "rsr", "median", "global"),
    ("F02", "rsr_abs_p95", "rsr", "p95", "global"),
    ("F03", "anterior_rsr_abs_median", "rsr", "median", "anterior"),
    ("F04", "anterior_rsr_abs_p95", "rsr", "p95", "anterior"),
    ("F05", "posterior_rsr_abs_median", "rsr", "median", "posterior"),
    ("F06", "posterior_rsr_abs_p95", "rsr", "p95", "posterior"),
    ("F07", "cavity_width_strain_rate_abs_median", "cavity", "median", "global"),
    ("F08", "cavity_width_strain_rate_abs_p95", "cavity", "p95", "global"),
    ("F09", "longitudinal_wall_strain_rate_abs_median", "longitudinal", "median", "global"),
    ("F10", "longitudinal_wall_strain_rate_abs_p95", "longitudinal", "p95", "global"),
    ("F11", "anterior_longitudinal_wall_strain_rate_abs_median", "longitudinal", "median", "anterior"),
    ("F12", "anterior_longitudinal_wall_strain_rate_abs_p95", "longitudinal", "p95", "anterior"),
    ("F13", "posterior_longitudinal_wall_strain_rate_abs_median", "longitudinal", "median", "posterior"),
    ("F14", "posterior_longitudinal_wall_strain_rate_abs_p95", "longitudinal", "p95", "posterior"),
    ("F15", "wall_curvature_change_rate_mm_inv_s_abs_median", "curvature", "median", "global"),
    ("F16", "wall_curvature_change_rate_mm_inv_s_abs_p95", "curvature", "p95", "global"),
    ("F17", "anterior_wall_curvature_change_rate_mm_inv_s_abs_median", "curvature", "median", "anterior"),
    ("F18", "anterior_wall_curvature_change_rate_mm_inv_s_abs_p95", "curvature", "p95", "anterior"),
    ("F19", "posterior_wall_curvature_change_rate_mm_inv_s_abs_median", "curvature", "median", "posterior"),
    ("F20", "posterior_wall_curvature_change_rate_mm_inv_s_abs_p95", "curvature", "p95", "posterior"),
)

CONTRACT = pd.DataFrame(
    FEATURE_CONTRACT,
    columns=["feature_id", "feature_name", "family", "statistic", "domain"],
)
CODE_TO_NAME = dict(zip(CONTRACT.feature_id, CONTRACT.feature_name))
NAME_TO_CODE = dict(zip(CONTRACT.feature_name, CONTRACT.feature_id))

PERTURB_REQUIRED = {
    "n_cases", "primary_icc", "ci_lower", "ci_upper",
    "median_cv_pct", "p95_cv_pct", "cv_lt20_fraction",
}
DURATION_REQUIRED = {
    "target_percent", "n_pairs", "icc_a1", "icc_ci_lower", "icc_ci_upper",
    "spearman_rho", "median_absolute_relative_error_pct",
    "p95_absolute_relative_error_pct",
}
QC_REQUIRED = {
    "n_original_finite", "n_paired_finite", "finite_to_nan_n",
    "availability_retention_fraction", "spearman",
    "median_srd_pct", "p95_srd_pct",
}
TEMPORAL_REQUIRED = {
    "n_cases", "n_temporal_spearman_available", "median_temporal_spearman",
    "p05_temporal_spearman", "median_valid_position_retention_fraction",
    "finite_to_nan_bins_total", "cases_with_finite_to_nan_bins",
    "median_case_median_bin_srd_pct", "p95_case_median_bin_srd_pct",
    "median_case_p95_bin_srd_pct", "median_aggregate_srd_pct",
}
SPATIAL_REQUIRED = {
    "n_cases", "n_spatial_spearman_available", "median_spatial_spearman",
    "p05_spatial_spearman", "median_valid_position_retention_fraction",
    "finite_to_nan_bins_total", "cases_with_finite_to_nan_bins",
    "median_case_median_bin_srd_pct", "p95_case_median_bin_srd_pct",
    "median_case_p95_bin_srd_pct", "median_aggregate_srd_pct",
}

FORBIDDEN_DECISION_COLUMN_TOKENS = (
    "auc", "pvalue", "p_value", "odds_ratio", "pregnancy",
    "outcome", "label", "association_result", "spearman_p",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    table = pd.read_csv(path, encoding="utf-8-sig")
    if table.empty:
        raise ValueError(f"{label} is empty: {path}")
    return table


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    present = set(columns)
    for name in candidates:
        if name in present:
            return name
    return None


def normalize_feature_identity(table: pd.DataFrame, source: str) -> pd.DataFrame:
    result = table.copy()
    code_col = first_existing(result.columns, ("feature_id", "feature_code"))
    name_col = first_existing(result.columns, ("feature_name", "feature"))
    if code_col is None and name_col is None:
        raise ValueError(f"{source}: missing feature identifier/name column")

    if code_col is not None:
        codes = result[code_col].astype("string").str.strip().str.upper()
        bad = sorted(set(codes.dropna()) - set(CODE_TO_NAME))
        if bad:
            raise ValueError(f"{source}: unknown feature ids: {bad}")
        result["_feature_id"] = codes
    else:
        names = result[name_col].astype("string").str.strip()
        unknown = sorted(set(names.dropna()) - set(NAME_TO_CODE))
        if unknown:
            raise ValueError(f"{source}: unknown feature names: {unknown}")
        result["_feature_id"] = names.map(NAME_TO_CODE)

    if name_col is not None:
        names = result[name_col].astype("string").str.strip()
        unknown = sorted(set(names.dropna()) - set(NAME_TO_CODE))
        if unknown:
            raise ValueError(f"{source}: unknown feature names: {unknown}")
        expected = result["_feature_id"].map(CODE_TO_NAME)
        mismatch = names.notna() & expected.notna() & names.ne(expected)
        if mismatch.any():
            rows = result.loc[mismatch, ["_feature_id", name_col]].head(10)
            raise ValueError(
                f"{source}: feature id/name mismatch:\n{rows.to_string(index=False)}"
            )
        result["_feature_name"] = names
    else:
        result["_feature_name"] = result["_feature_id"].map(CODE_TO_NAME)

    return result


def validate_contract_metadata(table: pd.DataFrame, source: str) -> None:
    meta = CONTRACT.set_index("feature_id")
    candidates = {
        "family": ("family", "feature_family"),
        "statistic": ("statistic", "statistic_type"),
        "domain": ("domain", "feature_domain"),
    }
    for contract_col, aliases in candidates.items():
        col = first_existing(table.columns, aliases)
        if col is None:
            continue
        expected = table["_feature_id"].map(meta[contract_col])
        observed = table[col].astype("string").str.strip().str.lower()
        mismatch = (
            observed.notna()
            & expected.notna()
            & observed.ne(expected.astype(str).str.lower())
        )
        if mismatch.any():
            rows = table.loc[mismatch, ["_feature_id", col]].head(10)
            raise ValueError(
                f"{source}: {contract_col} conflicts with frozen F01-F20 contract:\n"
                + rows.to_string(index=False)
            )


def require_columns(table: pd.DataFrame, required: set[str], source: str) -> None:
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {missing}")


def require_one_row_per_feature(table: pd.DataFrame, source: str) -> None:
    counts = table["_feature_id"].value_counts(dropna=False)
    missing = sorted(set(CODE_TO_NAME) - set(counts.index.dropna()))
    duplicates = counts[counts != 1].to_dict()
    if missing or duplicates or len(table) != 20:
        raise ValueError(
            f"{source}: expected exactly one row for each F01-F20; "
            f"missing={missing}, nonunique={duplicates}, rows={len(table)}"
        )


def prefixed_single_feature_table(
    table: pd.DataFrame,
    prefix: str,
    source: str,
) -> pd.DataFrame:
    require_one_row_per_feature(table, source)
    ignored = {
        "feature_id", "feature_code", "feature_name", "feature",
        "_feature_id", "_feature_name",
    }
    payload = [c for c in table.columns if c not in ignored]
    result = table[["_feature_id", *payload]].copy()
    result = result.rename(columns={"_feature_id": "feature_id"})
    result = result.rename(columns={c: f"{prefix}_{c}" for c in payload})
    return result.sort_values("feature_id").reset_index(drop=True)


def select_stratum(table: pd.DataFrame, source: str, stratum: str) -> pd.DataFrame:
    if "stratum" not in table.columns:
        return table.copy()
    available = sorted(table["stratum"].dropna().astype(str).unique())
    selected = table.loc[table["stratum"].astype(str).eq(stratum)].copy()
    if selected.empty:
        raise ValueError(
            f"{source}: requested stratum {stratum!r} not found; available={available}"
        )
    return selected


def build_duration_wide(
    table: pd.DataFrame,
    targets: tuple[int, ...],
) -> pd.DataFrame:
    require_columns(table, DURATION_REQUIRED, "duration summary")
    values = pd.to_numeric(table["target_percent"], errors="coerce")
    if values.isna().any():
        raise ValueError("duration summary: target_percent contains non-numeric values")
    table = table.copy()
    table["target_percent"] = values.astype(int)

    expected_targets = set(targets)
    observed_targets = set(table["target_percent"].unique())
    unexpected = sorted(observed_targets - expected_targets)
    if unexpected:
        raise ValueError(
            f"duration summary: unexpected target_percent values {unexpected}; "
            f"expected only {sorted(expected_targets)}"
        )

    identity_cols = {
        "feature_id", "feature_code", "feature_name", "feature",
        "_feature_id", "_feature_name", "target_percent", "reference_percent",
    }
    metric_cols = [c for c in table.columns if c not in identity_cols]
    result = CONTRACT[["feature_id"]].copy()
    for target in targets:
        part = table.loc[table["target_percent"].eq(target)].copy()
        require_one_row_per_feature(part, f"duration summary target={target}")
        part = part[["_feature_id", *metric_cols]].rename(
            columns={"_feature_id": "feature_id"}
        )
        part = part.rename(
            columns={c: f"duration_p{target:03d}_{c}" for c in metric_cols}
        )
        result = result.merge(
            part, on="feature_id", how="left", validate="one_to_one"
        )
    return result


def _positive_numeric(row: pd.Series, column: str) -> bool:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return bool(np.isfinite(value) and float(value) > 0)


def add_evidence_gate(
    evidence: pd.DataFrame,
    duration_targets: tuple[int, ...],
) -> pd.DataFrame:
    result = evidence.copy()
    statuses = []
    reasons = []
    for _, row in result.iterrows():
        problems: list[str] = []
        if not _positive_numeric(row, "perturb_n_cases"):
            problems.append("perturbation_not_evaluable")
        for target in duration_targets:
            if not _positive_numeric(row, f"duration_p{target:03d}_n_pairs"):
                problems.append(f"duration_{target}_not_evaluable")
        if not _positive_numeric(row, "qc_n_original_finite"):
            problems.append("qc_not_evaluable")
        if not _positive_numeric(row, "temporal_n_cases"):
            problems.append("temporal_not_evaluable")
        if not _positive_numeric(row, "spatial_n_cases"):
            problems.append("spatial_not_evaluable")
        if problems:
            statuses.append("NOT_EVALUABLE")
            reasons.append("|".join(problems))
        else:
            statuses.append("EVIDENCE_READY_FOR_PRESPECIFIED_DECISION")
            reasons.append("")
    result["paper1_evidence_status"] = statuses
    result["paper1_evidence_issue"] = reasons
    result["automatic_robustness_pass_created"] = False
    return result


def decision_template() -> pd.DataFrame:
    result = CONTRACT[["feature_id", "feature_name"]].copy()
    result["decision"] = ""
    result["role"] = ""
    result["rationale"] = ""
    result["decision_basis"] = "PAPER1_ONLY"
    return result


def load_decisions(path: Path) -> pd.DataFrame:
    table = read_csv(path, "decision file")
    lower_cols = [str(c).strip().lower() for c in table.columns]
    forbidden = [
        c
        for c in lower_cols
        if any(token in c for token in FORBIDDEN_DECISION_COLUMN_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            "decision file contains Paper-2/outcome-style result columns, which are "
            f"not allowed for feature selection: {forbidden}"
        )

    required = {"feature_id", "decision", "role", "rationale", "decision_basis"}
    require_columns(table, required, "decision file")
    table = normalize_feature_identity(table, "decision file")
    require_one_row_per_feature(table, "decision file")

    result = table[
        ["_feature_id", "decision", "role", "rationale", "decision_basis"]
    ].copy()
    result = result.rename(columns={"_feature_id": "feature_id"})
    result["decision"] = result["decision"].astype(str).str.strip().str.upper()
    result["role"] = result["role"].astype(str).str.strip().str.upper()
    result["decision_basis"] = (
        result["decision_basis"].astype(str).str.strip().str.upper()
    )

    allowed_decision = {"INCLUDE", "EXCLUDE", "REVIEW"}
    allowed_role = {"PRIMARY", "SECONDARY", "NONE", "REVIEW"}
    bad_decision = sorted(set(result.decision) - allowed_decision)
    bad_role = sorted(set(result.role) - allowed_role)
    if bad_decision:
        raise ValueError(f"decision file: invalid decision values {bad_decision}")
    if bad_role:
        raise ValueError(f"decision file: invalid role values {bad_role}")
    if set(result.decision_basis) != {"PAPER1_ONLY"}:
        raise ValueError("decision file: every decision_basis must be PAPER1_ONLY")

    invalid_pairs = result.loc[
        ((result.decision == "INCLUDE")
         & ~result.role.isin(["PRIMARY", "SECONDARY"]))
        | ((result.decision == "EXCLUDE") & (result.role != "NONE"))
        | ((result.decision == "REVIEW") & (result.role != "REVIEW"))
    ]
    if not invalid_pairs.empty:
        raise ValueError(
            "decision file: decision/role mismatch:\n"
            + invalid_pairs[["feature_id", "decision", "role"]].to_string(
                index=False
            )
        )

    primary_n = int((result.role == "PRIMARY").sum())
    if primary_n != 1:
        raise ValueError(
            f"decision file must freeze exactly one PRIMARY feature; found {primary_n}"
        )
    return result


def apply_decisions(
    evidence: pd.DataFrame,
    decisions: pd.DataFrame | None,
) -> pd.DataFrame:
    result = evidence.copy()
    if decisions is None:
        result["paper2_decision"] = "UNFROZEN"
        result["paper2_role"] = "UNFROZEN"
        result["paper2_decision_rationale"] = ""
        result["paper2_decision_basis"] = ""
        result["paper2_eligible"] = pd.Series(
            [pd.NA] * len(result), dtype="boolean"
        )
        return result

    result = result.merge(
        decisions, on="feature_id", how="left", validate="one_to_one"
    )
    include_not_evaluable = result.loc[
        result["decision"].eq("INCLUDE")
        & ~result["paper1_evidence_status"].eq(
            "EVIDENCE_READY_FOR_PRESPECIFIED_DECISION"
        )
    ]
    if not include_not_evaluable.empty:
        raise ValueError(
            "decision file includes feature(s) without evaluable Paper 1 evidence: "
            + ", ".join(include_not_evaluable.feature_id.astype(str))
        )

    result = result.rename(
        columns={
            "decision": "paper2_decision",
            "role": "paper2_role",
            "rationale": "paper2_decision_rationale",
            "decision_basis": "paper2_decision_basis",
        }
    )
    result["paper2_eligible"] = result["paper2_decision"].eq("INCLUDE")
    return result


def build_evidence(
    *,
    perturbation: pd.DataFrame,
    duration: pd.DataFrame,
    qc: pd.DataFrame,
    temporal: pd.DataFrame,
    spatial: pd.DataFrame,
    duration_targets: tuple[int, ...] = (75, 50, 25),
    qc_stratum: str = "pending_grade3",
    temporal_stratum: str = "pending_grade3",
    spatial_stratum: str = "pending_grade3",
) -> pd.DataFrame:
    p = normalize_feature_identity(perturbation, "perturbation summary")
    validate_contract_metadata(p, "perturbation summary")
    require_columns(p, PERTURB_REQUIRED, "perturbation summary")
    p = prefixed_single_feature_table(
        p, "perturb", "perturbation summary"
    )

    d = normalize_feature_identity(duration, "duration summary")
    validate_contract_metadata(d, "duration summary")
    d = build_duration_wide(d, duration_targets)

    q = normalize_feature_identity(qc, "QC summary")
    validate_contract_metadata(q, "QC summary")
    q = select_stratum(q, "QC summary", qc_stratum)
    require_columns(q, QC_REQUIRED, "QC summary")
    q = prefixed_single_feature_table(
        q, "qc", f"QC summary stratum={qc_stratum}"
    )

    t = normalize_feature_identity(temporal, "temporal summary")
    validate_contract_metadata(t, "temporal summary")
    t = select_stratum(t, "temporal summary", temporal_stratum)
    require_columns(t, TEMPORAL_REQUIRED, "temporal summary")
    t = prefixed_single_feature_table(
        t, "temporal", f"temporal summary stratum={temporal_stratum}"
    )

    s = normalize_feature_identity(spatial, "spatial summary")
    validate_contract_metadata(s, "spatial summary")
    s = select_stratum(s, "spatial summary", spatial_stratum)
    require_columns(s, SPATIAL_REQUIRED, "spatial summary")
    s = prefixed_single_feature_table(
        s, "spatial", f"spatial summary stratum={spatial_stratum}"
    )

    evidence = CONTRACT.copy()
    for part in (p, d, q, t, s):
        evidence = evidence.merge(
            part, on="feature_id", how="left", validate="one_to_one"
        )
    return add_evidence_gate(evidence, duration_targets)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--perturbation-summary", type=Path, required=True)
    p.add_argument("--duration-summary", type=Path, required=True)
    p.add_argument("--qc-summary", type=Path, required=True)
    p.add_argument("--temporal-summary", type=Path, required=True)
    p.add_argument("--spatial-summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new output directory")
    p.add_argument("--decision-file", type=Path)
    p.add_argument(
        "--duration-targets", nargs="+", type=int, default=[75, 50, 25]
    )
    p.add_argument("--qc-stratum", default="pending_grade3")
    p.add_argument("--temporal-stratum", default="pending_grade3")
    p.add_argument("--spatial-stratum", default="pending_grade3")
    p.add_argument("--paper1-code-commit", default="unrecorded")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    output.mkdir(parents=True)

    source_paths = {
        "perturbation_summary": args.perturbation_summary.resolve(),
        "duration_summary": args.duration_summary.resolve(),
        "qc_summary": args.qc_summary.resolve(),
        "temporal_summary": args.temporal_summary.resolve(),
        "spatial_summary": args.spatial_summary.resolve(),
    }
    tables = {
        key: read_csv(path, key) for key, path in source_paths.items()
    }

    targets = tuple(int(x) for x in args.duration_targets)
    if (
        len(set(targets)) != len(targets)
        or any(x <= 0 or x >= 100 for x in targets)
    ):
        raise ValueError(
            "duration targets must be unique integers strictly between 0 and 100"
        )

    evidence = build_evidence(
        perturbation=tables["perturbation_summary"],
        duration=tables["duration_summary"],
        qc=tables["qc_summary"],
        temporal=tables["temporal_summary"],
        spatial=tables["spatial_summary"],
        duration_targets=targets,
        qc_stratum=args.qc_stratum,
        temporal_stratum=args.temporal_stratum,
        spatial_stratum=args.spatial_stratum,
    )

    decisions = None
    if args.decision_file is not None:
        decisions = load_decisions(args.decision_file.resolve())
    gate = apply_decisions(evidence, decisions)

    evidence_path = output / "paper1_paper2_feature_evidence.csv"
    gate_path = output / "paper2_feature_gate.csv"
    template_path = output / "paper2_feature_decision_template.csv"
    evidence.to_csv(
        evidence_path, index=False, encoding="utf-8-sig", na_rep="NA"
    )
    gate.to_csv(
        gate_path, index=False, encoding="utf-8-sig", na_rep="NA"
    )
    decision_template().to_csv(
        template_path, index=False, encoding="utf-8-sig"
    )

    input_hashes = {
        key: sha256(path) for key, path in source_paths.items()
    }
    decision_hash = (
        None
        if args.decision_file is None
        else sha256(args.decision_file.resolve())
    )
    manifest = {
        "status": "paper1_to_paper2_feature_evidence_gate_created",
        "feature_contract": (
            "frozen F01-F20 from Paper 1 formal_feature_extraction.py"
        ),
        "feature_count": 20,
        "paper1_code_commit": args.paper1_code_commit,
        "paper1_inputs": {
            key: str(path) for key, path in source_paths.items()
        },
        "paper1_input_sha256": input_hashes,
        "duration_targets_pct": list(targets),
        "qc_stratum": args.qc_stratum,
        "temporal_stratum": args.temporal_stratum,
        "spatial_stratum": args.spatial_stratum,
        "automatic_robustness_threshold_created": False,
        "paper2_outcome_information_used": False,
        "paper2_association_results_used": False,
        "decision_file_used": args.decision_file is not None,
        "decision_file_sha256": decision_hash,
        "decision_rule": (
            "UNFROZEN evidence-only output"
            if decisions is None
            else (
                "externally frozen PAPER1_ONLY decision file; "
                "exactly one PRIMARY required"
            )
        ),
        "evidence_ready_count": int(
            evidence.paper1_evidence_status.eq(
                "EVIDENCE_READY_FOR_PRESPECIFIED_DECISION"
            ).sum()
        ),
        "not_evaluable_count": int(
            evidence.paper1_evidence_status.eq("NOT_EVALUABLE").sum()
        ),
        "outputs": {
            "evidence": str(evidence_path),
            "gate": str(gate_path),
            "decision_template": str(template_path),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if decisions is None:
        conclusion = (
            "Evidence aggregation complete. Feature roles remain UNFROZEN; "
            "no automatic Paper 2 selection was performed."
        )
    else:
        selected = gate.loc[
            gate.paper2_eligible,
            ["feature_id", "feature_name", "paper2_role"],
        ]
        conclusion = "Frozen Paper-1-only decisions applied: " + "; ".join(
            f"{r.feature_id}={r.paper2_role}"
            for r in selected.itertuples()
        )

    report = f"""# Paper 1 -> Paper 2 feature gate

Features: 20 frozen F01-F20.
Evidence-ready features: {manifest['evidence_ready_count']}.
Not evaluable: {manifest['not_evaluable_count']}.

This program consolidates Paper 1 perturbation, proportional truncation,
Grade-3 mask sensitivity, normalized-time, and normalized-space evidence.
It deliberately does not create new stable/unstable cutoffs because the
source Paper 1 analyses do not define such thresholds.

{conclusion}

Clinical pregnancy labels, Paper 2 association p-values, odds ratios, and
AUC are not inputs to this gate.
"""
    (output / "REPORT.md").write_text(report, encoding="utf-8")

    print(
        gate[
            [
                "feature_id",
                "feature_name",
                "paper1_evidence_status",
                "paper2_decision",
                "paper2_role",
                "paper2_eligible",
            ]
        ].to_string(index=False)
    )
    print(f"\nCOMPLETE: {output}")


if __name__ == "__main__":
    main()
