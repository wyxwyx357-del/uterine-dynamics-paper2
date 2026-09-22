# Paper 1 -> Paper 2 feature gate

## Purpose

`scripts/04_audit_paper1_feature_eligibility.py` consolidates the frozen
Paper 1 evidence for F01-F20 before any Paper 2 clinical association analysis.

It does **not** create a new robustness cutoff and it does **not** select
features using Paper 2 pregnancy labels, p-values, odds ratios, or AUC.

## Required Paper 1 inputs

The expected current Paper 1 result files are:

1. Analytical perturbation robustness:
   - `feature_summary.csv`
   - produced by `run_all_patient_perturbation_stability.py`

2. Proportional observation-window truncation:
   - `clip_duration_pairwise_summary.csv`
   - produced by `run_clip_duration_statistics.py`

3. Grade-3 mask-only sensitivity:
   - `Paper1_QC敏感性正式汇总.csv`
   - produced by `report_quality_sensitivity.py`

4. Normalized-time structure:
   - `Paper1_时间结构稳定性正式汇总.csv`

5. Normalized-space structure:
   - `Paper1_空间结构稳定性正式汇总.csv`

The default QC/temporal/spatial stratum is `pending_grade3`, matching the
predefined Grade-3 mask-only sensitivity analysis.

## Evidence-only run

Example:

```bat
python scripts\04_audit_paper1_feature_eligibility.py ^
  --perturbation-summary "PATH\feature_summary.csv" ^
  --duration-summary "PATH\clip_duration_pairwise_summary.csv" ^
  --qc-summary "PATH\Paper1_QC敏感性正式汇总.csv" ^
  --temporal-summary "PATH\Paper1_时间结构稳定性正式汇总.csv" ^
  --spatial-summary "PATH\Paper1_空间结构稳定性正式汇总.csv" ^
  --output "PATH\Paper2_Paper1特征资格证据"
```

Outputs:

- `paper1_paper2_feature_evidence.csv`
- `paper2_feature_gate.csv`
- `paper2_feature_decision_template.csv`
- `manifest.json`
- `REPORT.md`

In an evidence-only run every feature remains `UNFROZEN`. This is deliberate.

## Final feature decision

After the Paper 1 evidence table is reviewed and the Paper 2 feature set is
prespecified, fill a copy of `paper2_feature_decision_template.csv`.

Required columns:

- `feature_id`
- `decision`: `INCLUDE`, `EXCLUDE`, or `REVIEW`
- `role`: `PRIMARY`, `SECONDARY`, `NONE`, or `REVIEW`
- `rationale`
- `decision_basis`: must be `PAPER1_ONLY`

A finalized decision file must contain exactly one `PRIMARY` feature.

Run again with:

```bat
python scripts\04_audit_paper1_feature_eligibility.py ... ^
  --decision-file "PATH\paper2_feature_decisions_frozen.csv" ^
  --output "PATH\Paper2_Paper1特征资格冻结"
```

The script rejects decision files that contain outcome-style result columns
such as pregnancy AUC, outcome p-values, or odds ratios.

## Interpretation boundary

`EVIDENCE_READY_FOR_PRESPECIFIED_DECISION` means that the feature has
evaluable records in every required Paper 1 evidence source. It does **not**
mean that an automatic robustness threshold was passed.

Paper 1 source analyses explicitly report descriptive robustness for
proportional truncation, QC sensitivity, temporal structure, and spatial
structure without defining stable/unstable thresholds. Therefore the
Paper 1 -> Paper 2 gate must not invent one.
