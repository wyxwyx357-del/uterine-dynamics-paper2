# Paper 2 end-to-end execution and audit flow

This is the code path for the current article: exploratory associations of four Paper 1-frozen local short-video measurements with six endometrial Doppler indices. The earlier clinician-activity analysis is retained; pregnancy is supplementary. The steps below describe actual repository programs and external inputs, not a claim that the complete patient-level Doppler chain has been formally reaccepted.

## 0. Freeze inputs and record versions

Record `git status`, branch, HEAD, the exact external input paths, SHA256 hashes, file creation/provenance dates, and the pre-Doppler date of the `PAPER1_ONLY` decision. Keep this record outside Git with the private data release. Use a new output directory for each run and preserve prior results. If already confirmed 01–03 files are the intended source, reuse those exact files; rerunning OCR or rematching dates is not needed just to rerun statistics.

Do not place patient names, raw workbooks, patient-level master tables, videos, or DICOM in Git. Verify that the clinical table and video refer to the already confirmed same examination. The 361-patient DICOM review is separate; this repository does not establish common VI/FI/VFI Gain, PRF, or wall-filter settings.

## 1. Identity and date chain

| Step | Input | Program and output | Acceptance check |
|---|---|---|---|
| 01 | External video root | `01_check_video_exam_date.py` → `video_exam_date_audit.csv`, mismatch review, summary | OCR/video examination date is reviewed; retain the confirmed source file. |
| 02 | 01 CSV + external clinician workbook, sheets `患者信息提取` and `匹配质控` | `02_build_video_clinical_match.py` → three-way workbook, including `01_全部病例三方核对` | Resolve ambiguous/conflicting case IDs and nonunique clinical rows. Do not accept a guessed ID or date. |
| 03 | 02 workbook | `03_audit_patient_data.py` → patient audit workbook, including `02_患者级状态` | Record base, activity, pregnancy, duplicate and date statuses. Activity 0 is valid; do not use obsolete prediction candidate output. |

Example commands from the repository root (substitute private paths):

```powershell
python -X utf8 scripts/01_check_video_exam_date.py --video-root "PATH_TO_VIDEO_ROOT" --output-root "NEW_01_DIR"
python -X utf8 scripts/02_build_video_clinical_match.py --video-audit "CONFIRMED_01_DIR/video_exam_date_audit.csv" --clinical-workbook "PATH_TO_CLINICAL_WORKBOOK.xlsx" --output "NEW_02_DIR/three_way_audit.xlsx"
python -X utf8 scripts/03_audit_patient_data.py --input "NEW_02_DIR/three_way_audit.xlsx" --output "NEW_03_DIR/patient_audit.xlsx"
```

Keep the confirmed 01–03 artifacts if they have already been accepted. Script 05 rejects a Paper 1 video identity or OCR-confirmed date that differs from the patient audit. Script 11 checks the exact script-03 workbook hash and matched/video date fields against the frozen master.

## 2. Frozen feature gate

`04_audit_paper1_feature_eligibility.py` consumes five external Paper 1 evidence summaries: perturbation, duration, QC, normalized time, and normalized space. It inventories F01–F20. An evidence-only run is `UNFROZEN`; only a separate, dated, externally reviewed `PAPER1_ONLY` decision file makes a final gate. Archive that decision file and its hash so the chronology can be audited. The script-04 gate and its manifest must be retained together.

The only formal Paper 2 measurements are:

| ID | Paper 1 field | Original activity role |
|---|---|---|
| F01 | `rsr_abs_median` | PRIMARY |
| F07 | `cavity_width_strain_rate_abs_median` | SECONDARY |
| F09 | `longitudinal_wall_strain_rate_abs_median` | SECONDARY |
| F15 | `wall_curvature_change_rate_mm_inv_s_abs_median` | SECONDARY |

The other 16 F01–F20 fields are excluded from formal Paper 2 hypotheses. No Doppler, activity, pregnancy, P-value, OR, or AUC result may alter this decision. See `PAPER1_TO_PAPER2_FEATURE_GATE.md` for the five source filenames and exact script-04 options.

## 3. Patient master

```powershell
python -X utf8 scripts/05_build_paper2_analysis_master.py `
  --patient-audit "CONFIRMED_03_DIR/patient_audit.xlsx" `
  --video-date-audit "CONFIRMED_01_DIR/video_exam_date_audit.csv" `
  --paper1-features "PATH_TO_PAPER1_PATIENT_FEATURES.csv" `
  --paper1-source-manifest "PATH_TO_PAPER1_SOURCE_MANIFEST.json" `
  --frozen-gate "FROZEN_04_DIR/paper2_feature_gate.csv" `
  --output "NEW_05_DIR"
```

Script 05 verifies the Paper 1 feature-table hash against its source manifest, the 20-row frozen gate, one row per numeric case ID, Paper 1 source video, filename/OCR/matched exam dates, and one-to-one patient merge. It keeps all base rows. Each of F01/F07/F09/F15 is finite-or-missing independently; `Fxx_available` and `eligible_activity_Fxx`/`eligible_pregnancy_Fxx` are independently derived. A missing F15 cannot remove another feature's patient. It validates inherited activity/pregnancy/date eligibility, preserves activity count 0, and writes `paper2_analysis_master.csv`, XLSX, `cohort_counts.csv`, `manifest.json`, and a report. The manifest hashes all formal input files and the three tabular outputs.

Check base, Paper 1 feature-match, activity, pregnancy, and each feature-specific cohort count. Do not silently replace a source or reuse a manifest from a different master.

## 4. Statistical branches, in research order

**Original activity family (06).** Use the script-05 CSV and manifest:

```powershell
python -X utf8 scripts/06_analyze_clinician_activity_association.py --master "NEW_05_DIR/paper2_analysis_master.csv" --master-manifest "NEW_05_DIR/manifest.json" --output "NEW_06_DIR"
```

Eight patient-level Spearman comparisons use each feature and direction's own available patients. Forward/reverse counts are per minute and zero is valid. Two F01 tests form the PRIMARY Holm family; six F07/F09/F15 tests form the SECONDARY Holm family. Permutation, paired bootstrap, fixed seeds, and `NOT_EVALUABLE` handling are specified in `PAPER2_ACTIVITY_ASSOCIATION_PLAN.md`.

**Separate F01 temporal/spatial exploration (07).** This optional branch followed the activity analysis and uses Paper 1 profile files with its own four-comparison Holm family. It is never an input to script 08 or 11. Its exact source and command are in `PAPER2_SPATIOTEMPORAL_EXPLORATION.md`.

**Supplementary pregnancy (08).** After 06, run script 08 with the same script-05 CSV and manifest and a new output directory. Its clinical-outcome derivation was amended after the initial results; see `PAPER2_CLINICAL_PREGNANCY_PLAN.md`. This is not the article's main endpoint or a prediction/AUC workflow.

```powershell
python -X utf8 scripts/08_analyze_clinical_pregnancy.py --master "NEW_05_DIR/paper2_analysis_master.csv" --master-manifest "NEW_05_DIR/manifest.json" --output "NEW_08_DIR"
```

**Current exploratory Doppler family (11).** This was added after the four features had been frozen and after the earlier activity/pregnancy work. Use the *same* master and exact script-03 workbook hashed in its manifest. The external QC release must contain `patient_quality.csv` and `PRIVATE_source_inventory.csv`. Preserve the exact 14-case topology repair CSV.

```powershell
python -X utf8 scripts/11_analyze_doppler_associations.py `
  --master "NEW_05_DIR/paper2_analysis_master.csv" `
  --master-manifest "NEW_05_DIR/manifest.json" `
  --audit "CONFIRMED_03_DIR/patient_audit.xlsx" `
  --qc "PATH_TO_QC_RELEASE_DIR" `
  --repair "PATH_TO_14_CASE_TOPOLOGY_REPAIR.csv" `
  --output "NEW_11_DIR"
```

Script 11 makes all 24 feature × Doppler Spearman comparisons, one 24-test Holm family, an availability table, specified outlier/quality/clinical sensitivities, aggregate figures, report, and input/output hash manifest. F09–VI is a *post-result* focus for later new-patient validation, not a primary hypothesis of this original cohort. Script 11 retains the four archived original-cohort point estimates as an optional legacy reproduction gate (`--verify-reference-rho`) plus a specific QC/repair contract. The archived rho gate should be used only when reproducing that frozen cohort; it is not a validity condition for a legitimately corrected cohort or future new patients. Do not infer identical VI/FI/VFI acquisition settings from DICOM.

**Other separate code (09–10, 12).** Local scripts 09–10, if present, explore wall synchrony and do not change the frozen feature set. Script 12 is an earlier focused Doppler check. None replaces the 24-comparison result or enters it automatically.

## 5. Output lineage and formal-release gate

```text
External video + clinician workbook
  → 01 video date CSV → 02 three-way workbook → 03 patient audit workbook
External Paper 1 evidence + dated PAPER1_ONLY decision
  → 04 frozen F01–F20 gate
External Paper 1 patient features + source manifest
  + 01 CSV + 03 workbook + 04 gate
  → 05 master CSV + cohort counts + input/output hash manifest
  → 06 eight activity results
  → 07 separate F01 time/space exploration (optional)
  → 08 supplementary pregnancy results
  + 03 workbook + external QC release + 14-case repair
  → 11 twenty-four exploratory Doppler results + sensitivities + manifest
```

**MISSING FROM REPOSITORY:** Paper 1 tracking/feature extraction, original clinician Doppler-value extraction, external `PAPER1_ONLY` decision provenance, QC release generation, and topology repair generation. The repository can consume these inputs but cannot recreate them alone. Formal acceptance still requires a separate, read-only patient-level check from clinician VI/FI/VFI fields through case-ID merge, frozen features, cohort counts (including the reported 305/298 cohorts), all 24 results, correction, adjusted/quality sensitivities, and 14-case exclusion. This document and the local synthetic tests do not perform or certify that check.

Before reporting a result, retain the exact code revision, input/output manifests and hashes, feature decision chronology, cohort-flow counts, and analysis reports together. Never select a feature or method from the same cohort's significance result.
