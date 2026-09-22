# uterine-dynamics-paper2

Code and analysis plan for Paper 2 of the short transvaginal ultrasound uterine dynamics project.

## Scope

Paper 2 evaluates:
1. associations between Paper 1 robustness-qualified local uterine dynamic measurements and clinician-recorded uterine activity;
2. adjusted associations with clinical pregnancy;
3. exploratory incremental prediction beyond prespecified clinical information.

This repository does **not** contain raw patient videos, DICOM files, patient spreadsheets, identifying information, or Paper 1 tracking outputs.

## Current pipeline

- `scripts/01_check_video_exam_date.py` — verify video-displayed examination date.
- `scripts/02_build_video_clinical_match.py` — link video dates to the clinical workbook by case ID and audit date concordance.
- `scripts/03_audit_patient_data.py` — audit patient-level completeness and candidate analysis cohorts.
- `scripts/04_audit_paper1_feature_eligibility.py` — consolidate frozen Paper 1 robustness evidence for F01-F20 and enforce a Paper-1-only feature decision gate.

The Paper 1 → Paper 2 feature evidence gate is implemented. Feature roles remain unfrozen until a separate PAPER1_ONLY decision file is reviewed and frozen; association analyses must not begin before that decision.
