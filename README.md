# uterine-dynamics-paper2

Code and analysis plan for Paper 2 of the short transvaginal ultrasound uterine dynamics project.

## Scope

Paper 2 evaluates:
1. associations between Paper 1 robustness-qualified local uterine dynamic measurements and clinician-recorded uterine activity;
2. adjusted associations with clinical pregnancy (future stage);
3. exploratory incremental prediction beyond prespecified clinical information (future stage).

This repository does **not** contain raw patient videos, DICOM files, patient spreadsheets, identifying information, or Paper 1 tracking outputs.

## Current pipeline

- scripts/01_check_video_exam_date.py — verify video-displayed examination date.
- scripts/02_build_video_clinical_match.py — link video dates to the clinical workbook by case ID and audit date concordance.
- scripts/03_audit_patient_data.py — audit patient-level completeness and candidate analysis cohorts, preserving recorded zero activity counts.
- scripts/04_audit_paper1_feature_eligibility.py — consolidate Paper 1 robustness evidence and require an independently frozen PAPER1_ONLY feature-role decision file.
- scripts/05_build_paper2_analysis_master.py — build the verified patient-level master from the frozen feature gate, original clinician counts and video/date checks. The fixed feature roles for this analysis are F01 PRIMARY and F07/F09/F15 SECONDARY.
- scripts/06_analyze_clinician_activity_association.py — eight original frozen F01/F07/F09/F15 versus forward/reverse clinician-count comparisons; separate Holm primary and secondary families.
- scripts/07_explore_spatiotemporal_activity.py — **separate exploratory** associations of two newly derived F01 original-profile heterogeneity indices with forward/reverse counts (four comparisons, separate Holm family). Does **not** modify script 06 or promote new features into the Paper 1 gate.
- scripts/11_analyze_doppler_associations.py — separate exploratory analysis of four frozen features against six endometrial Doppler measures (24 comparisons), with quality and clinical sensitivity checks.
- scripts/12_doppler_sensitivity.py — earlier focused Doppler sensitivity checks retained for traceability.

Read docs/PAPER2_ACTIVITY_ASSOCIATION_PLAN.md for script-06 methods and docs/PAPER2_SPATIOTEMPORAL_EXPLORATION.md for script-07 inputs, method, and a ready-to-edit PowerShell command.

Read docs/PAPER2_DOPPLER_ASSOCIATIONS.md for the Doppler analysis inputs, method, and run commands. Patient-level source files and generated results stay outside Git.

Paper 1 source profiles and patient-level output CSVs, not aggregate stability tables, are required for script 07. They are read locally and are not committed to this repository. The new F01-derived temporal/spatial indices are exploratory and **not** separately qualified by the Paper 1 robustness gate.
