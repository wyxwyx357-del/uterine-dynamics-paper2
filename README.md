# uterine-dynamics-paper2

Code and analysis contracts for Paper 2: **exploratory associations between local measurements from short transvaginal ultrasound videos and endometrial Doppler indices**.

The four patient-level measurements were frozen from Paper 1 evidence before the Doppler analysis: F01 `rsr_abs_median` (PRIMARY in the original activity family), and F07 `cavity_width_strain_rate_abs_median`, F09 `longitudinal_wall_strain_rate_abs_median`, F15 `wall_curvature_change_rate_mm_inv_s_abs_median` (SECONDARY in that family). These are operational video measurements, not verified complete peristaltic waves, tissue contraction force, true tissue strain, or propagation direction.

Paper 2 retains the original eight associations with clinician-recorded forward/reverse **activities per minute**. The later 24 feature–Doppler comparisons are exploratory; F09–VI became a proposed future validation relationship only after those results were seen. Clinical pregnancy is a separate supplementary analysis. No prediction/AUC workflow is part of the current paper.

## Run and audit order

Read [the full execution flow](docs/PAPER2_END_TO_END_FLOW.md) before running any script. Source files and generated patient-level data stay outside Git.

1. `scripts/01_check_video_exam_date.py` → video examination-date audit.
2. `scripts/02_build_video_clinical_match.py` → video/clinical identity and date linkage.
3. `scripts/03_audit_patient_data.py` → patient-level audit, retaining recorded zero activity.
4. `scripts/04_audit_paper1_feature_eligibility.py` → F01–F20 evidence and external `PAPER1_ONLY` frozen decision.
5. `scripts/05_build_paper2_analysis_master.py` → frozen four-feature patient master and input/output hashes.
6. `scripts/06_analyze_clinician_activity_association.py` → original eight activity comparisons.
7. `scripts/07_explore_spatiotemporal_activity.py` → optional, separate F01-derived activity exploration.
8. `scripts/08_analyze_clinical_pregnancy.py` → supplementary pregnancy analysis with a documented post-result outcome amendment.
9. `scripts/11_analyze_doppler_associations.py` → current-paper 24-comparison exploratory Doppler analysis, requiring the exact script-05 master, manifest, and clinical audit workbook.

`scripts/12_doppler_sensitivity.py` preserves an earlier focused check and is not the 24-comparison pipeline. Scripts 09–10, when present in a working tree, are an outcome-blind wall-synchrony measurement exploration; they do not enter the frozen feature set or current Doppler results.

The repository does not contain raw videos, DICOM files, patient spreadsheets, direct identifiers, or Paper 1 tracking outputs. The upstream Paper 1 feature extraction, Doppler clinical extraction, outer-wall QC generation, and 14-case topology repair are external input-producing steps; their artifacts and hashes must be retained for formal acceptance.
