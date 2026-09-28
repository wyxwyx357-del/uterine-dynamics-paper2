# Exploratory Doppler associations

The Paper 2 Doppler analysis is a separate exploratory family. It compares the four frozen Paper 1 features (F01, F07, F09, F15) with six endometrial Doppler measures (SD, PI, RI, VI, FI, VFI), giving 24 pairwise comparisons. It does not change the prespecified clinician activity or pregnancy analyses.

## Code

- `scripts/11_analyze_doppler_associations.py`: complete 24-comparison analysis, Holm correction across all 24 tests, plots, and quality/clinical sensitivity checks.
- `scripts/12_doppler_sensitivity.py`: earlier focused, read-only checks for F09/F15 versus VI/VFI. It prints aggregate results only and is retained for traceability.
- `scripts/06_analyze_clinician_activity_association.py`: shared Spearman, permutation, bootstrap, and Holm functions used by script 11. Its statistical implementation is unchanged.

All source tables stay outside Git. Script 11 requires the frozen patient master CSV, patient audit workbook, QC release directory containing `patient_quality.csv` and `PRIVATE_source_inventory.csv`, and the 14-case topology repair CSV. The inventory is used only for local linkage; no patient-level joined table is exported. The output directory must not exist before the run.

From this repository root, run:

```powershell
python -X utf8 scripts/11_analyze_doppler_associations.py `
  --master "PATH_TO/paper2_analysis_master.csv" `
  --audit "PATH_TO/patient_audit.xlsx" `
  --qc "PATH_TO/qc_release_directory" `
  --repair "PATH_TO/14_case_topology_repair.csv" `
  --output "outputs/doppler_run"
```

For the earlier focused check:

```powershell
python -X utf8 scripts/12_doppler_sensitivity.py `
  --master "PATH_TO/paper2_analysis_master.csv" `
  --audit "PATH_TO/patient_audit.xlsx"
```

The verified 2026-09-28 run is retained in the parent workspace at `analysis_outputs/doppler_complete_20260928/results_verified/`. Its README explains the source and quality checks. The sibling `results/` directory was an initial intermediate run with an invalid quality join and must not be used for reporting.

Script 11 reproduces the frozen 2026-09-28 cohort and explicitly checks four previously reported point estimates. It requires the documented QC and repair inputs, so it is not a generic analysis for new patients. New-cohort analysis rules should be frozen separately before use.
