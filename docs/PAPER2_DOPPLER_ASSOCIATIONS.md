# Exploratory Doppler associations

The current Paper 2 article centers on this later exploratory family. It compares the four previously frozen Paper 1 features (F01, F07, F09, F15) with six endometrial Doppler measures (S/D, PI, RI, VI, FI, VFI), giving 24 pairwise comparisons. It does not change the earlier clinician activity analyses or the supplementary pregnancy analysis. F09–VI became a proposed future validation focus only after the original-cohort results were reviewed.

## Code

- `scripts/11_analyze_doppler_associations.py`: complete 24-comparison analysis, Holm correction across all 24 tests, plots, and quality/clinical sensitivity checks.
- `scripts/12_doppler_sensitivity.py`: earlier focused, read-only checks for F09/F15 versus VI/VFI. It prints aggregate results only and is retained for traceability.
- `scripts/06_analyze_clinician_activity_association.py`: shared Spearman, permutation, bootstrap, and Holm functions used by script 11. Its statistical implementation is unchanged.

All source tables stay outside Git. Script 11 requires the frozen patient master CSV and its script-05 manifest, the exact patient audit workbook hashed by that manifest, a QC release directory containing `patient_quality.csv` and `PRIVATE_source_inventory.csv`, and the 14-case topology repair CSV. It verifies the master, audit, gate, case IDs, confirmed examination dates, and repair-list IDs/source dates before producing outputs. The inventory is used only for local linkage; no patient-level joined table is exported. The output directory must not exist before the run.

From this repository root, run:

```powershell
python -X utf8 scripts/11_analyze_doppler_associations.py `
  --master "PATH_TO/paper2_analysis_master.csv" `
  --master-manifest "PATH_TO/manifest.json" `
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

Script 11 can optionally enforce the four archived 2026-09-28 point estimates with `--verify-reference-rho` when the purpose is legacy-cohort reproduction. That gate is not a validity condition for a legitimately corrected cohort or future new patients. The documented QC and repair inputs remain required; new-cohort analysis rules should be frozen separately before use.
