# Paper 2 clinical pregnancy association (script 08)

This is a separate pregnancy association analysis using the frozen script-05 master. It does not change scripts 02–07, the eight activity comparisons, feature roles, or the Paper 1 feature gate. No AUC, classifier, feature selection, or incremental prediction is performed.

## Outcome amendment after the first script-08 run

The initial 241-patient run treated `clinical_pregnancy = '-'` as missing. Source review showed that all 93 such rows had `biochemical_pregnancy = '否'`. The initial result therefore described clinical pregnancy only among biochemical-positive patients, not the full cohort. It is retained as an audit artifact and is superseded for the full-cohort question.

The amended full-cohort clinical outcome retains recorded clinical `是` as 1 and recorded `否` as 0. A clinical `-` is coded 0 **only when** the same source row records biochemical `否`. Any other biochemical/clinical combination is rejected. This rule is a documented post-result protocol amendment, not a result-blinded original prespecification. Script 08 reads the original patient-audit workbook named and hashed by the script-05 manifest, verifies its patient IDs and recorded clinical values against the master, and records the derivation counts in `outcome_derivation.json`. The frozen master itself remains unchanged. This assumes the two source fields describe the same treatment episode; that clinical provenance must be checked before a final clinical claim.

## Cohorts and model hierarchy

- F01 is PRIMARY. Its adjusted logistic association is the main pregnancy inference in this stage. F07, F09 and F15 are SECONDARY.
- The existing `eligible_pregnancy_Fxx` flags are checked against the recorded clinical outcome and the script-05 manifest. Because they exclude the structurally blank biochemical-negative rows, the amended analysis eligibility is derived separately as verified three-way audit pass, known amended outcome, and the feature's own available measurement. Features do not share a four-feature complete-case cohort.
- Each feature has three reported models: `UNADJUSTED_FULL` on its eligible cohort; `UNADJUSTED_MATCHED` on the same covariate-complete patients as the adjusted model; and `ADJUSTED` on those covariate-complete patients. No covariate is imputed.
- The fixed adjustment set contains numeric age, BMI, infertility years, transferred embryo count, and endometrial thickness; categorical infertility type, embryo type, endometrial type, and cycle type. No variable is selected or removed by its association result.
- The imaging coefficient is reported as an odds ratio (OR) for an increase of one feature-specific eligible-cohort interquartile range (IQR), with a Wald 95% confidence interval and two-sided Wald P value. The IQR is held fixed across that feature's three models. Numeric adjustment columns are centered and divided by their complete-case standard deviation for numerical stability; this does not change the imaging OR.
- Categories are converted to stripped strings, sorted lexically, and encoded by indicators for all but the first level. The first level is the reference; exact observed levels and indicators are recorded in `model_diagnostics.json`. The labels are not assigned clinical meanings by this code.
- The three adjusted secondary P values form one Holm family. Failed planned models stay in the family as P=1 for correction and retain a missing displayed P. Unadjusted models receive no multiplicity claim.

## Diagnostics and failure rules

The script verifies the master CSV and frozen gate hashes, feature roles, unique patient IDs, 0/1 pregnancy outcome, parseability and eligibility flags, and manifest cohort counts before analysis. Its cohort-flow file reports feature-specific positive/negative numbers, covariate missingness by column, and the adjusted complete-case size.

Each model requires both outcome classes, a nonconstant imaging feature, a full-rank design matrix, and no complete or quasi-complete separation. Separation is checked by a linear program before fitting. Category counts below five and outcome counts by level are recorded as sparsity diagnostics. A sparse category alone is not silently collapsed; if it makes a coefficient nonidentifiable, the model receives a failure status. Models that do not converge or yield nonfinite estimates have missing OR, CI and P. No fallback model, covariate deletion, category merge, or significance-driven alternative is run.

`pregnancy_associations.csv`, `pregnancy_cohort_flow.csv`, `model_diagnostics.json`, `outcome_derivation.json`, `manifest.json`, and `REPORT.md` are written to a new output directory. They contain no patient names or direct identifiers. The code revision and input/output hashes are recorded in the manifest. The results describe association, not causation. A biochemical-pregnancy association is not estimated by this script; the biochemical field supplies the clinical-outcome derivation rule.

## Execution

From this repository, after tests pass:

```powershell
python -m pytest -q
python scripts/08_analyze_clinical_pregnancy.py --master "C:\PATH\TO\SCRIPT05_OUTPUT\paper2_analysis_master.csv" --master-manifest "C:\PATH\TO\SCRIPT05_OUTPUT\manifest.json" --output "C:\PATH\TO\NEW_SCRIPT08_OUTPUT"
```

Use the verified formal script-05 master, not script-07's preliminary subset. If an adjusted model fails, retain its diagnostic status and review the underlying data and protocol before any separately declared sensitivity analysis.
