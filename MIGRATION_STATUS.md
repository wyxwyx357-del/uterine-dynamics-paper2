# Migration status

## Completed
- Repository scope and privacy boundary frozen.
- Video examination-date audit migrated.
- Video-to-clinical workbook date linkage audit migrated.
- Patient-level Paper 2 completeness audit migrated.
- Raw clinical/image data and generated outputs are excluded from version control.

## Known pre-migration reference results
These are local reference results and are **not** committed as patient data:
- 334 unique case IDs passed the three-way date audit.
- 325 had both forward and reverse clinician-recorded activity values.
- 241 had parseable clinical pregnancy outcome.
- 240 met the preliminary complete-case check for the candidate prediction fields.

## Required regression acceptance before new experiments
Run scripts 01-03 on the same local inputs and verify that the migrated repository reproduces the reference cohort counts above. Any discrepancy must be resolved before Paper 2 association analyses.

## Not yet implemented
- Paper 1 -> Paper 2 feature eligibility audit.
- Frozen Paper 2 analysis master table builder.
- Activity association analysis.
- Clinical pregnancy association analysis.
- Exploratory incremental prediction analysis.

No Paper 2 feature may be selected using Paper 2 association P values, odds ratios, or AUC.
