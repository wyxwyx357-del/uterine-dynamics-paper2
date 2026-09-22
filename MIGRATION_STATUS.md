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

## Regression acceptance
Migration regression has passed after the minimal clinical-column resolution fix in `scripts/02_build_video_clinical_match.py`.

Verified reference results:
- 334 unique case IDs.
- 334 PASS_THREE_WAY.
- 325 activity-association candidates.
- 325 with both forward and reverse activity available.
- 241 pregnancy-association candidates.
- 240 preliminary prediction complete cases.

Patient-level case ID sets matched the pre-migration outputs. Zero-valued activity records remained valid observations.

## Not yet implemented
- Paper 1 -> Paper 2 feature eligibility audit.
- Frozen Paper 2 analysis master table builder.
- Activity association analysis.
- Clinical pregnancy association analysis.
- Exploratory incremental prediction analysis.

No Paper 2 feature may be selected using Paper 2 association P values, odds ratios, or AUC.
