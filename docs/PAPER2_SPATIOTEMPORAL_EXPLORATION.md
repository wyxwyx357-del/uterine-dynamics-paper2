# Paper 2 exploratory temporal and spatial activity analyses (script 07)

This is a new post-primary exploratory analysis. It does not alter the frozen eight script-06 comparisons, change F01's primary role, select features using activity/pregnancy results, or claim that the new derived features passed the Paper 1 feature gate.

## Fixed script-07 definitions (before seeing script-07 results)

Source: Paper 1 ORIGINAL five-bin F01 (rsr_abs_median) patient-level normalized-time and normalized cervix-to-fundus position profiles. These are not the Grade-3 shadow values or population-level robustness summaries. Both Paper 1 scripts already write patient-level profiles and summaries with a source_manifest.json. No tracking rerun is required if those files exist.

- Temporal heterogeneity: median absolute deviation (MAD) of five original normalized-time bin medians, divided by their median.
- Spatial heterogeneity: MAD of five original normalized cervix-to-fundus position bin medians, divided by their median. The frozen normalized section axis is not physical distance or peristaltic propagation.
- A metric exists only when all five original bins are finite, each has at least one valid original measurement, and their median is strictly positive. Undefined profiles stay missing, never imputed.
- Use the original F01 activity-eligible patients, intersected separately with each available derived metric and each clinician count. Zero counts are valid.
- Four tests: temporal heterogeneity versus forward and reverse counts; spatial heterogeneity versus forward and reverse counts. One separate Holm family of four planned comparisons, retaining non-evaluable tests in family size.
- Statistics imported unchanged from script 06: average-rank Spearman, 9,999 two-sided paired permutations and 5,000 paired-patient percentile bootstrap resamples with independent fixed seeds. No clinical adjustment or pregnancy outcome analysis.

Script 07 checks script-05 master SHA256 and gate SHA256, Paper 1 source-manifest cases and five-bin contracts, original F01 aggregate agreement, and unique patient ID plus original video filename date. It rejects mismatches rather than combining different videos or versions.

## Inputs and execution

Paper 1 time output: 归一化时间分箱曲线_长表.csv, 患者级时间结构比较_长表.csv, source_manifest.json.
Paper 1 space output: 归一化空间分箱曲线_长表.csv, 患者级空间结构比较_长表.csv, source_manifest.json.

From the Paper 2 repository in PowerShell, replace the three placeholder directories:

~~~powershell
git pull origin main
python -m pytest -q tests/test_clinician_activity_association.py tests/test_explore_spatiotemporal_activity.py
$master = "C:\PATH\TO\SCRIPT05_OUTPUT"
$time = "C:\PATH\TO\PAPER1_TIME_OUTPUT"
$space = "C:\PATH\TO\PAPER1_SPACE_OUTPUT"
python scripts/07_explore_spatiotemporal_activity.py --master "$master\paper2_analysis_master.csv" --master-manifest "$master\manifest.json" --temporal-profiles "$time\归一化时间分箱曲线_长表.csv" --temporal-summary "$time\患者级时间结构比较_长表.csv" --temporal-source-manifest "$time\source_manifest.json" --spatial-profiles "$space\归一化空间分箱曲线_长表.csv" --spatial-summary "$space\患者级空间结构比较_长表.csv" --spatial-source-manifest "$space\source_manifest.json" --output "C:\PATH\TO\NEW_SCRIPT07_OUTPUT"
~~~

Choose a NEW output directory. Outputs are exploratory_patient_features.csv, exploratory_activity_associations.csv, REPORT.md and manifest.json with input/output SHA256 and analysis settings. Script 06 and its original outputs are unchanged.

**Interpretation:** these new heterogeneity ratios themselves have not been separately validated by Paper 1's robustness experiment. Association is descriptive and does not establish agreement, peristaltic wave frequency/direction, causality, or prediction.