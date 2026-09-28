# Paper 2 clinician-recorded activity association (script 06)

This plan is fixed before running script 06 on the formal patient master. It tests association, not agreement, causation, or prediction.

- PRIMARY family (2 tests): F01 versus clinician-recorded forward activity frequency; F01 versus clinician-recorded reverse activity frequency.
- SECONDARY family (6 tests): F07, F09, and F15, each versus forward and reverse activity frequency. Both fields record events per minute, and a recorded 0 is a valid observation.
- Each comparison uses only its own `eligible_activity_Fxx` patients and a finite frozen measurement. No four-measurement complete-case intersection. Zero activity counts are valid; negative or noninteger counts are data errors.
- Statistic: Spearman rank correlation with average ranks for ties. No clinical covariates.
- Two-sided raw P: 9,999 random permutations of one variable's patient pairing, with the plus-one Monte Carlo correction and fixed seed 20260922 plus the comparison's fixed index (0 through 7). No asymptotic P-value substitution.
- 95% CI: percentile interval from 5,000 paired-patient bootstrap resamples, using seed 20270922 plus the same comparison index. Invalid constant-variable bootstrap replicates are counted and omitted; if none are valid, CI is unavailable. A comparison with fewer than 3 patients or a constant observed variable is NOT_EVALUABLE, with no rho/P/CI.
- Holm adjusted P values are calculated separately for the two prespecified families. Non-evaluable planned tests remain in the family as P=1 for correction and retain missing displayed P values.
- Report all eight comparisons, including non-evaluable ones. No result-dependent changes to tests, cohorts, features, covariates, or methods.
- Clinical pregnancy, regression, odds ratios, AUC, and prediction are out of scope.
