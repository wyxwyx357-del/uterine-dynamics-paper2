# Paper 2 scope

## Current article question

How are local dynamic measurements from short transvaginal ultrasound videos associated with clinician-recorded uterine activity frequency and endometrial Doppler indices? The Doppler part is exploratory. The unit of analysis is the patient-level case record with a confirmed same-examination link.

## Research chronology and families

1. Paper 1 evidence and an external `PAPER1_ONLY` decision froze F01, F07, F09, and F15 before the later Doppler analysis. F01 was PRIMARY and the other three SECONDARY for the original activity plan.
2. The original activity analysis has two F01 forward/reverse comparisons in one Holm family and six F07/F09/F15 forward/reverse comparisons in another. Counts are per minute; zero is observed data.
3. Clinical pregnancy was examined separately and is supplementary to the present article. Its clinical-outcome derivation was amended after an initial analysis; see `docs/PAPER2_CLINICAL_PREGNANCY_PLAN.md`.
4. The later exploratory Doppler analysis compares four frozen features with S/D, PI, RI, VI, FI, and VFI, giving one planned 24-comparison Holm family. F09–VI was identified as a future validation focus after reviewing the original-cohort exploratory results. It was not an original primary hypothesis.
5. Script 07 and local scripts 09–10, when present, are separate measurement explorations. They do not extend the frozen four-feature family. Script 12 is a historical focused Doppler check.

## Interpretation and data boundaries

- These features are local video measurements. They do not establish a complete uterine peristaltic wave, contraction force, true tissue strain, or propagation direction.
- Each feature uses its own available patients. Missing F15 does not remove a patient's F01/F07/F09 results. No Paper 2 outcome or P value can change the frozen feature gate.
- No prediction, AUC, new feature selection, or new model is part of the present article plan.
- Input files and patient-level outputs remain outside Git. Formal claims require archived input files, hashes, code revision, cohort counts, and result manifests.
- Available DICOM does not establish identical Gain, PRF, or wall-filter settings for VI/FI/VFI across patients. Spectral Doppler settings cannot substitute for these acquisition settings.

See `docs/PAPER2_END_TO_END_FLOW.md` for the execution and audit chain.
