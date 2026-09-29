"""End-to-end synthetic smoke test for script 11.

This test uses only generated, patient-free data. It exercises the real script-11
file/merge/QC/repair/output path while replacing the computationally expensive
24-association Monte Carlo loop and sensitivity bootstrap with deterministic
fast stand-ins. Unit tests cover the underlying statistics separately.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


activity = load("activity_e2e", "06_analyze_clinician_activity_association.py")
doppler = load("doppler_e2e", "11_analyze_doppler_associations.py")


class DopplerEndToEndSyntheticTest(unittest.TestCase):
    def build_inputs(self, root: Path, n: int = 30):
        ids = [str(1000 + i) for i in range(n)]
        dates = [f"202401{(i % 28) + 1:02d}" for i in range(n)]

        master = pd.DataFrame({
            "case_id": ids,
            "matched_exam_date8": dates,
            "video_real_exam_date8": dates,
            "final_audit_status": ["PASS_THREE_WAY"] * n,
            "date_audit_pass": [True] * n,
            "paper1_filename_date8": dates,
            "peristalsis_forward_count": np.arange(n) % 4,
            "peristalsis_reverse_count": (np.arange(n) + 1) % 4,
            "female_age": 28 + (np.arange(n) % 9),
            "female_bmi": 19.0 + (np.arange(n) % 7) * 0.8,
            "endometrial_thickness_mm": 7.0 + (np.arange(n) % 8) * 0.4,
            "cycle_type": np.where(np.arange(n) % 2 == 0, "natural", "artificial"),
            "endometrial_type": np.where(np.arange(n) % 3 == 0, "A", "B"),
        })
        for j, (feature, column) in enumerate(activity.FEATURES.items(), start=1):
            master[column] = 0.5 * j + np.arange(n) * (0.01 + j * 0.001)
            master[f"eligible_activity_{feature}"] = True

        master_path = root / "paper2_analysis_master.csv"
        master.to_csv(master_path, index=False)

        audit = pd.DataFrame({
            "case_id": ids,
            "matched_exam_date8": dates,
            "video_real_exam_date8": dates,
            "final_audit_status": ["PASS_THREE_WAY"] * n,
            "flow_sd": 1.5 + np.arange(n) * 0.01,
            "flow_pi": 0.8 + np.arange(n) * 0.005,
            "flow_ri": 0.5 + np.arange(n) * 0.003,
            "flow_vi": 10.0 + np.arange(n) * 0.2,
            "flow_fi": 20.0 + np.arange(n) * 0.15,
            "flow_vfi": 4.0 + np.arange(n) * 0.08,
            "treatment_protocol": np.where(np.arange(n) % 2 == 0, "P1", "P2"),
        })
        audit_path = root / "patient_audit.xlsx"
        audit.to_excel(audit_path, sheet_name="02_患者级状态", index=False)

        gate_path = root / "paper2_feature_gate.csv"
        gate_path.write_text("synthetic frozen gate\n", encoding="utf-8")

        manifest = {
            "output_sha256": {"master_csv": doppler.sha(master_path)},
            "input_sha256": {"patient_audit": doppler.sha(audit_path)},
            "input_paths": {"frozen_gate": str(gate_path)},
            "frozen_gate_sha256": doppler.sha(gate_path),
            "eligible_feature_ids": list(activity.FEATURES),
            "primary": "F01",
            "secondary": ["F07", "F09", "F15"],
            "decision_basis": "PAPER1_ONLY",
            "cohort_counts": {
                "base_rows": n,
                **{f"activity_{feature}_eligible": n for feature in activity.FEATURES},
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        qc_dir = root / "qc"
        qc_dir.mkdir()
        analysis_ids = [f"A{i:03d}" for i in range(n)]
        inventory = pd.DataFrame({
            "case_id": [f"SYN_{case_id}_{date}" for case_id, date in zip(ids, dates)],
            "analysis_id": analysis_ids,
        })
        inventory.to_csv(qc_dir / "PRIVATE_source_inventory.csv", index=False)

        quality = pd.DataFrame({
            "analysis_id": analysis_ids,
            "status": ["OK"] * n,
            "F01": master[activity.FEATURES["F01"]].to_numpy(),
            "global_speed_median_original_px_s": 1.0 + np.arange(n) * 0.02,
            "outer_all_fb_p95_original_px": 0.5 + np.arange(n) * 0.01,
            "fps": 25 + (np.arange(n) % 3),
            "duration_s": 50 + (np.arange(n) % 5),
            "outer_all_pcc_median": 0.8 + (np.arange(n) % 4) * 0.02,
            "formal_pair_valid_ratio": 0.7 + (np.arange(n) % 5) * 0.03,
        })
        # Script 11 allows F01 differences for the documented repaired cases only.
        quality.loc[:13, "F01"] += 0.001
        quality.to_csv(qc_dir / "patient_quality.csv", index=False)

        repair_path = root / "topology_repair.csv"
        pd.DataFrame({
            "case_id": [f"SYN_{ids[i]}_{dates[i]}" for i in range(14)]
        }).to_csv(repair_path, index=False)

        return master_path, manifest_path, audit_path, qc_dir, repair_path

    @staticmethod
    def write_fast_stats(root: Path) -> Path:
        """Wrapper around the real script-06 statistics with small Monte Carlo counts."""
        real = ROOT / "scripts" / "06_analyze_clinician_activity_association.py"
        wrapper = root / "fast_stats.py"
        wrapper.write_text(
            "from pathlib import Path\n"
            "import importlib.util\n"
            f"REAL = Path({str(real)!r})\n"
            "spec = importlib.util.spec_from_file_location('real_activity_stats', REAL)\n"
            "real = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(real)\n"
            "real.PERMUTATIONS = 9\n"
            "real.BOOTSTRAPS = 20\n"
            "FEATURES = real.FEATURES\n"
            "validate_master = real.validate_master\n"
            "spearman = real.spearman\n"
            "holm_with_planned_family = real.holm_with_planned_family\n"
            "def association(x, y, index):\n"
            "    real.PERMUTATIONS = 9\n"
            "    real.BOOTSTRAPS = 20\n"
            "    return real.association(x, y, index)\n",
            encoding="utf-8",
        )
        return wrapper

    @staticmethod
    def fast_boot_partial(a, seed, b=1000):
        value = doppler.partial(a)
        return value - 0.05 if np.isfinite(value) else np.nan, \
            value + 0.05 if np.isfinite(value) else np.nan, 10

    def test_script11_full_synthetic_path_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            master, manifest, audit, qc_dir, repair = self.build_inputs(root)
            fast_stats = self.write_fast_stats(root)
            output = root / "out"

            argv = [
                "11_analyze_doppler_associations.py",
                "--master", str(master),
                "--master-manifest", str(manifest),
                "--audit", str(audit),
                "--qc", str(qc_dir),
                "--repair", str(repair),
                "--output", str(output),
            ]

            old_argv = sys.argv
            try:
                sys.argv = argv
                with mock.patch.object(doppler, "STAT", fast_stats), \
                        mock.patch.object(doppler, "boot_partial", self.fast_boot_partial), \
                        contextlib.redirect_stdout(io.StringIO()):
                    doppler.main()
            finally:
                sys.argv = old_argv

            expected_files = {
                "availability.csv",
                "associations_24.csv",
                "adjusted_sensitivities.csv",
                "outlier_sensitivities.csv",
                "quality_strata_descriptive.csv",
                "quality_correlations_descriptive.csv",
                "01_distributions.png",
                "02_raw_scatter.png",
                "03_rank_scatter.png",
                "clinical_metadata.json",
                "REPORT.md",
                "manifest.json",
            }
            self.assertTrue(expected_files.issubset({p.name for p in output.iterdir()}))

            associations = pd.read_csv(output / "associations_24.csv")
            self.assertEqual(len(associations), 24)
            self.assertEqual(
                set(zip(associations["feature"], associations["doppler"])),
                {(f, d) for f in activity.FEATURES for d in doppler.DOPPLER},
            )

            adjusted = pd.read_csv(output / "adjusted_sensitivities.csv")
            self.assertEqual(len(adjusted), 4 * (len(doppler.MODELS) + 1))
            self.assertEqual(
                set(adjusted["model"]),
                {*doppler.MODELS, "motion_error_exclude_14_repaired"},
            )

            robust = pd.read_csv(output / "outlier_sensitivities.csv")
            self.assertEqual(len(robust), 4)

            strata = pd.read_csv(output / "quality_strata_descriptive.csv")
            self.assertEqual(len(strata), 4 * 2 * 2)

            availability = pd.read_csv(output / "availability.csv")
            self.assertEqual(
                set(doppler.DOPPLER).issubset(set(availability["field"])),
                True,
            )

            recorded = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["status"], "complete")
            self.assertTrue(recorded["exploratory"])
            self.assertEqual(recorded["base_n"], 30)
            self.assertEqual(len(recorded["input_sha256"]), 9)

            report = (output / "REPORT.md").read_text(encoding="utf-8")
            self.assertIn("24项分析", report)
            self.assertIn("临床、采集及质量敏感性", report)


if __name__ == "__main__":
    unittest.main()
