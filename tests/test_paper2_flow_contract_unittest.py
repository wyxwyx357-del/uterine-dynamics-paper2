"""Synthetic, patient-free checks runnable with unittest or pytest."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


activity = load("activity_contract", "06_analyze_clinician_activity_association.py")
doppler = load("doppler_contract", "11_analyze_doppler_associations.py")
video = load("video_contract", "02_build_video_clinical_match.py")
audit_script = load("audit_contract", "03_audit_patient_data.py")
master_script = load("master_contract", "05_build_paper2_analysis_master.py")


class Paper2FlowContractTest(unittest.TestCase):
    def test_patient_audit_keeps_zero_and_omits_prediction_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "three_way.xlsx"
            output = Path(temp) / "patient_audit.xlsx"
            pd.DataFrame({
                "case_id": ["101", "102"],
                "final_audit_status": ["PASS_THREE_WAY", "PASS_THREE_WAY"],
                "peristalsis_forward_count": [0, None],
                "peristalsis_reverse_count": [None, None],
                "clinical_pregnancy": ["是", "否"],
            }).to_excel(source, sheet_name=audit_script.INPUT_SHEET, index=False)
            old_argv = sys.argv
            try:
                sys.argv = ["03_audit_patient_data.py", "--input", str(source),
                            "--output", str(output)]
                with contextlib.redirect_stdout(io.StringIO()):
                    audit_script.main()
            finally:
                sys.argv = old_argv
            with pd.ExcelFile(output) as workbook:
                self.assertFalse(any("预测候选" in sheet for sheet in workbook.sheet_names))
            table = pd.read_excel(output, sheet_name="02_患者级状态")
            self.assertTrue(bool(table.loc[0, "eligible_activity_association"]))
            self.assertFalse(bool(table.loc[1, "eligible_activity_association"]))
            self.assertEqual(table.loc[0, "peristalsis_forward_count"], 0)
            self.assertNotIn("prediction_candidate_complete_case", table)

    def test_ambiguous_case_id_and_conflicting_video_identity(self):
        for module in (video, audit_script):
            with self.assertRaisesRegex(ValueError, "ambiguous case_id"):
                module.clean_case_id("patient 101 or 102")
            self.assertEqual(module.clean_case_id("patient 00101"), "101")
        row = {"patient_folder": "A_101_20240101", "video_filename": "B_102_20240101.mp4",
               "video_relpath": "A_101_20240101/B_102_20240101.mp4", "name_date": "20240101"}
        self.assertEqual(video.parse_video_identity(row)[2], "CONFLICTING_CASE_ID")
        self.assertEqual(video.parse_video_identity(row)[0], "")
        row["video_filename"] = "B_101_20240102.mp4"
        row["video_relpath"] = "A_101_20240101/B_101_20240102.mp4"
        self.assertEqual(video.parse_video_identity(row)[2], "CONFLICTING_FILENAME_DATE")

    def test_master_recomputes_inherited_eligibility(self):
        base = {column: "MATCH" for column in master_script.STATUS_COLUMNS}
        base.update({column: 1 for column in master_script.CLINICAL_COLUMNS})
        base.update(case_id="101", video_filenames="A_101_20240101.mp4",
                    clinical_pregnancy_binary_qc=1,
                    video_real_exam_date8="20240101", matched_exam_date8="20240101",
                    final_audit_status="PASS_THREE_WAY", date_audit_pass=True,
                    eligible_activity_association=False, eligible_pregnancy_association=True,
                    clinical_pregnancy_parseable_01=True,
                    peristalsis_forward_count=0, peristalsis_reverse_count=0)
        gate_rows = []
        for index in range(1, 21):
            feature = f"F{index:02d}"
            role = "PRIMARY" if feature == "F01" else "SECONDARY" if feature in {
                "F07", "F09", "F15"} else "NONE"
            gate_rows.append({"feature_id": feature,
                              "feature_name": master_script.FEATURE_NAMES.get(feature, f"other_{feature}"),
                              "paper2_decision": "EXCLUDE" if role == "NONE" else "INCLUDE",
                              "paper2_role": role, "paper2_eligible": role != "NONE",
                              "paper2_decision_basis": "PAPER1_ONLY"})
        gate = pd.DataFrame(gate_rows)
        features = pd.DataFrame([{"case_id": "A_101_20240101", **{
            row["feature_name"]: 1.0 for row in gate_rows}}])
        video_audit = pd.DataFrame([{"video_filename": "A_101_20240101.mp4",
                                     "name_date": "20240101", "ocr_date": "20240101",
                                     "ocr_status": "UNIQUE_OCR_DATE"}])
        with self.assertRaisesRegex(ValueError, "activity eligibility disagrees"):
            master_script.build_master(pd.DataFrame([base]), features, gate, video_audit)
        base["eligible_activity_association"] = True
        base["date_audit_pass"] = False
        with self.assertRaisesRegex(ValueError, "date eligibility disagrees"):
            master_script.build_master(pd.DataFrame([base]), features, gate, video_audit)
        base["date_audit_pass"] = True
        features.loc[0, master_script.FEATURE_NAMES["F15"]] = np.nan
        built, _ = master_script.build_master(pd.DataFrame([base]), features, gate, video_audit)
        self.assertTrue(bool(built.loc[0, "eligible_activity_F01"]))
        self.assertTrue(bool(built.loc[0, "eligible_activity_F07"]))
        self.assertTrue(bool(built.loc[0, "eligible_activity_F09"]))
        self.assertFalse(bool(built.loc[0, "eligible_activity_F15"]))

    def make_sources(self, directory: Path):
        ids = ["101", "102", "103"]
        master = pd.DataFrame({
            "case_id": ids, "matched_exam_date8": ["20240101", "20240102", "20240103"],
            "video_real_exam_date8": ["20240101", "20240102", "20240103"],
            "final_audit_status": ["PASS_THREE_WAY"] * 3,
            "date_audit_pass": [True] * 3,
            "peristalsis_forward_count": [0, 1, 2],
            "peristalsis_reverse_count": [2, 1, 0],
        })
        for feature, column in activity.FEATURES.items():
            master[column] = [1.0, 2.0, 3.0]
            master[f"eligible_activity_{feature}"] = [True] * 3
        audit = master[["case_id", "matched_exam_date8", "video_real_exam_date8",
                        "final_audit_status"]].copy()
        master_path = directory / "paper2_analysis_master.csv"
        audit_path = directory / "patient_audit.xlsx"
        gate_path = directory / "gate.csv"
        manifest_path = directory / "manifest.json"
        master.to_csv(master_path, index=False)
        audit.to_excel(audit_path, sheet_name="02_患者级状态", index=False)
        gate_path.write_text("frozen synthetic gate\n", encoding="utf-8")
        manifest = {
            "output_sha256": {"master_csv": doppler.sha(master_path)},
            "input_sha256": {"patient_audit": doppler.sha(audit_path)},
            "input_paths": {"frozen_gate": str(gate_path)},
            "frozen_gate_sha256": doppler.sha(gate_path),
            "eligible_feature_ids": list(activity.FEATURES),
            "primary": "F01", "secondary": ["F07", "F09", "F15"],
            "decision_basis": "PAPER1_ONLY",
            "cohort_counts": {"base_rows": 3, **{
                f"activity_{feature}_eligible": 3 for feature in activity.FEATURES}},
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return master_path, manifest_path, audit_path, gate_path, manifest

    def test_frozen_master_audit_gate_and_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.make_sources(Path(temp))
            master, audit, gate, _ = doppler.verified_inputs(*paths[:3], activity)
            self.assertEqual(len(master), 3)
            self.assertEqual(len(audit), 3)
            self.assertEqual(gate, paths[3])
            paths[3].write_text("changed gate\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gate SHA256"):
                doppler.verified_inputs(*paths[:3], activity)

    def test_changed_master_and_audit_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            master_path, manifest_path, audit_path, _, _ = self.make_sources(Path(temp))
            master_path.write_text(master_path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "Master CSV SHA256"):
                doppler.verified_inputs(master_path, manifest_path, audit_path, activity)
        with tempfile.TemporaryDirectory() as temp:
            master_path, manifest_path, audit_path, _, _ = self.make_sources(Path(temp))
            table = pd.read_excel(audit_path, sheet_name="02_患者级状态")
            table.loc[0, "matched_exam_date8"] = 20240201
            table.to_excel(audit_path, sheet_name="02_患者级状态", index=False)
            with self.assertRaisesRegex(ValueError, "Patient audit SHA256"):
                doppler.verified_inputs(master_path, manifest_path, audit_path, activity)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["input_sha256"]["patient_audit"] = doppler.sha(audit_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "matched_exam_date8 differs"):
                doppler.verified_inputs(master_path, manifest_path, audit_path, activity)

    def test_repair_inventory_requires_exact_ids_and_source_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            repair_path = Path(temp) / "repair.csv"
            ids = [str(100 + index) for index in range(14)]
            master = pd.DataFrame({"case_id": ids,
                                   "paper1_filename_date8": ["20240101"] * 14})
            repaired = pd.DataFrame({"case_id": [f"A_{case}_20240101" for case in ids]})
            repaired.to_csv(repair_path, index=False)
            self.assertEqual(doppler.verified_repair_ids(repair_path, master), set(ids))
            repaired.loc[0, "case_id"] = "A_100_20240102"
            repaired.to_csv(repair_path, index=False)
            with self.assertRaisesRegex(ValueError, "source date differs"):
                doppler.verified_repair_ids(repair_path, master)
            repaired = repaired.iloc[:-1]
            repaired.to_csv(repair_path, index=False)
            with self.assertRaisesRegex(ValueError, "14 distinct"):
                doppler.verified_repair_ids(repair_path, master)

    def test_all_24_planned_comparisons_keep_constant_variable(self):
        table = pd.DataFrame({column: [1., 2., 3., 4., 5.] for column in doppler.DOPPLER})
        for feature, column in activity.FEATURES.items():
            table[column] = [1., 2., 3., 4., 5.] if feature != "F15" else [1.] * 5
        old_permutations, old_bootstraps = activity.PERMUTATIONS, activity.BOOTSTRAPS
        try:
            activity.PERMUTATIONS, activity.BOOTSTRAPS = 9, 20
            with contextlib.redirect_stdout(io.StringIO()):
                result = doppler.association_table(table, activity, activity.FEATURES)
        finally:
            activity.PERMUTATIONS, activity.BOOTSTRAPS = old_permutations, old_bootstraps
        self.assertEqual(len(result), 24)
        self.assertEqual(result.loc[result.feature.eq("F15"), "status"].tolist(),
                         ["NOT_EVALUABLE"] * 6)
        self.assertTrue(result.loc[result.feature.eq("F15"), "holm_p_24"].isna().all())
        self.assertTrue(np.isfinite(result.loc[result.feature.eq("F01"), "holm_p_24"]).all())


if __name__ == "__main__":
    unittest.main()
