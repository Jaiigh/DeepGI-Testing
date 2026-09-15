import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from modules.procedure.controller import ProcedureController, atomic_write_json


RESULT = {"lesion_type": "polyp", "location": "sigmoid colon", "size_mm": 6,
          "procedure": "biopsy", "biopsy_forceps": True, "biopsy_pieces": 2, "pathology": True}


class ProcedureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.seconds = 100.0
        self.wall = datetime(2026, 9, 15, 12, tzinfo=timezone(timedelta(hours=7)))
        self.controller = ProcedureController(self.directory.name, monotonic=lambda: self.seconds,
                                              now=lambda: self.wall)

    def start_withdrawal(self):
        self.controller.start(" case-1 ", " patient-1 ")
        self.controller.start_withdrawal()

    def test_required_ids_and_transition_guards(self):
        controller = self.controller
        for ids in (("", "p"), ("c", " ")):
            with self.assertRaises(ValueError):
                controller.start(*ids)
        self.assertIsNone(controller.accept_trigger())
        for action in (controller.start_withdrawal, controller.end, controller.new_case):
            with self.assertRaises(ValueError):
                action()
        controller.start("case", "patient")
        self.assertIsNone(controller.accept_trigger())
        with self.assertRaises(ValueError):
            controller.start("another", "patient")
        with self.assertRaises(ValueError):
            controller.end()
        controller.start_withdrawal()
        with self.assertRaises(ValueError):
            controller.start_withdrawal()
        controller.end()
        with self.assertRaises(ValueError):
            controller.end()
        self.assertIsNone(controller.accept_trigger())

    def test_timer_freezes_at_end_despite_pending_finding_and_wall_clock_change(self):
        self.start_withdrawal()
        controller = self.controller
        self.seconds += 30
        finding_id = controller.accept_trigger()
        self.assertIsNone(controller.accept_trigger())
        self.wall -= timedelta(days=2)
        self.seconds += 12
        self.assertEqual(controller.duration, 42)
        controller.end()
        self.assertFalse(controller.finalized)
        self.assertIsNone(controller.accept_trigger())
        self.seconds += 100
        controller.add_finding(finding_id, "polyp", RESULT)
        controller.finish_finding(finding_id)
        self.assertTrue(controller.finalized)
        self.assertEqual(controller.duration, 42)
        self.assertEqual(controller.case["summary"]["withdrawal_duration_seconds"], 42)
        self.assertEqual(controller.case["findings"][0]["result"], RESULT)

    def test_event_order_json_export_and_isolation(self):
        self.start_withdrawal()
        controller = self.controller
        finding_id = controller.accept_trigger()
        controller.add_finding(finding_id, "พบ polyp", RESULT)
        controller.finish_finding(finding_id)
        controller.end()
        expected = ["CASE_START", "LANDMARK_DETECTED", "WITHDRAWAL_START",
                    "FINDING_DETECTED", "LANDMARK_DETECTED", "CASE_END"]
        self.assertEqual([event["type"] for event in controller.case["events"]], expected)
        first_path = controller.path
        saved = json.loads(first_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, controller.snapshot())
        self.assertTrue(saved["started_at"].endswith("+07:00"))
        export_path = Path(self.directory.name) / "export.json"
        controller.export(export_path)
        self.assertEqual(json.loads(export_path.read_text(encoding="utf-8")), saved)
        controller.new_case()
        self.assertEqual(controller.phase, "Ready")
        controller.start("case-1", "patient-1")
        self.assertNotEqual(controller.path, first_path)
        self.assertEqual(controller.case["findings"], [])
        self.assertEqual(controller.duration, 0)
        self.assertEqual(json.loads(first_path.read_text(encoding="utf-8")), saved)

    def test_zero_findings_and_id_not_used_as_path(self):
        self.controller.start("../../escape", "patient")
        self.controller.start_withdrawal()
        self.controller.end()
        self.assertTrue(self.controller.finalized)
        self.assertEqual(self.controller.case["summary"]["finding_count"], 0)
        self.assertEqual(self.controller.path.parent, Path(self.directory.name))

    def test_save_failure_retains_data_and_blocks_reset_and_export(self):
        self.start_withdrawal()
        controller = self.controller
        def fail(*args):
            raise OSError("disk full")
        controller._writer = fail
        finding_id = controller.accept_trigger()
        self.assertFalse(controller.add_finding(finding_id, "polyp", RESULT))
        controller.finish_finding(finding_id)
        controller.end()
        self.assertTrue(controller.dirty)
        self.assertEqual(controller.save_error, "disk full")
        self.assertEqual(len(controller.case["findings"]), 1)
        with self.assertRaises(ValueError):
            controller.new_case()
        with self.assertRaises(ValueError):
            controller.export(Path(self.directory.name) / "export.json")
        controller._writer = atomic_write_json
        self.assertTrue(controller.save())
        self.assertFalse(controller.dirty)
        controller.new_case()

    def test_interruption_preserves_pending_outcome_without_case_end(self):
        self.start_withdrawal()
        controller = self.controller
        finding_id = controller.accept_trigger()
        self.seconds += 10
        controller.interrupt()
        self.assertIsNone(controller.accept_trigger())
        self.seconds += 20
        controller.record_error("Extracting", "bad JSON", "polyp", finding_id)
        controller.finish_finding(finding_id)
        self.assertEqual(controller.duration, 10)
        self.assertTrue(controller.case["incomplete"])
        self.assertFalse(controller.finalized)
        self.assertIsNone(controller.case["ended_at"])
        self.assertNotIn("CASE_END", [e["type"] for e in controller.case["events"]])
        self.assertEqual(controller.case["processing_errors"][0]["transcription"], "polyp")

    def test_stale_duplicate_findings_and_pending_export_rejected(self):
        self.start_withdrawal()
        controller = self.controller
        finding_id = controller.accept_trigger()
        with self.assertRaises(ValueError):
            controller.add_finding("stale", "polyp", RESULT)
        controller.add_finding(finding_id, "polyp", RESULT)
        with self.assertRaises(ValueError):
            controller.add_finding(finding_id, "polyp", RESULT)
        controller.end()
        with self.assertRaises(ValueError):
            controller.export("unused.json")


if __name__ == "__main__":
    unittest.main()
