"""Caecum signals work without a GUI, audio device, or model packages."""

import copy
import tempfile
import unittest
from unittest.mock import patch

from modules.caecum import CaecumDetected, CaecumModule
from modules.procedure.controller import ProcedureController


class CaecumTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        renderer = patch("modules.procedure.controller.generate_pdf_from_json")
        renderer.start()
        self.addCleanup(renderer.stop)
        self.seconds = 10.0
        self.controller = ProcedureController(directory.name, monotonic=lambda: self.seconds)
        self.module = CaecumModule(self.controller)

    def test_detection_records_existing_events_and_starts_timer_once(self):
        self.controller.start("case", "patient")
        event = CaecumDetected(self.controller.case["internal_id"], "vision")
        self.module.handle(event)
        self.assertEqual(self.controller.phase, "Withdrawal")
        self.assertEqual([e["type"] for e in self.controller.case["events"]],
                         ["CASE_START", "LANDMARK_DETECTED", "WITHDRAWAL_START"])
        self.assertEqual(self.controller.case["events"][1]["location"], "CECUM")
        self.seconds += 12
        before = copy.deepcopy(self.controller.case)
        with self.assertRaises(ValueError):
            self.module.handle(event)
        self.assertEqual(self.controller.case, before)
        self.assertEqual(self.controller.duration, 12)

    def test_ready_completed_and_incomplete_cases_reject_without_mutation(self):
        with self.assertRaises(ValueError):
            self.module.handle(CaecumDetected("not-a-case"))
        self.assertIsNone(self.controller.case)
        self.controller.start("case", "patient")
        event = CaecumDetected(self.controller.case["internal_id"])
        self.module.handle(event)
        self.controller.end()
        before = self.controller.snapshot()
        with self.assertRaises(ValueError):
            self.module.handle(event)
        self.assertEqual(before, self.controller.snapshot())
        self.controller.new_case()
        self.controller.start("case", "patient")
        event = CaecumDetected(self.controller.case["internal_id"])
        self.controller.interrupt()
        before = self.controller.snapshot()
        with self.assertRaises(ValueError):
            self.module.handle(event)
        self.assertEqual(before, self.controller.snapshot())

    def test_old_internal_id_rejected_even_when_entered_case_id_is_reused(self):
        self.controller.start("same-case", "same-patient")
        old_event = CaecumDetected(self.controller.case["internal_id"])
        self.module.handle(old_event)
        self.controller.end()
        self.controller.new_case()
        self.controller.start("same-case", "same-patient")
        before = self.controller.snapshot()
        with self.assertRaises(ValueError):
            self.module.handle(old_event)
        self.assertEqual(before, self.controller.snapshot())


if __name__ == "__main__":
    unittest.main()
