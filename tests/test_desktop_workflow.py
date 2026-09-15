"""Headless integration tests using the desktop's real message handlers."""

import json
import queue
import tempfile
import threading
import time
import unittest

from desktop_app import ProcedureApp
from modules.procedure.controller import ProcedureController
from modules.procedure.voice_worker import VoiceWorker


RESULT = {"lesion_type": "polyp", "location": "rectum", "size_mm": 5,
          "procedure": None, "biopsy_forceps": False, "biopsy_pieces": None, "pathology": False}


class Value:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeBackend:
    def __init__(self):
        self.triggers = queue.Queue()
        self.release = threading.Event()
        self.processing = threading.Event()
        self.closed = False
        self.prepare_failure = False
        self.failure_stage = None
        self.empty = False
        self.feedback_saved = None
        self.prepared = 0

    def prepare(self):
        self.prepared += 1
        if self.prepare_failure:
            raise RuntimeError("model missing")

    def wait(self, cancel):
        while not cancel.is_set():
            try:
                value = self.triggers.get(timeout=0.01)
                if isinstance(value, Exception):
                    raise value
                return value
            except queue.Empty:
                pass
        return False

    def process(self, status, result):
        self.processing.set()
        status("Recording")
        if not self.release.wait(timeout=3):
            raise RuntimeError("test did not release finding")
        if self.empty:
            return None
        status("Extracting")
        if self.failure_stage != "Extracting":
            self.feedback_saved = result("five millimeter polyp", RESULT)
        if self.failure_stage:
            error = RuntimeError("simulated failure")
            error.stage = self.failure_stage
            error.transcription = "five millimeter polyp"
            raise error
        return RESULT

    def close(self):
        self.closed = True


class DesktopWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.backend = FakeBackend()
        self.worker = VoiceWorker(self.backend)
        self.app = ProcedureApp.__new__(ProcedureApp)
        self.app.worker = self.worker
        self.app.controller = ProcedureController(self.directory.name)
        self.app.models_ready = False
        self.app.loading = True
        self.app.listening = False
        self.app.closing = False
        self.app.voice_error = False
        self.app.worker_stopped = False
        self.app.notice_text = Value()
        self.app.voice_text = Value()
        self.app.case_id = Value("CASE-1")
        self.app.patient_id = Value("PATIENT-1")
        self.app._refresh = lambda: None
        self.app._refresh_controls = lambda: None
        self.worker.start()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.backend.release.set()
        self.worker.shutdown()
        self.pump_until(lambda: self.app.worker_stopped)
        self.worker.join(timeout=1)
        self.assertFalse(self.worker.is_alive())
        self.directory.cleanup()

    def pump_until(self, condition):
        deadline = time.monotonic() + 4
        while not condition():
            self.assertLess(time.monotonic(), deadline, "worker/UI handshake timed out")
            try:
                message = self.worker.messages.get(timeout=0.05)
            except queue.Empty:
                continue
            self.app._handle(message)

    def start(self):
        self.worker.prepare()
        self.pump_until(lambda: self.app.models_ready)
        self.app.start_case()
        self.app.start_withdrawal()

    def begin_finding(self):
        self.backend.triggers.put(True)
        self.pump_until(lambda: self.app.controller.pending is not None)
        self.assertTrue(self.backend.processing.wait(1))

    def test_end_during_finding_freezes_timer_and_waits_for_result(self):
        self.start()
        self.begin_finding()
        self.app.end_case()
        frozen = self.app.controller.duration
        self.assertFalse(self.app.controller.finalized)
        self.backend.release.set()
        self.pump_until(lambda: not self.app.listening)
        self.assertTrue(self.app.controller.finalized)
        self.assertEqual(self.app.controller.duration, frozen)
        self.assertTrue(self.backend.feedback_saved)
        saved = json.loads(self.app.controller.path.read_text())
        self.assertEqual(saved["summary"]["finding_count"], 1)
        self.assertEqual(saved["findings"][0]["result"], RESULT)
        self.assertIsNone(saved["pending_finding"])

    def test_end_before_trigger_acceptance_rejects_queued_trigger(self):
        self.start()
        self.backend.triggers.put(True)
        while True:
            message = self.worker.messages.get(timeout=2)
            if message.kind == "trigger":
                break
            self.app._handle(message)
        self.app.end_case()
        self.app._handle(message)
        self.pump_until(lambda: not self.app.listening)
        self.assertFalse(self.backend.processing.is_set())
        self.assertEqual(self.app.controller.case["findings"], [])
        self.assertTrue(self.app.controller.finalized)

    def test_microphone_failure_preserves_controls_and_allows_retry(self):
        self.start()
        self.backend.triggers.put(RuntimeError("microphone disconnected"))
        self.pump_until(lambda: not self.app.listening)
        self.assertEqual(self.app.controller.phase, "Withdrawal")
        self.assertTrue(self.app.voice_error)
        self.app.retry_voice()
        self.begin_finding()
        self.app.end_case()
        self.backend.release.set()
        self.pump_until(lambda: not self.app.listening)
        self.assertEqual(len(self.app.controller.case["findings"]), 1)

    def test_initialization_failure_can_retry(self):
        self.backend.prepare_failure = True
        self.worker.prepare()
        self.pump_until(lambda: not self.app.loading)
        self.app.start_case()
        self.assertEqual(self.app.controller.phase, "Ready")
        self.backend.prepare_failure = False
        self.app.retry_voice()
        self.pump_until(lambda: self.app.models_ready)
        self.assertEqual(self.backend.prepared, 2)

    def test_extraction_error_preserves_transcript_and_finalizes(self):
        self.start()
        self.backend.failure_stage = "Extracting"
        self.begin_finding()
        self.app.end_case()
        self.backend.release.set()
        self.pump_until(lambda: not self.app.listening)
        self.assertTrue(self.app.controller.finalized)
        self.assertEqual(len(self.app.controller.case["findings"]), 0)
        error = self.app.controller.case["processing_errors"][0]
        self.assertEqual(error["transcription"], "five millimeter polyp")
        self.assertEqual(error["stage"], "Extracting")

    def test_tts_error_does_not_remove_saved_finding(self):
        self.start()
        self.backend.failure_stage = "Speaking"
        self.begin_finding()
        self.app.end_case()
        self.backend.release.set()
        self.pump_until(lambda: not self.app.listening)
        self.assertEqual(len(self.app.controller.case["findings"]), 1)
        self.assertEqual(self.app.controller.case["summary"]["processing_error_count"], 1)

    def test_empty_recording_returns_to_listening_without_a_finding(self):
        self.start()
        self.backend.empty = True
        self.begin_finding()
        self.backend.release.set()
        self.pump_until(lambda: self.app.controller.pending is None)
        self.assertEqual(self.app.controller.case["findings"], [])
        self.assertIn("No finding heard", self.app.notice_text.get())
        self.assertTrue(self.app.listening)
        self.app.end_case()
        self.pump_until(lambda: not self.app.listening)

    def test_shutdown_finishes_accepted_finding_in_incomplete_case(self):
        self.start()
        self.begin_finding()
        self.app.controller.interrupt()
        self.app.closing = True
        self.worker.shutdown()
        self.backend.release.set()
        self.pump_until(lambda: self.app.worker_stopped)
        saved = json.loads(self.app.controller.path.read_text())
        self.assertTrue(self.backend.closed)
        self.assertTrue(saved["incomplete"])
        self.assertFalse(saved["finalized"])
        self.assertEqual(len(saved["findings"]), 1)
        self.assertIsNone(saved["ended_at"])


if __name__ == "__main__":
    unittest.main()
