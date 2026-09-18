"""Headless workflow tests: real controller/worker, fake audio, mocked PDF renderer."""

import json
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from modules.caecum import CaecumDetected
from modules.procedure.workflow import ProcedureWorkflow
from modules.procedure.controller import ProcedureController
from modules.procedure.voice_worker import VoiceWorker


RESULT = {"lesion_type": "polyp", "location": "rectum", "size_mm": 5,
          "procedure": None, "biopsy_forceps": False, "biopsy_pieces": None, "pathology": False}


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
        self.addCleanup(self.directory.cleanup)
        renderer = patch("modules.procedure.controller.generate_pdf_from_json")
        self.pdf = renderer.start()
        self.addCleanup(renderer.stop)
        self.backend = FakeBackend()
        self.worker = VoiceWorker(self.backend)
        self.workflow = ProcedureWorkflow(ProcedureController(self.directory.name), self.worker)
        self.original_writer = self.workflow.controller._writer
        self.addCleanup(self.cleanup)

    def cleanup(self):
        self.backend.release.set()
        self.workflow.controller._writer = self.original_writer
        if self.worker.ident is None:
            self.workflow.start()
        self.workflow.close(confirmed=True)
        self.pump_until(lambda: self.workflow.worker_stopped)
        self.worker.join(timeout=1)
        self.assertFalse(self.worker.is_alive())

    def pump_until(self, condition):
        deadline = time.monotonic() + 4
        while not condition():
            self.assertLess(time.monotonic(), deadline, "worker/workflow handshake timed out")
            self.workflow.process_pending()
            time.sleep(0.001)

    def start(self):
        self.workflow.start()
        self.pump_until(lambda: self.workflow.models_ready)
        self.assertTrue(self.workflow.start_case("CASE-1", "PATIENT-1"))
        self.assertTrue(self.workflow.dispatch(CaecumDetected(self.workflow.state().case_internal_id)))

    def begin_finding(self):
        self.backend.triggers.put(True)
        self.pump_until(lambda: self.workflow.controller.pending is not None)
        self.assertTrue(self.backend.processing.wait(1))

    def test_end_during_finding_freezes_timer_and_waits_for_result(self):
        self.start()
        self.begin_finding()
        self.workflow.end_case()
        frozen = self.workflow.controller.duration
        self.assertFalse(self.workflow.controller.finalized)
        self.backend.release.set()
        self.pump_until(lambda: not self.workflow.listening)
        self.assertTrue(self.workflow.controller.finalized)
        self.assertEqual(self.workflow.controller.duration, frozen)
        self.assertTrue(self.backend.feedback_saved)
        saved = json.loads(self.workflow.controller.path.read_text())
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
            self.workflow._handle_worker_message(message)
        self.workflow.end_case()
        self.worker.messages.put(message)
        self.pump_until(lambda: not self.workflow.listening)
        self.assertFalse(self.backend.processing.is_set())
        self.assertEqual(self.workflow.controller.case["findings"], [])
        self.assertTrue(self.workflow.controller.finalized)

    def test_microphone_failure_preserves_controls_and_allows_retry(self):
        self.start()
        self.backend.triggers.put(RuntimeError("microphone disconnected"))
        self.pump_until(lambda: not self.workflow.listening)
        self.assertEqual(self.workflow.controller.phase, "Withdrawal")
        self.assertTrue(self.workflow.voice_error)
        self.workflow.retry_voice()
        self.begin_finding()
        self.workflow.end_case()
        self.backend.release.set()
        self.pump_until(lambda: not self.workflow.listening)
        self.assertEqual(len(self.workflow.controller.case["findings"]), 1)

    def test_initialization_failure_can_retry(self):
        self.backend.prepare_failure = True
        self.workflow.start()
        self.pump_until(lambda: not self.workflow.loading)
        self.workflow.start_case("CASE-1", "PATIENT-1")
        self.assertEqual(self.workflow.controller.phase, "Ready")
        self.backend.prepare_failure = False
        self.workflow.retry_voice()
        self.pump_until(lambda: self.workflow.models_ready)
        self.assertEqual(self.backend.prepared, 2)

    def test_extraction_error_preserves_transcript_and_finalizes(self):
        self.start()
        self.backend.failure_stage = "Extracting"
        self.begin_finding()
        self.workflow.end_case()
        self.backend.release.set()
        self.pump_until(lambda: not self.workflow.listening)
        self.assertTrue(self.workflow.controller.finalized)
        self.assertEqual(len(self.workflow.controller.case["findings"]), 0)
        error = self.workflow.controller.case["processing_errors"][0]
        self.assertEqual(error["transcription"], "five millimeter polyp")
        self.assertEqual(error["stage"], "Extracting")

    def test_tts_error_does_not_remove_saved_finding(self):
        self.start()
        self.backend.failure_stage = "Speaking"
        self.begin_finding()
        self.workflow.end_case()
        self.backend.release.set()
        self.pump_until(lambda: not self.workflow.listening)
        self.assertEqual(len(self.workflow.controller.case["findings"]), 1)
        self.assertEqual(self.workflow.controller.case["summary"]["processing_error_count"], 1)

    def test_empty_recording_returns_to_listening_without_a_finding(self):
        self.start()
        self.backend.empty = True
        self.begin_finding()
        self.backend.release.set()
        self.pump_until(lambda: self.workflow.controller.pending is None)
        self.assertEqual(self.workflow.controller.case["findings"], [])
        self.assertIn("No finding heard", self.workflow.state().notice)
        self.assertTrue(self.workflow.listening)
        self.workflow.end_case()
        self.pump_until(lambda: not self.workflow.listening)

    def test_shutdown_finishes_accepted_finding_in_incomplete_case(self):
        self.start()
        self.begin_finding()
        self.assertTrue(self.workflow.close(confirmed=True))
        self.backend.release.set()
        self.pump_until(lambda: self.workflow.worker_stopped)
        saved = json.loads(self.workflow.controller.path.read_text())
        self.assertTrue(self.backend.closed)
        self.assertTrue(saved["incomplete"])
        self.assertFalse(saved["finalized"])
        self.assertEqual(len(saved["findings"]), 1)
        self.assertIsNone(saved["ended_at"])

    def test_button_and_background_signal_share_handler_and_owner_thread(self):
        self.workflow.start()
        self.pump_until(lambda: self.workflow.models_ready)
        self.workflow.start_case("CASE-1", "PATIENT-1")
        event = CaecumDetected(self.workflow.state().case_internal_id, "vision")
        owner = threading.get_ident()
        mutation_threads = []
        original = self.workflow.controller.start_withdrawal

        def track_transition():
            mutation_threads.append(threading.get_ident())
            return original()

        with patch.object(self.workflow.controller, "start_withdrawal", side_effect=track_transition), \
                patch.object(self.worker, "listen", wraps=self.worker.listen) as listen:
            producer = threading.Thread(target=self.workflow.submit_event, args=(event,))
            producer.start()
            producer.join(timeout=1)
            self.assertEqual(self.workflow.state().phase, "Insertion")
            self.assertEqual(mutation_threads, [])
            self.workflow.process_pending()
            self.assertEqual(self.workflow.state().phase, "Withdrawal")
            self.assertEqual(mutation_threads, [owner])
            self.assertFalse(self.workflow.dispatch(CaecumDetected(event.case_internal_id, "button")))
            listen.assert_called_once()
        self.assertEqual(len(self.workflow.controller.case["events"]), 3)

    def test_direct_background_dispatch_is_rejected_before_state_changes(self):
        self.workflow.start()
        self.pump_until(lambda: self.workflow.models_ready)
        self.workflow.start_case("CASE-1", "PATIENT-1")
        event = CaecumDetected(self.workflow.state().case_internal_id)
        failures = []

        def invalid_call():
            try:
                self.workflow.dispatch(event)
            except RuntimeError as exc:
                failures.append(str(exc))

        producer = threading.Thread(target=invalid_call)
        producer.start()
        producer.join(timeout=1)
        self.assertEqual(len(failures), 1)
        self.assertIn("submit_event", failures[0])
        self.assertEqual(self.workflow.state().phase, "Insertion")
        self.assertFalse(self.workflow.listening)

    def test_queued_old_signal_cannot_start_withdrawal_in_new_case(self):
        self.start()
        old_id = self.workflow.state().case_internal_id
        self.workflow.end_case()
        self.pump_until(lambda: not self.workflow.listening)
        self.workflow.submit_event(CaecumDetected(old_id, "vision"))
        self.assertTrue(self.workflow.new_case())
        self.workflow.start_case("CASE-1", "PATIENT-1")
        self.workflow.process_pending()
        self.assertEqual(self.workflow.state().phase, "Insertion")
        self.assertEqual(len(self.workflow.controller.case["events"]), 1)
        self.assertFalse(self.workflow.listening)

    def test_unavailable_signal_does_not_start_worker(self):
        with patch.object(self.worker, "listen", wraps=self.worker.listen) as listen:
            self.assertFalse(self.workflow.dispatch(CaecumDetected("stale")))
            self.assertEqual(self.workflow.state().phase, "Ready")
            self.workflow.start()
            self.pump_until(lambda: self.workflow.models_ready)
            self.assertFalse(self.workflow.dispatch(CaecumDetected("stale")))
            listen.assert_not_called()

    def test_save_failure_retains_finding_until_retry_and_blocks_export_reset(self):
        self.start()
        self.begin_finding()

        def fail(*args):
            raise OSError("disk full")

        self.workflow.controller._writer = fail
        self.workflow.end_case()
        self.backend.release.set()
        self.pump_until(lambda: not self.workflow.listening)
        state = self.workflow.state()
        self.assertEqual(len(state.case["findings"]), 1)
        self.assertIn("retry_save", state.actions)
        self.assertNotIn("export", state.actions)
        self.assertNotIn("new_case", state.actions)
        self.assertFalse(self.backend.feedback_saved)
        self.assertFalse(self.workflow.new_case())
        self.assertFalse(self.workflow.export_json(self.directory.name + "/export.json"))
        self.workflow.controller._writer = self.original_writer
        self.assertTrue(self.workflow.retry_save())
        self.assertIn("export", self.workflow.state().actions)
        self.assertTrue(self.workflow.export_json(self.directory.name + "/export.json"))
        self.assertTrue(self.workflow.new_case())
        self.assertEqual(self.workflow.state().phase, "Ready")

    def test_close_confirmation_and_save_retry_do_not_invent_normal_end(self):
        self.start()
        self.assertTrue(self.workflow.state().needs_close_confirmation)
        self.assertFalse(self.workflow.close(confirmed=False))
        self.assertFalse(self.workflow.controller.case["incomplete"])

        def fail(*args):
            raise OSError("disk full")

        self.workflow.controller._writer = fail
        self.assertFalse(self.workflow.close(confirmed=True))
        self.assertTrue(self.workflow.controller.case["incomplete"])
        self.assertFalse(self.workflow.state().ready_to_close)
        self.assertFalse(self.workflow.state().needs_close_confirmation)
        self.assertNotIn("end", self.workflow.state().actions)
        self.workflow.controller._writer = self.original_writer
        self.assertTrue(self.workflow.retry_save())
        self.assertTrue(self.workflow.close())
        self.pump_until(lambda: self.workflow.state().ready_to_close)
        self.assertIsNone(self.workflow.controller.case["ended_at"])

    def test_view_state_is_detached_and_contains_action_rules(self):
        self.assertNotIn("start", self.workflow.state("case", "patient").actions)
        self.workflow.start()
        self.pump_until(lambda: self.workflow.models_ready)
        self.assertNotIn("start", self.workflow.state("case", " ").actions)
        self.assertIn("start", self.workflow.state("case", "patient").actions)
        self.workflow.start_case("case", "patient")
        state = self.workflow.state()
        self.assertFalse(state.identity_editable)
        self.assertIn("caecum", state.actions)
        self.assertNotIn("end", state.actions)
        state.case["events"].clear()
        self.assertEqual(len(self.workflow.controller.case["events"]), 1)


if __name__ == "__main__":
    unittest.main()
