"""Application orchestration without Tkinter or audio/model imports.

Create and operate this object on the application thread. External producers
may call submit_event() from any thread; process_pending() applies their input
on the application thread. dispatch() applies a signal immediately there.
"""

import queue
import threading
from dataclasses import dataclass
from typing import Optional

from modules.caecum import CaecumDetected, CaecumModule
from modules.procedure.controller import ProcedureController
from modules.procedure.voice_worker import VoiceWorker


@dataclass(frozen=True)
class WorkflowState:
    phase: str
    phase_label: str
    duration_seconds: float
    case: Optional[dict]
    case_internal_id: Optional[str]
    voice_status: str
    notice: str
    save_status: str
    actions: frozenset
    identity_editable: bool
    retry_voice_label: str
    needs_close_confirmation: bool
    ready_to_close: bool
    report_filename: str


class ProcedureWorkflow:
    def __init__(self, controller=None, worker=None):
        self.controller = controller if controller is not None else ProcedureController()
        self.worker = worker if worker is not None else VoiceWorker()
        self.caecum = CaecumModule(self.controller)
        self._application_thread = threading.get_ident()
        self._events = queue.Queue()
        self._started = False
        self.models_ready = False
        self.loading = True
        self.listening = False
        self.voice_error = False
        self.closing = False
        self.worker_stopped = False
        self.voice_status = "Loading"
        self.notice = "Preparing voice models…"

    def _check_thread(self):
        if threading.get_ident() != self._application_thread:
            raise RuntimeError("Use submit_event() from background threads.")

    def start(self):
        self._check_thread()
        if not self._started:
            self._started = True
            self.worker.start()
            self.worker.prepare()

    def _active(self):
        return not (self.closing or self.worker_stopped or
                    (self.controller.case and self.controller.case["incomplete"]))

    def _action(self, action):
        self._check_thread()
        try:
            action()
        except ValueError as exc:
            self.notice = str(exc)
            return False
        return True

    def start_case(self, case_id, patient_id):
        self._check_thread()
        if not self._active() or not self.models_ready:
            return False
        return self._action(lambda: self.controller.start(case_id, patient_id))

    def dispatch(self, event):
        """Apply the same event contract used by submit_event(), synchronously."""
        self._check_thread()
        if not isinstance(event, CaecumDetected):
            self.notice = "Unsupported procedure event."
            return False
        if not self._active() or not self.models_ready:
            self.notice = "Caecum signal rejected: the procedure is not available."
            return False
        if not self._action(lambda: self.caecum.handle(event)):
            return False
        self._listen()
        return True

    def submit_event(self, event):
        """Thread-safe enqueue only; no controller, worker, or UI mutation here."""
        self._events.put(event)

    def _listen(self):
        self.listening = True
        self.voice_error = False
        self.notice = ""
        self.worker.listen()

    def end_case(self):
        self._check_thread()
        if not self._active():
            return False
        # Freeze time before returning to the UI/event loop, including while
        # an accepted finding is still recording or extracting.
        if not self._action(self.controller.end):
            return False
        self.worker.stop_listening()
        return True

    def retry_voice(self):
        self._check_thread()
        if not self._active() or self.loading or self.listening:
            return False
        if self.models_ready and self.controller.phase != "Withdrawal":
            return False
        self.notice = ""
        self.voice_error = False
        if not self.models_ready:
            self.loading = True
            self.worker.prepare()
        else:
            self._listen()
        return True

    def retry_save(self):
        self._check_thread()
        return self.controller.save()

    def export_json(self, path):
        self._check_thread()
        if not self._active():
            return False
        try:
            self.controller.export(path)
        except Exception as exc:
            self.notice = f"Export failed: {exc}"
            return False
        self.notice = f"Exported to {path}"
        return True

    def new_case(self):
        self._check_thread()
        if not self._active() or self.listening:
            return False
        if not self._action(self.controller.new_case):
            return False
        self.voice_error = False
        self.voice_status = "Ready"
        self.notice = ""
        return True

    def close(self, confirmed=False):
        """Request graceful shutdown; confirmation dialogs belong to the UI."""
        self._check_thread()
        if self.closing:
            return True
        if self.state().needs_close_confirmation:
            if not confirmed:
                return False
            self.controller.interrupt()
            self.worker.stop_listening()
        if self.controller.dirty and not self.controller.save():
            self.notice = "Cannot close while results are unsaved. Use Retry Save, then close again."
            return False
        self.closing = True
        self.worker.shutdown()
        return True

    def process_pending(self):
        """Drain bounded batches so a busy signal producer cannot starve the UI."""
        self._check_thread()
        for messages, handle in ((self.worker.messages, self._handle_worker_message),
                                 (self._events, self.dispatch)):
            for _ in range(100):
                try:
                    message = messages.get_nowait()
                except queue.Empty:
                    break
                handle(message)
        if self.closing and self.worker_stopped and self.controller.dirty:
            self.notice = "Voice worker stopped. Retry Save to finish closing without losing results."

    def _handle_worker_message(self, message):
        kind, data = message.kind, message.data
        if kind == "trigger":
            finding_id = self.controller.accept_trigger() if self._active() else None
            message.reply.put(finding_id)
        elif kind == "result":
            saved = False
            try:
                saved = self.controller.add_finding(**data)
            except ValueError as exc:
                self.notice = str(exc)
            finally:
                # An accepted finding can still deliver its result during close.
                message.reply.put(saved)
        elif kind == "done":
            self._action(lambda: self.controller.finish_finding(data["finding_id"]))
        elif kind == "status":
            self.voice_status = data["text"]
        elif kind == "ready":
            self.loading = False
            self.models_ready = True
            self.voice_status = "Ready"
            self.notice = "Models ready. Enter case and patient IDs to start."
        elif kind == "init_error":
            self.loading = False
            self.voice_error = True
            self.voice_status = "Error"
            self.notice = f"Initialization failed: {data['message']} Check config.py and dependencies, then Retry."
        elif kind == "error":
            self.voice_error = True
            self.voice_status = "Error"
            self.notice = f"{data['stage']}: {data['message']} Use Retry Listening to try again."
            self.controller.record_error(**data)
        elif kind == "notice":
            self.notice = data["text"]
        elif kind == "idle":
            self.listening = False
            if not self.voice_error:
                self.voice_status = "Stopped" if self.controller.phase == "Completed" else "Ready"
        elif kind == "stopped":
            self.worker_stopped = True
            self.listening = False

    def state(self, case_id="", patient_id=""):
        """Detached view data and action availability; no widgets or dialogs."""
        self._check_thread()
        controller = self.controller
        active = self._active()
        ready = active and controller.phase == "Ready"
        available = {
            "start": ready and self.models_ready and bool(case_id.strip() and patient_id.strip()),
            "caecum": active and controller.phase == "Insertion" and self.models_ready,
            "end": active and controller.phase == "Withdrawal",
            "retry_voice": active and not self.loading and not self.listening and
                           (not self.models_ready or (controller.phase == "Withdrawal" and self.voice_error)),
            "retry_save": controller.dirty,
            "export": active and controller.finalized and not controller.dirty,
            "new_case": active and controller.finalized and not controller.dirty and not self.listening,
        }
        phase_label = controller.phase
        if controller.phase == "Completed" and controller.pending:
            phase_label += " — finishing current finding"
        if self.closing:
            phase_label = "Closing — waiting for voice worker"
        save_status = ""
        if controller.save_error:
            save_status = f"Not saved: {controller.save_error}. Results retained in memory; use Retry Save."
        elif controller.case:
            save_status = f"Saved: {controller.path}"
        case = controller.snapshot()
        return WorkflowState(
            phase=controller.phase, phase_label=phase_label, duration_seconds=controller.duration,
            case=case, case_internal_id=case["internal_id"] if case else None,
            voice_status=self.voice_status, notice=self.notice, save_status=save_status,
            actions=frozenset(name for name, enabled in available.items() if enabled),
            identity_editable=ready,
            retry_voice_label="Retry Loading" if not self.models_ready else "Retry Listening",
            needs_close_confirmation=bool(case and not controller.finalized and not case["incomplete"]),
            ready_to_close=self.closing and self.worker_stopped and not controller.dirty,
            report_filename=controller.path.name if case else "",
        )
