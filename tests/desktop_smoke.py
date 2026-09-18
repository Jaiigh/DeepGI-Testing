"""Optional real-window smoke test; fake voice only, no microphone or models.

Run from the repository root: python tests/desktop_smoke.py
The window exercises both a pending-finding case and a zero-finding case,
then closes automatically. Requires a desktop session and Python with Tk.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tkinter as tk

from desktop_app import ProcedureApp
from modules.procedure.controller import ProcedureController
from modules.procedure.voice_worker import VoiceWorker
from modules.procedure.workflow import ProcedureWorkflow
from test_desktop_workflow import FakeBackend


def main():
    # This checks UI/workflow behavior, not the existing PDF formatter.
    with tempfile.TemporaryDirectory() as directory, patch("modules.procedure.controller.generate_pdf_from_json"):
        backend = FakeBackend()
        root = tk.Tk()
        workflow = ProcedureWorkflow(ProcedureController(directory), VoiceWorker(backend))
        app = ProcedureApp(root, workflow=workflow)
        step = 0
        frozen = None
        deadline = time.monotonic() + 15
        failures = []

        def fail(exc):
            failures.append(exc)
            backend.release.set()
            workflow.worker.shutdown()
            root.destroy()

        root.report_callback_exception = lambda kind, value, traceback: fail(value)

        def exercise():
            nonlocal step, frozen
            try:
                assert time.monotonic() < deadline, "Desktop smoke test timed out"
                if step == 0 and workflow.models_ready:
                    assert app.start_button.instate(["disabled"])
                    app.case_id.set("SMOKE-1")
                    app.patient_id.set("PATIENT-1")
                    assert app.start_button.instate(["!disabled"])
                    app.start_button.invoke()
                    assert workflow.controller.phase == "Insertion"
                    assert app.case_entry.instate(["disabled"])
                    app.caecum_button.invoke()
                    assert workflow.controller.phase == "Withdrawal"
                    backend.triggers.put(True)
                    step = 1
                elif step == 1 and workflow.controller.pending:
                    app.end_button.invoke()
                    frozen = workflow.controller.duration
                    assert workflow.controller.phase == "Completed"
                    assert app.export_button.instate(["disabled"])
                    assert "finishing current finding" in app.phase_text.get()
                    backend.release.set()
                    step = 2
                elif step == 2 and workflow.controller.finalized and not workflow.listening:
                    assert workflow.controller.duration == frozen
                    assert len(app.findings.get_children()) == 1
                    assert app.export_button.instate(["!disabled"])
                    assert '"finding_count": 1' in app.summary_text.get("1.0", "end")
                    export = Path(directory) / "export.json"
                    with patch("desktop_app.filedialog.asksaveasfilename", return_value=str(export)):
                        app.export_button.invoke()
                    assert json.loads(export.read_text())["summary"]["finding_count"] == 1
                    app.new_button.invoke()
                    assert workflow.controller.phase == "Ready"
                    assert not app.findings.get_children()
                    app.case_id.set("SMOKE-2")
                    app.patient_id.set("PATIENT-2")
                    app.start_button.invoke()
                    app.caecum_button.invoke()
                    app.end_button.invoke()
                    step = 3
                elif step == 3 and not workflow.listening:
                    assert workflow.controller.finalized
                    assert workflow.controller.case["summary"]["finding_count"] == 0
                    app.close()
                    return
            except Exception as exc:
                fail(exc)
                return
            root.after(50, exercise)

        root.after(50, exercise)
        root.mainloop()
        if failures:
            raise failures[0]
        assert backend.closed
        print("Desktop smoke passed: phase buttons, pending finding, timer, results, export, new case, shutdown.")


if __name__ == "__main__":
    main()
