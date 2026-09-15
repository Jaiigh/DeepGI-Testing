"""Launch with python desktop_app.py on the microphone/model computer."""

import json
import queue
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from modules.procedure.controller import ProcedureController
from modules.procedure.voice_worker import VoiceWorker


class ProcedureApp:
    def __init__(self, root, controller=None, worker=None):
        self.root = root
        self.controller = controller or ProcedureController()
        self.worker = worker or VoiceWorker()
        self.models_ready = False
        self.loading = True
        self.listening = False
        self.voice_error = False
        self.closing = False
        self.worker_stopped = False
        self._display_signature = object()
        self.case_id = tk.StringVar()
        self.patient_id = tk.StringVar()
        self.phase_text = tk.StringVar(value="Ready")
        self.timer_text = tk.StringVar(value="00:00")
        self.voice_text = tk.StringVar(value="Loading")
        self.notice_text = tk.StringVar(value="Preparing voice models…")
        self.save_text = tk.StringVar()
        self.count_text = tk.StringVar(value="0 findings")
        self._build()
        self.case_id.trace_add("write", lambda *_: self._refresh_controls())
        self.patient_id.trace_add("write", lambda *_: self._refresh_controls())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker.start()
        self.worker.prepare()
        self._tick()

    def _build(self):
        self.root.title("DeepGI · Procedure workflow")
        self.root.geometry("1080x800")
        self.root.minsize(850, 650)
        style = ttk.Style(self.root)
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Timer.TLabel", font=("Consolas", 28, "bold"))
        style.configure("TButton", padding=(10, 8))
        body = ttk.Frame(self.root, padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="DeepGI | Procedure workflow", style="Title.TLabel").pack(anchor="w")
        ttk.Label(body, text="Ready  →  Insertion  →  Withdrawal  →  Completed").pack(anchor="w", pady=(4, 16))

        identity = ttk.Frame(body)
        identity.pack(fill="x")
        ttk.Label(identity, text="Case ID").grid(row=0, column=0, sticky="w")
        self.case_entry = ttk.Entry(identity, textvariable=self.case_id, width=30)
        self.case_entry.grid(row=1, column=0, padx=(0, 20), sticky="ew")
        ttk.Label(identity, text="Patient ID").grid(row=0, column=1, sticky="w")
        self.patient_entry = ttk.Entry(identity, textvariable=self.patient_id, width=30)
        self.patient_entry.grid(row=1, column=1, sticky="ew")

        current = ttk.LabelFrame(body, text="Current procedure", padding=12)
        current.pack(fill="x", pady=16)
        ttk.Label(current, textvariable=self.phase_text, style="Title.TLabel").pack(side="left")
        clock = ttk.Frame(current)
        clock.pack(side="right")
        ttk.Label(clock, text="Withdrawal time").pack()
        ttk.Label(clock, textvariable=self.timer_text, style="Timer.TLabel").pack()

        controls = ttk.Frame(body)
        controls.pack(fill="x")
        self.start_button = ttk.Button(controls, text="Start Procedure", command=self.start_case)
        self.caecum_button = ttk.Button(controls, text="Found Caecum / Start Withdrawal", command=self.start_withdrawal)
        self.end_button = ttk.Button(controls, text="Reached Anus / End Procedure", command=self.end_case)
        for widget in (self.start_button, self.caecum_button, self.end_button):
            widget.pack(side="left", padx=(0, 8))

        audio = ttk.Frame(body)
        audio.pack(fill="x", pady=(14, 4))
        ttk.Label(audio, text="Voice: ").pack(side="left")
        ttk.Label(audio, textvariable=self.voice_text).pack(side="left")
        self.retry_voice_button = ttk.Button(audio, text="Retry", command=self.retry_voice)
        self.retry_voice_button.pack(side="right")
        ttk.Label(body, text='During withdrawal: say “Hey DeepGI”, wait for the prompt, then dictate your finding.').pack(anchor="w")
        ttk.Label(body, textvariable=self.notice_text, wraplength=960).pack(anchor="w", pady=(4, 10))

        tabs = ttk.Notebook(body)
        tabs.pack(fill="both", expand=True)
        findings_tab = ttk.Frame(tabs, padding=8)
        tabs.add(findings_tab, text="Findings")
        ttk.Label(findings_tab, textvariable=self.count_text).pack(anchor="w")
        columns = ("time", "lesion", "location", "size", "procedure")
        self.findings = ttk.Treeview(findings_tab, columns=columns, show="headings", height=5)
        for column, title in zip(columns, ("Captured", "Lesion", "Location", "Size (mm)", "Procedure")):
            self.findings.heading(column, text=title)
            self.findings.column(column, width=140, stretch=True)
        self.findings.pack(fill="x")
        self.findings.bind("<<TreeviewSelect>>", self._select_finding)
        self.finding_detail = self._text_area(findings_tab, height=5)
        event_tab = ttk.Frame(tabs)
        tabs.add(event_tab, text="Event log")
        self.event_text = self._text_area(event_tab)
        summary_tab = ttk.Frame(tabs)
        tabs.add(summary_tab, text="Summary / JSON")
        self.summary_text = self._text_area(summary_tab)

        footer = ttk.Frame(body)
        footer.pack(fill="x", pady=(12, 0))
        self.export_button = ttk.Button(footer, text="Export JSON", command=self.export)
        self.export_button.pack(side="right")
        self.new_button = ttk.Button(footer, text="New Case", command=self.new_case)
        self.new_button.pack(side="right", padx=8)
        self.retry_save_button = ttk.Button(footer, text="Retry Save", command=self.retry_save)
        self.retry_save_button.pack(side="left")
        ttk.Label(body, textvariable=self.save_text, wraplength=960).pack(anchor="w", pady=(6, 0))

    @staticmethod
    def _text_area(parent, height=12):
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", height=height, state="disabled")
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(fill="both", expand=True)
        return text

    @staticmethod
    def _set_text(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", value)
        widget.configure(state="disabled")

    def start_case(self):
        if not self.models_ready or self.closing:
            return
        self._action(lambda: self.controller.start(self.case_id.get(), self.patient_id.get()))

    def start_withdrawal(self):
        if self._action(self.controller.start_withdrawal):
            self._listen()

    def _listen(self):
        self.listening = True
        self.voice_error = False
        self.notice_text.set("")
        self.worker.listen()

    def end_case(self):
        # Timer and phase change on this thread before a queued trigger can be accepted.
        if self._action(self.controller.end):
            self.worker.stop_listening()

    def retry_voice(self):
        if self.closing or self.loading or self.listening:
            return
        self.notice_text.set("")
        self.voice_error = False
        if not self.models_ready:
            self.loading = True
            self.worker.prepare()
        elif self.controller.phase == "Withdrawal":
            self._listen()
        self._refresh_controls()

    def retry_save(self):
        self.controller.save()
        self._refresh()

    def _action(self, action):
        try:
            action()
        except ValueError as exc:
            self.notice_text.set(str(exc))
            return False
        self._refresh()
        return True

    def export(self):
        if not self.controller.finalized or self.controller.dirty:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Export case JSON", defaultextension=".json",
            initialfile=self.controller.path.name, filetypes=[("JSON", "*.json")])
        if path:
            try:
                self.controller.export(path)
                self.notice_text.set(f"Exported to {path}")
            except Exception as exc:
                self.notice_text.set(f"Export failed: {exc}")

    def new_case(self):
        if self.listening:
            return
        if self._action(self.controller.new_case):
            self.case_id.set("")
            self.patient_id.set("")
            self.voice_error = False
            self.voice_text.set("Ready")
            self.notice_text.set("")
            self.case_entry.focus_set()
            self._refresh()

    def _handle(self, message):
        kind, data = message.kind, message.data
        if kind == "trigger":
            finding_id = None if self.closing else self.controller.accept_trigger()
            message.reply.put(finding_id)
        elif kind == "result":
            saved = False
            try:
                saved = self.controller.add_finding(**data)
            except ValueError as exc:
                self.notice_text.set(str(exc))
            finally:
                message.reply.put(saved)
        elif kind == "done":
            self._action(lambda: self.controller.finish_finding(data["finding_id"]))
        elif kind == "status":
            self.voice_text.set(data["text"])
        elif kind == "ready":
            self.loading = False
            self.models_ready = True
            self.voice_text.set("Ready")
            self.notice_text.set("Models ready. Enter case and patient IDs to start.")
        elif kind == "init_error":
            self.loading = False
            self.voice_error = True
            self.voice_text.set("Error")
            self.notice_text.set(f"Initialization failed: {data['message']} Check config.py and dependencies, then Retry.")
        elif kind == "error":
            self.voice_error = True
            self.voice_text.set("Error")
            self.notice_text.set(f"{data['stage']}: {data['message']} Use Retry Listening to try again.")
            self.controller.record_error(**data)
        elif kind == "notice":
            self.notice_text.set(data["text"])
        elif kind == "idle":
            self.listening = False
            if not self.voice_error:
                self.voice_text.set("Stopped" if self.controller.phase == "Completed" else "Ready")
        elif kind == "stopped":
            self.worker_stopped = True
            self.listening = False

    def _refresh_controls(self):
        phase = self.controller.phase
        active = not self.closing and not (self.controller.case and self.controller.case["incomplete"])
        ready = active and phase == "Ready"
        for entry in (self.case_entry, self.patient_entry):
            entry.configure(state="normal" if ready else "disabled")
        enabled = {
            self.start_button: ready and self.models_ready and bool(self.case_id.get().strip() and self.patient_id.get().strip()),
            self.caecum_button: active and phase == "Insertion",
            self.end_button: active and phase == "Withdrawal",
            self.retry_voice_button: active and not self.loading and not self.listening and
                                     (not self.models_ready or (phase == "Withdrawal" and self.voice_error)),
            self.retry_save_button: self.controller.dirty,
            self.export_button: active and self.controller.finalized and not self.controller.dirty,
            self.new_button: active and self.controller.finalized and not self.controller.dirty and not self.listening,
        }
        for widget, state in enabled.items():
            widget.configure(state="normal" if state else "disabled")
        self.retry_voice_button.configure(text="Retry Loading" if not self.models_ready else "Retry Listening")

    def _refresh(self):
        controller = self.controller
        phase = controller.phase
        if phase == "Completed" and controller.pending:
            phase += " — finishing current finding"
        if self.closing:
            phase = "Closing — waiting for voice worker"
        self.phase_text.set(phase)
        seconds = int(controller.duration)
        self.timer_text.set(f"{seconds // 60:02d}:{seconds % 60:02d}")
        if controller.save_error:
            self.save_text.set(f"Not saved: {controller.save_error}. Results retained in memory; use Retry Save.")
        elif controller.case:
            self.save_text.set(f"Saved: {controller.path}")
        else:
            self.save_text.set("")
        self._refresh_controls()
        case = controller.case
        # Don't overwrite selection or scroll position on every timer tick.
        signature = (case["internal_id"], len(case["events"]), len(case["findings"]),
                     case["finalized"], bool(controller.pending)) if case else None
        if signature == self._display_signature:
            return
        self._display_signature = signature
        selected = self.findings.selection()
        self.findings.delete(*self.findings.get_children())
        self.count_text.set(f"{len(case['findings']) if case else 0} findings")
        for finding in case["findings"] if case else []:
            result = finding["result"]
            self.findings.insert("", "end", iid=finding["finding_id"], values=(
                finding["capture_started_at"][11:19], result.get("lesion_type") or "—",
                result.get("location") or "—", result.get("size_mm") if result.get("size_mm") is not None else "—",
                result.get("procedure") or "—"))
        if selected and self.findings.exists(selected[0]):
            self.findings.selection_set(selected[0])
        else:
            self._set_text(self.finding_detail, "Select a finding to see its transcription and all seven fields.")
        events = "\n".join(json.dumps(event, ensure_ascii=False) for event in case["events"]) if case else ""
        self._set_text(self.event_text, events)
        summary = json.dumps(controller.snapshot(), ensure_ascii=False, indent=2) if controller.finalized else "Final summary will appear after completion and any pending finding finishes."
        self._set_text(self.summary_text, summary)

    def _select_finding(self, _event=None):
        selected = self.findings.selection()
        if not selected or not self.controller.case:
            return
        finding = next(item for item in self.controller.case["findings"] if item["finding_id"] == selected[0])
        self._set_text(self.finding_detail, json.dumps(finding, ensure_ascii=False, indent=2))

    def close(self):
        if self.closing:
            return
        if self.controller.case and not self.controller.finalized:
            if not messagebox.askyesno("Close active case?", "Save this case as incomplete and close? Any accepted finding will finish first.", parent=self.root):
                return
            self.controller.interrupt()
            self.worker.stop_listening()
        if self.controller.dirty and not self.controller.save():
            self.notice_text.set("Cannot close while results are unsaved. Use Retry Save, then close again.")
            self._refresh()
            return
        self.closing = True
        self.worker.shutdown()
        self._refresh()

    def _tick(self):
        try:
            while True:
                self._handle(self.worker.messages.get_nowait())
        except queue.Empty:
            pass
        self._refresh()
        if self.closing and self.worker_stopped:
            if self.controller.dirty:
                self.notice_text.set("Voice worker stopped. Retry Save to finish closing without losing results.")
            else:
                self.root.destroy()
                return
        self.root.after(100, self._tick)


def main():
    root = tk.Tk()
    ProcedureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
