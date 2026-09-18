"""Launch with python desktop_app.py on the microphone/model computer."""

import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from modules.caecum import CaecumDetected
from modules.procedure.workflow import ProcedureWorkflow


class ProcedureApp:
    def __init__(self, root, workflow=None):
        self.root = root
        self.workflow = workflow if workflow is not None else ProcedureWorkflow()
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
        self.case_id.trace_add("write", lambda *_: self._refresh_controls(self._state()))
        self.patient_id.trace_add("write", lambda *_: self._refresh_controls(self._state()))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.workflow.start()
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

    def _state(self):
        return self.workflow.state(self.case_id.get(), self.patient_id.get())

    def start_case(self):
        self.workflow.start_case(self.case_id.get(), self.patient_id.get())
        self._refresh()

    def start_withdrawal(self):
        state = self._state()
        self.workflow.dispatch(CaecumDetected(state.case_internal_id, source="button"))
        self._refresh()

    def end_case(self):
        self.workflow.end_case()
        self._refresh()

    def retry_voice(self):
        self.workflow.retry_voice()
        self._refresh()

    def retry_save(self):
        self.workflow.retry_save()
        self._refresh()

    def export(self):
        state = self._state()
        if "export" not in state.actions:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Export case JSON", defaultextension=".json",
            initialfile=state.report_filename, filetypes=[("JSON", "*.json")])
        if path:
            self.workflow.export_json(path)
            self._refresh()

    def new_case(self):
        if self.workflow.new_case():
            self.case_id.set("")
            self.patient_id.set("")
            self.case_entry.focus_set()
        self._refresh()

    def _refresh_controls(self, state):
        for entry in (self.case_entry, self.patient_entry):
            entry.configure(state="normal" if state.identity_editable else "disabled")
        buttons = {
            "start": self.start_button, "caecum": self.caecum_button,
            "end": self.end_button, "retry_voice": self.retry_voice_button,
            "retry_save": self.retry_save_button, "export": self.export_button,
            "new_case": self.new_button,
        }
        for action, widget in buttons.items():
            widget.configure(state="normal" if action in state.actions else "disabled")
        self.retry_voice_button.configure(text=state.retry_voice_label)

    def _refresh(self, state=None):
        state = state if state is not None else self._state()
        self.phase_text.set(state.phase_label)
        seconds = int(state.duration_seconds)
        self.timer_text.set(f"{seconds // 60:02d}:{seconds % 60:02d}")
        self.voice_text.set(state.voice_status)
        self.notice_text.set(state.notice)
        self.save_text.set(state.save_status)
        self._refresh_controls(state)
        case = state.case
        # Presentation cache: preserve selection and scroll position between ticks.
        signature = (case["internal_id"], len(case["events"]), len(case["findings"]),
                     case["finalized"], bool(case["pending_finding"])) if case else None
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
        summary = json.dumps(case, ensure_ascii=False, indent=2) if case and case["finalized"] else "Final summary will appear after completion and any pending finding finishes."
        self._set_text(self.summary_text, summary)

    def _select_finding(self, _event=None):
        selected = self.findings.selection()
        case = self._state().case
        if not selected or not case:
            return
        finding = next((item for item in case["findings"] if item["finding_id"] == selected[0]), None)
        if finding is not None:
            self._set_text(self.finding_detail, json.dumps(finding, ensure_ascii=False, indent=2))

    def close(self):
        state = self._state()
        confirmed = False
        if state.needs_close_confirmation:
            confirmed = messagebox.askyesno(
                "Close active case?",
                "Save this case as incomplete and close? Any accepted finding will finish first.",
                parent=self.root)
            if not confirmed:
                return
        self.workflow.close(confirmed=confirmed)
        self._refresh()

    def _tick(self):
        self.workflow.process_pending()
        state = self._state()
        self._refresh(state)
        if state.ready_to_close:
            self.root.destroy()
            return
        self.root.after(100, self._tick)


def main():
    root = tk.Tk()
    ProcedureApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
