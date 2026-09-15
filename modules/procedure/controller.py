"""Single-case state machine. Called on the UI thread; independent of Tk/audio."""

import copy
import json
import os
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as stream:
            temporary = stream.name
            json.dump(data, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


class ProcedureController:
    def __init__(self, report_dir="outputs/reports", monotonic=time.monotonic,
                 now=None, writer=atomic_write_json):
        self.report_dir = Path(report_dir)
        self._clock = monotonic
        self._now = now or (lambda: datetime.now().astimezone())
        self._writer = writer
        self.case = None
        self.pending = None
        self._withdrawal_start = None
        self._withdrawal_stop = None
        self.save_error = None
        self.dirty = False

    @property
    def phase(self):
        return self.case["phase"] if self.case else "Ready"

    @property
    def path(self):
        return self.report_dir / ("case_" + self.case["internal_id"] + ".json")

    @property
    def duration(self):
        if self._withdrawal_start is None:
            return 0.0
        stop = self._withdrawal_stop
        return max(0.0, (self._clock() if stop is None else stop) - self._withdrawal_start)

    @property
    def finalized(self):
        return bool(self.case and self.case["finalized"])

    def _timestamp(self):
        return self._now().isoformat()

    def _require(self, phase):
        if self.phase != phase or (self.case and self.case["incomplete"]):
            raise ValueError("This action is not available in the current phase.")

    def _event(self, kind, timestamp=None, **values):
        self.case["events"].append({"type": kind, "timestamp": timestamp or self._timestamp(), **values})

    def start(self, case_id, patient_id):
        self._require("Ready")
        if not case_id.strip() or not patient_id.strip():
            raise ValueError("Enter both case ID and patient ID.")
        timestamp = self._timestamp()
        self.case = {
            "schema_version": 1, "internal_id": uuid.uuid4().hex,
            "case_id": case_id.strip(), "patient_id": patient_id.strip(),
            "procedure_type": "COLONOSCOPY", "phase": "Insertion",
            "started_at": timestamp, "caecum_at": None,
            "withdrawal_started_at": None, "ended_at": None,
            "interrupted_at": None, "incomplete": False, "finalized": False,
            "withdrawal_duration_seconds": 0.0, "pending_finding": None,
            "events": [], "findings": [], "processing_errors": [], "summary": None,
        }
        self._event("CASE_START", timestamp)
        self.save()

    def start_withdrawal(self):
        self._require("Insertion")
        self._withdrawal_start = self._clock()
        timestamp = self._timestamp()
        self.case.update(phase="Withdrawal", caecum_at=timestamp, withdrawal_started_at=timestamp)
        self._event("LANDMARK_DETECTED", timestamp, location="CECUM")
        self._event("WITHDRAWAL_START", timestamp)
        self.save()

    def accept_trigger(self):
        if self.phase != "Withdrawal" or self.pending or self.case["incomplete"]:
            return None
        self.pending = {"finding_id": uuid.uuid4().hex, "capture_started_at": self._timestamp()}
        self.save()
        return self.pending["finding_id"]

    def add_finding(self, finding_id, transcription, result):
        self._check_pending(finding_id)
        if any(item["finding_id"] == finding_id for item in self.case["findings"]):
            raise ValueError("Finding already recorded.")
        finding = {
            **self.pending, "completed_at": self._timestamp(),
            "transcription": transcription, "result": copy.deepcopy(result),
        }
        self.case["findings"].append(finding)
        self._event("FINDING_DETECTED", finding["completed_at"], finding_id=finding_id)
        return self.save()

    def _check_pending(self, finding_id):
        if not self.pending or self.pending["finding_id"] != finding_id:
            raise ValueError("Stale or unknown finding.")

    def record_error(self, stage, message, transcription="", finding_id=None):
        if not self.case:
            return
        error = {"timestamp": self._timestamp(), "stage": stage, "message": str(message),
                 "transcription": transcription, "finding_id": finding_id}
        self.case["processing_errors"].append(error)
        self._event("PROCESSING_ERROR", error["timestamp"], stage=stage, finding_id=finding_id)
        self._finalize()
        self.save()

    def finish_finding(self, finding_id):
        self._check_pending(finding_id)
        self.pending = None
        self._finalize()
        self.save()

    def end(self):
        self._require("Withdrawal")
        self._withdrawal_stop = self._clock()
        timestamp = self._timestamp()
        self.case.update(phase="Completed", ended_at=timestamp)
        self._event("LANDMARK_DETECTED", timestamp, location="ANUS")
        self._event("CASE_END", timestamp)
        self._finalize()
        self.save()

    def _finalize(self):
        if self.phase == "Completed" and not self.pending and not self.case["incomplete"]:
            self.case["finalized"] = True
            self.case["summary"] = {
                "finding_count": len(self.case["findings"]),
                "withdrawal_duration_seconds": self.duration,
                "processing_error_count": len(self.case["processing_errors"]),
            }

    def interrupt(self):
        if self.case and not self.finalized and not self.case["incomplete"]:
            if self._withdrawal_start is not None and self._withdrawal_stop is None:
                self._withdrawal_stop = self._clock()
            timestamp = self._timestamp()
            self.case.update(incomplete=True, interrupted_at=timestamp)
            self._event("CASE_INTERRUPTED", timestamp)
            self.save()

    def snapshot(self):
        if not self.case:
            return None
        data = copy.deepcopy(self.case)
        data["withdrawal_duration_seconds"] = self.duration
        data["pending_finding"] = copy.deepcopy(self.pending)
        return data

    def save(self):
        if not self.case:
            return True
        self.dirty = True
        try:
            self._writer(self.path, self.snapshot())
        except Exception as exc:
            self.save_error = str(exc)
            return False
        self.dirty = False
        self.save_error = None
        return True

    def export(self, path):
        if not self.finalized or self.dirty:
            raise ValueError("Finish the case and save its results before exporting.")
        self._writer(path, self.snapshot())

    def new_case(self):
        if not self.finalized or self.dirty:
            raise ValueError("Finish the case and save its results before starting another.")
        self.case = None
        self.pending = None
        self._withdrawal_start = self._withdrawal_stop = None
        self.save_error = None
