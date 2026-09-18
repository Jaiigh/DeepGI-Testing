"""Caecum input and withdrawal transition; independent of UI and audio."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CaecumDetected:
    """Bind a detection to the case observed by its producer, not a future case."""

    case_internal_id: str
    source: str = "button"


class CaecumModule:
    def __init__(self, controller):
        self.controller = controller

    def handle(self, event: CaecumDetected):
        """Apply a detection on the application thread; reject without side effects.

        ProcedureController remains the authority on valid phases, timestamps,
        timer state, persistence, and the existing JSON event names.
        """
        case = self.controller.case
        if not case or event.case_internal_id != case["internal_id"]:
            raise ValueError("Caecum signal does not belong to the current case.")
        self.controller.start_withdrawal()
