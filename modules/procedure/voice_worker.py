"""One audio/model worker. Request/reply messages serialize decisions on the UI."""

import queue
import threading
from dataclasses import dataclass, field


@dataclass
class Message:
    kind: str
    data: dict = field(default_factory=dict)
    reply: object = None


class NativeVoiceBackend:
    def prepare(self):
        # Delay heavy imports so Tk and the phase controller start immediately.
        import config
        from pathlib import Path
        if not (Path(config.LLM_LORA_PATH) / "adapter_config.json").is_file():
            raise FileNotFoundError("Set LLM_LORA_PATH in config.py to your LoRA adapter folder.")
        import pipeline
        from modules.asr.transcriber import warmup as warmup_asr
        from modules.voice_activation.detector import warmup as warmup_trigger
        self.pipeline = pipeline
        pipeline.warmup()
        pipeline.warmup_llm()
        warmup_asr()
        warmup_trigger()

    def wait(self, cancel):
        return self.pipeline.wait_for_trigger(cancel)

    def process(self, status, result):
        return self.pipeline.process_finding(on_status=status, on_result=result, strict=True)

    def close(self):
        # Release playback/recording even when an audio operation raised.
        import sys
        if "sounddevice" in sys.modules:
            sys.modules["sounddevice"].stop()


class VoiceWorker(threading.Thread):
    def __init__(self, backend=None):
        super().__init__(name="DeepGI voice", daemon=True)
        self.backend = backend or NativeVoiceBackend()
        self.messages = queue.Queue()
        self.commands = queue.Queue()
        self.cancel_listening = threading.Event()
        self.stopping = threading.Event()

    def send(self, kind, **data):
        self.messages.put(Message(kind, data))

    def _request(self, kind, **data):
        reply = queue.Queue(maxsize=1)
        self.messages.put(Message(kind, data, reply))
        # The UI keeps draining messages during graceful shutdown, including
        # result acknowledgements for an already accepted finding.
        return reply.get()

    def prepare(self):
        self.commands.put("prepare")

    def listen(self):
        self.cancel_listening.clear()
        self.commands.put("listen")

    def stop_listening(self):
        self.cancel_listening.set()

    def shutdown(self):
        self.stopping.set()
        self.stop_listening()
        self.commands.put("shutdown")

    def _listen(self):
        while not self.cancel_listening.is_set() and not self.stopping.is_set():
            self.send("status", text="Listening")
            try:
                found = self.backend.wait(self.cancel_listening)
            except Exception as exc:
                if not self.cancel_listening.is_set() and not self.stopping.is_set():
                    self.send("error", stage="Listening", message=str(exc), transcription="")
                return
            if not found or self.cancel_listening.is_set() or self.stopping.is_set():
                return
            finding_id = self._request("trigger")
            if finding_id is None:
                return
            failed = False
            try:
                outcome = self.backend.process(
                    lambda text: self.send("status", text=text),
                    lambda text, result: self._request(
                        "result", finding_id=finding_id, transcription=text, result=result),
                )
                if outcome is None:
                    self.send("notice", text="No finding heard. Say Hey DeepGI to try again.")
            except Exception as exc:
                failed = True
                self.send("error", stage=getattr(exc, "stage", "Processing"),
                          message=str(exc), transcription=getattr(exc, "transcription", ""),
                          finding_id=finding_id)
            finally:
                self.send("done", finding_id=finding_id)
            # Explicit retry avoids loops of audio/model failures.
            if failed:
                return

    def run(self):
        try:
            while not self.stopping.is_set():
                command = self.commands.get()
                if self.stopping.is_set() or command == "shutdown":
                    break
                if command == "prepare":
                    self.send("status", text="Loading")
                    try:
                        self.backend.prepare()
                        self.send("ready")
                    except Exception as exc:
                        self.send("init_error", message=str(exc))
                elif command == "listen":
                    self._listen()
                    self.send("idle")
        finally:
            try:
                self.backend.close()
            except Exception as exc:
                self.send("notice", text=f"Audio cleanup: {exc}")
            self.send("stopped")
