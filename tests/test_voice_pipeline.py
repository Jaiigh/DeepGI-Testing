"""Exercise actual control-flow functions without importing audio/GPU packages."""

import ast
import collections
import queue
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from modules.llm.qwen_lora import QwenLoraExtractor


ROOT = Path(__file__).resolve().parents[1]


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace


class PipelineTests(unittest.TestCase):
    def pipeline(self, *, empty=False, extraction_error=False, tts_error=False, saved=True):
        self.events = []
        def speak(text):
            self.events.append("speak")
            if tts_error and len([e for e in self.events if e == "speak"]) == 2:
                raise RuntimeError("speaker disconnected")
        def transcribe(on_status):
            on_status("Transcribing")
            return "" if empty else "six millimeter polyp"
        def extract(text, strict):
            self.assertTrue(strict)
            if extraction_error:
                raise ValueError("bad JSON")
            return {"lesion_type": "polyp"}
        def persist(text, result):
            self.events.append("persist")
            return saved
        namespace = load_functions("pipeline.py", {"process_finding", "FindingProcessingError"}, {
            "speak": speak, "transcribe_finding": transcribe, "extract_finding": extract,
            "_spoken_summary": lambda result: "polyp", "_append_finding": persist,
        })
        return namespace

    def test_result_saved_before_speech(self):
        namespace = self.pipeline()
        statuses = []
        result = namespace["process_finding"](on_status=statuses.append, strict=True)
        self.assertEqual(self.events, ["speak", "persist", "speak"])
        self.assertEqual(statuses, ["Prompting", "Recording", "Transcribing", "Extracting", "Speaking"])
        self.assertEqual(result["transcription"], "six millimeter polyp")

    def test_empty_asr_does_not_save(self):
        namespace = self.pipeline(empty=True)
        self.assertIsNone(namespace["process_finding"](on_status=lambda _: None, strict=True))
        self.assertNotIn("persist", self.events)

    def test_extraction_failure_preserves_transcript(self):
        namespace = self.pipeline(extraction_error=True)
        with self.assertRaises(namespace["FindingProcessingError"]) as caught:
            namespace["process_finding"](on_status=lambda _: None, strict=True)
        self.assertEqual(caught.exception.stage, "Extracting")
        self.assertEqual(caught.exception.transcription, "six millimeter polyp")
        self.assertNotIn("persist", self.events)

    def test_tts_failure_happens_after_persistence(self):
        namespace = self.pipeline(tts_error=True)
        with self.assertRaises(namespace["FindingProcessingError"]) as caught:
            namespace["process_finding"](on_status=lambda _: None, strict=True)
        self.assertEqual(caught.exception.stage, "Speaking")
        self.assertEqual(self.events, ["speak", "persist", "speak"])

    def test_save_failure_suppresses_feedback(self):
        namespace = self.pipeline(saved=False)
        namespace["process_finding"](on_status=lambda _: None, strict=True)
        self.assertEqual(self.events, ["speak", "persist"])

    def test_strict_json_reports_malformed_output_and_keeps_legacy_default(self):
        for text in ("not JSON", "[1,2]", "{broken}", "null"):
            with self.assertRaises(ValueError):
                QwenLoraExtractor._extract_json(text, strict=True)
            self.assertEqual(QwenLoraExtractor._extract_json(text), {})
        self.assertEqual(QwenLoraExtractor._extract_json('```json\n{"size_mm":6}\n```', strict=True), {"size_mm": 6})
        result = QwenLoraExtractor._normalize_result({"size_mm": 6.0, "extra": "discard"})
        self.assertEqual(len(result), 7)
        self.assertEqual(result["size_mm"], 6)


class DetectorCancellationTests(unittest.TestCase):
    def detector(self, mode, cancel, cancel_during_inference=False, empty=False):
        stream = Mock()
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock(return_value=False)
        audio_queue = queue.Queue()
        audio_queue.put(b"speech")
        audio_queue.put(b"silence")
        if empty:
            def get(timeout):
                self.assertLessEqual(timeout, 0.1)
                cancel.set()
                raise queue.Empty
            audio_queue = SimpleNamespace(get=get)
        def classify(*args):
            if cancel_during_inference:
                cancel.set()
            return True
        namespace = {
            "config": SimpleNamespace(AUDIO_DEVICE=7, USE_FINETUNED_VAD=mode == "cnn"),
            "collections": collections, "queue": queue, "audio_queue": audio_queue,
            "sd": SimpleNamespace(RawInputStream=Mock(return_value=stream)),
            "FRAME_MS": 20, "SAMPLE_RATE": 16000, "PRE_ROLL_FRAMES": 15,
            "FRAME_SAMPLES": 320, "CHANNELS": 1, "END_SILENCE_FRAMES": 1,
            "MAX_TRIGGER_FRAMES": 2, "audio_callback": lambda *args: None,
            "clear_audio_queue": lambda: None,
            "vad": SimpleNamespace(is_speech=lambda frame, rate: frame == b"speech"),
            "_get_whisper_model": lambda: object(), "load_finetuned_vad_model": lambda: None,
            "_transcribe_voiced_frames": classify, "process_voiced_frames": classify,
        }
        return load_functions("modules/voice_activation/detector.py",
                              {"_wait_for_trigger_base", "_wait_for_trigger_finetuned", "wait_for_trigger"}, namespace), stream

    def test_cancel_empty_queue_and_close_stream_for_both_backends(self):
        for mode in ("whisper", "cnn"):
            cancel = threading.Event()
            namespace, stream = self.detector(mode, cancel, empty=True)
            self.assertFalse(namespace["wait_for_trigger"](cancel))
            stream.__exit__.assert_called_once()
            self.assertEqual(namespace["sd"].RawInputStream.call_args.kwargs["device"], 7)

    def test_cancel_during_inference_discards_trigger(self):
        for mode in ("whisper", "cnn"):
            cancel = threading.Event()
            namespace, stream = self.detector(mode, cancel, cancel_during_inference=True)
            self.assertFalse(namespace["wait_for_trigger"](cancel))
            stream.__exit__.assert_called_once()

    def test_legacy_call_without_cancellation_still_detects(self):
        for mode in ("whisper", "cnn"):
            namespace, stream = self.detector(mode, threading.Event())
            self.assertTrue(namespace["wait_for_trigger"]())
            stream.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
