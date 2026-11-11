"""The live pipeline, Stages 1-6:

    webcam frame -> landmarks -> sign segmentation -> sign classification ->
    gloss buffer -> gloss-to-sentence translation -> TTS -> audio output

Run from the project root:
    python -m src.pipeline.live                             # webcam, preview window, audio
    python -m src.pipeline.live --duration 90 --no-window
    python -m src.pipeline.live --source data/raw/replay/utterances.mp4    # a video, replayed in real time

Every stage boundary is timestamped on one high-resolution clock (time.perf_counter). Each spoken
utterance is logged with its latency breakdown, from "gesture completes" (the
last hand motion of its last sign) to "audio starts" (sound reaching the
speaker), and written to logs/pipeline_<time>.jsonl; see trace.py for the
stages. A summary table prints on exit.

Two threads: the capture thread does the per-frame work (tracking,
segmentation, classification, gloss buffer); the speech thread does
translation, TTS and playback, so a slow sentence never stalls the camera.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from src.capture import FEATURE_DIM, LAYOUT, HolisticTracker, draw_landmarks, extract_features
from src.capture.segmenter import SegmenterConfig, SignSegmenter
from src.capture.sequence import trim_to_hands
from src.recognition.model import SignRecognizer
from src.translation import ClaudeTranslator, RuleBasedTranslator
from src.tts import create_engine, emphasis_for_sentence, expression_intensity, pace_from_durations, synthesize

from .audio import AudioPlayer
from .gloss_buffer import BufferConfig, GlossBuffer
from .sources import open_source
from .trace import SignEvent, TraceLog, UtteranceTrace, now_ms, summarize

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("signspeak.pipeline")
WINDOW = "SignSpeak - live pipeline"
_DEFAULT_TORCH_THREADS = torch.get_num_threads()


@dataclass
class PipelineConfig:
    source: str = "0"
    checkpoint: Path = ROOT / "models" / "asl_citizen_sign_classifier.pt"
    # Faster sign-end detection than the segmenter's own defaults (200 ms pause,
    # smoothing 0.6): measured -140 ms on detect with no extra split signs.
    segmenter: SegmenterConfig = field(default_factory=lambda: SegmenterConfig(pause_ms=120, smoothing=0.8))
    buffer: BufferConfig = field(default_factory=BufferConfig)
    # "rules" by default: offline, instant and always available. The Claude path
    # is real but has never run end-to-end (our key returns 401), and an API call
    # puts a network round trip inside the latency budget, so it is opt-in:
    # "claude" forces it, "auto" uses it when ANTHROPIC_API_KEY is set.
    translator: str = "rules"
    voice: str | None = None
    duration_s: float | None = None
    show_window: bool = True
    play_audio: bool = True
    log_dir: Path = ROOT / "logs"
    # Performance levers; src/pipeline/benchmark.py measures each one.
    preview_scale: float = 0.5  # half-size preview: -4 ms/frame of drawing
    torch_threads: int | None = 1  # one thread beats four for this small model (4.4 vs 5.5 ms)
    track_max_width: int | None = 640  # track webcam frames at 640x360: -10 ms/frame
    source_size: tuple[int, int] | None = None  # resize source frames, e.g. a replay to webcam size
    loop_source: bool = False  # restart a video source when it ends (demo backup mode)
    # A slow or flaky network must never stall speech: give up on the API fast and
    # speak the offline translation instead.
    translate_timeout_s: float = 3.0


class PipelineListener:
    """Callbacks for a UI; every method is optional. on_ready, on_frame, on_sign
    and on_utterance run on the capture thread, on_spoken and on_speech_done on
    the speech thread, so keep them quick."""

    def on_ready(self, info: dict) -> None:
        pass

    def on_error(self, message: str) -> None:
        pass

    def on_frame(self, frame: np.ndarray, landmarks, signing: bool) -> None:
        pass

    def on_sign(self, sign: SignEvent, kept: bool) -> None:
        pass

    def on_utterance(self, trace: UtteranceTrace) -> None:
        pass

    def on_spoken(self, trace: UtteranceTrace) -> None:
        pass

    def on_speech_done(self, trace: UtteranceTrace) -> None:
        pass


class SpeechStage:
    """Translation -> TTS -> playback for one utterance, stamping each boundary."""

    def __init__(self, translator, engine, player: AudioPlayer | None):
        self.translator = translator
        self.engine = engine
        self.player = player

    def speak(self, trace: UtteranceTrace):
        trace.translate_start_ms = now_ms()
        trace.sentence = self.translator.translate([sign.label for sign in trace.signs])
        trace.translate_end_ms = now_ms()

        trace.pace = pace_from_durations([sign.duration_s for sign in trace.signs])
        regions = emphasis_for_sentence(trace.sentence, [(sign.label, sign.intensity) for sign in trace.signs])
        words = trace.sentence.split()
        trace.emphasized = [" ".join(words[r.start : r.end]) for r in regions]
        audio = synthesize(trace.sentence, trace.pace, regions, engine=self.engine)
        trace.tts_end_ms = now_ms()
        trace.audio_seconds = audio.duration

        if self.player is None or audio.duration == 0:
            trace.audio_start_ms = trace.tts_end_ms  # nothing to play: measured up to rendering
            return trace, None
        playback = self.player.play(audio)
        trace.audio_start_ms = playback.start_ms
        trace.audio_played = True
        return trace, playback


class Pipeline:
    def __init__(self, config: PipelineConfig, listener: PipelineListener | None = None, source=None):
        self.config = config
        self.listener = listener or PipelineListener()
        self._source = source  # a ready frame source (read/close) in place of config.source, e.g. the demo's switchable one
        self._stop = threading.Event()
        self.translator_name = ""
        self.voice = ""
        self.recognizer = SignRecognizer(config.checkpoint)
        self.segmenter = SignSegmenter(config.segmenter)
        self.buffer = GlossBuffer(config.buffer)
        self.traces: list[UtteranceTrace] = []
        self.warmup: dict[str, float] = {}
        self.trace_log = TraceLog(config.log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        self._queue: queue.Queue = queue.Queue()
        self._speech_ready = threading.Event()
        self._speech_error: BaseException | None = None
        self._count = 0
        self._last_said = ""
        self._last_latency: float | None = None
        self.track_ms: list[float] = []  # tracking time per frame
        self.signs_seen = 0

    # ------------------------------------------------------------- speech thread

    def stop(self) -> None:
        """Ask run() to finish (from any thread)."""
        self._stop.set()

    def _make_translator(self):
        use_claude = self.config.translator == "claude" or (
            self.config.translator == "auto" and os.environ.get("ANTHROPIC_API_KEY")
        )
        if use_claude:
            return ClaudeTranslator(timeout=self.config.translate_timeout_s, max_retries=0)
        return RuleBasedTranslator()

    def _speech_thread(self) -> None:
        try:
            started = now_ms()
            translator = self._make_translator()
            translator.translate(["hello"])  # first call: connection setup, or discovering the API is unusable
            self.warmup["translator"] = now_ms() - started
            if isinstance(translator, ClaudeTranslator) and translator.api_disabled:
                log.warning("Claude API unusable (see above); translating with the offline rules this session")
            using_api = isinstance(translator, ClaudeTranslator) and not translator.api_disabled
            self.translator_name = f"Claude ({translator.model})" if using_api else "offline rules"
            log.info("translator: %s", self.translator_name)

            started = now_ms()
            engine = create_engine(self.config.voice)
            ready = synthesize("Ready.", engine=engine)  # first render loads the voice
            self.warmup["tts"] = now_ms() - started
            self.voice = engine.voice
            log.info("speech engine: %s (%s)", engine.name, engine.voice)

            player = None
            if self.config.play_audio:
                started = now_ms()
                player = AudioPlayer(ready.sample_rate)  # opened once, kept running
                self.warmup["audio_device"] = now_ms() - started
                log.info("audio output: %d Hz, device latency %.0f ms", player.sample_rate, player.latency_s * 1000)
            stage = SpeechStage(translator, engine, player)
        except BaseException as error:  # noqa: BLE001 - reported to the capture thread
            self._speech_error = error
            self._speech_ready.set()
            return
        self._speech_ready.set()
        try:
            self._speak_until_done(stage)
        finally:
            if player is not None:
                player.close()

    def _speak_until_done(self, stage: SpeechStage) -> None:

        while (trace := self._queue.get()) is not None:
            playback = None
            try:
                trace, playback = stage.speak(trace)
                self._last_said, self._last_latency = trace.sentence, trace.stages()["end_to_end"]
                log.info("%s", trace.describe())
                self.listener.on_spoken(trace)
            except Exception as error:  # noqa: BLE001 - keep the session alive
                trace.error = repr(error)
                log.exception("speech stage failed for utterance #%d", trace.index)
                self.listener.on_error(f"Couldn't speak that sentence ({error})")
            self.traces.append(trace)
            self.trace_log.write(trace)
            if playback is not None:
                playback.wait()  # don't talk over ourselves
            self.listener.on_speech_done(trace)

    # ------------------------------------------------------------- capture thread

    def run(self) -> str:
        torch.set_num_threads(self.config.torch_threads or _DEFAULT_TORCH_THREADS)
        speech = threading.Thread(target=self._speech_thread, name="speech", daemon=True)
        speech.start()
        started = now_ms()
        tracker = HolisticTracker()
        self.warmup["tracker"] = now_ms() - started
        started = now_ms()
        self._warm_classifier()
        self.warmup["classifier"] = now_ms() - started
        self._speech_ready.wait(120)
        if self._speech_error is not None:
            self.listener.on_error(f"Speech output failed to start: {self._speech_error}")
            raise RuntimeError("speech stage failed to start") from self._speech_error

        try:
            source = self._source or open_source(self.config.source, self.config.source_size, self.config.loop_source)
        except Exception as error:
            tracker.close()
            self.listener.on_error(f"Couldn't open the video source: {error}")
            raise
        self.listener.on_ready(
            {
                "translator": self.translator_name,
                "voice": self.voice,
                "labels": list(self.recognizer.label_names),
                "warmup_ms": {k: round(v) for k, v in self.warmup.items()},
            }
        )
        deadline = time.monotonic() + self.config.duration_s if self.config.duration_s else None
        frames, loop_started = 0, time.monotonic()
        log.info("running on %s; stop with q in the window, Ctrl+C, or --duration", self.config.source)
        try:
            while True:
                frame = source.read()
                if frame is None:
                    break
                started = now_ms()
                landmarks = tracker.process(_fit_width(frame, self.config.track_max_width))
                self.track_ms.append(now_ms() - started)
                frames += 1
                segment = self.segmenter.push(extract_features(landmarks), landmarks.timestamp_ms)
                if segment is not None:
                    self._on_segment(segment)
                hands = landmarks.left_hand is not None or landmarks.right_hand is not None
                complete = self.buffer.poll(now_ms(), hands, self.segmenter.in_sign)
                if complete:
                    self._enqueue(*complete)
                self.listener.on_frame(frame, landmarks, self.segmenter.in_sign)
                if self.config.show_window and not self._show(frame, landmarks):
                    break
                if self._stop.is_set() or (deadline is not None and time.monotonic() >= deadline):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            fps = frames / max(time.monotonic() - loop_started, 1e-6)
            segment = self.segmenter.flush()
            if segment is not None:
                self._on_segment(segment)
            rest = self.buffer.flush("end")
            if rest:
                self._enqueue(*rest)
            self._queue.put(None)
            speech.join(timeout=60)
            source.close()
            tracker.close()
            cv2.destroyAllWindows()
        report = summarize(self.traces, fps, self.warmup)
        if self.track_ms:
            report = f"Tracking: {np.median(self.track_ms):.0f} ms per frame (median); {self.signs_seen} signs recognized.\n" + report
        log.info("per-utterance log: %s", self.trace_log.path)
        return report

    def _warm_classifier(self) -> None:
        sequence = np.zeros((20, FEATURE_DIM), np.float32)
        sequence[:, LAYOUT["right_hand.present"]] = 1.0
        sequence[:, LAYOUT["pose.present"]] = 1.0
        self.recognizer.predict(sequence)

    def _on_segment(self, segment) -> None:
        detected = now_ms()
        top = self.recognizer.predict(segment.features, top_k=3)
        classified = now_ms()
        if not top:
            log.info("segment of %d frames had no hands; skipped", segment.num_frames)
            return
        self.signs_seen += 1
        trimmed = trim_to_hands(segment.features)
        if trimmed is None:
            trimmed = segment.features
        sign = SignEvent(
            label=top[0][0],
            confidence=top[0][1],
            gesture_end_ms=float(segment.gesture_end_ms),
            detected_ms=detected,
            classified_ms=classified,
            duration_s=sign_duration_s(segment.timestamps_ms, len(trimmed)),
            intensity=expression_intensity(trimmed),
            alternatives=[(label, round(p, 3)) for label, p in top],
        )
        kept = self.buffer.add(sign)
        self.listener.on_sign(sign, kept)
        log.info(
            "sign %-12s p=%.2f  detect %4.0f ms  classify %3.0f ms  %s  (then: %s)",
            sign.label,
            sign.confidence,
            detected - sign.gesture_end_ms,
            classified - detected,
            "buffered" if kept else "DROPPED, too unsure",
            ", ".join(f"{label} {p:.2f}" for label, p in top[1:]),
        )

    def _enqueue(self, signs: list[SignEvent], reason: str) -> None:
        self._count += 1
        trace = UtteranceTrace(self._count, signs, now_ms(), reason)
        log.info("utterance #%d complete (%s): %s", trace.index, reason, " ".join(s.label for s in signs))
        self.listener.on_utterance(trace)
        self._queue.put(trace)

    def _show(self, frame, landmarks) -> bool:
        scale = self.config.preview_scale
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        view = cv2.flip(frame, 1)
        draw_landmarks(view, landmarks, mirror=True)  # landmarks are normalized, so any size works
        lines = [
            ("SIGNING" if self.segmenter.in_sign else "waiting") + f"   motion {self.segmenter.motion:.2f}",
            "buffer: " + (" ".join(self.buffer.pending) or "-"),
            "said:   " + (self._last_said or "-"),
        ]
        if self._last_latency is not None:
            lines.append(f"last end-to-end latency: {self._last_latency:.0f} ms")
        for row, text in enumerate(lines):
            cv2.putText(view, text, (12, 30 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27) and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1


def sign_duration_s(timestamps_ms: np.ndarray, frames: int) -> float:
    """Seconds spanned by `frames` frames at the segment's measured frame rate.
    Not frames / the training rate (15 fps): at a webcam's 8 fps that read every
    sign as twice as fast, and pace pinned speech at its fastest setting."""
    if len(timestamps_ms) < 2:
        return 0.0
    interval_ms = (timestamps_ms[-1] - timestamps_ms[0]) / (len(timestamps_ms) - 1)
    return frames * interval_ms / 1000


def _fit_width(frame: np.ndarray, max_width: int | None) -> np.ndarray:
    """Downscale (keeping the aspect ratio) if the frame is wider than max_width."""
    if not max_width or frame.shape[1] <= max_width:
        return frame
    scale = max_width / frame.shape[1]
    return cv2.resize(frame, (max_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults, buffer_defaults = PipelineConfig().segmenter, BufferConfig()  # the pipeline's tuned values
    parser.add_argument("--source", default="0", help="webcam index, or a video file to replay in real time")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--no-audio", action="store_true", help="render speech but don't play it")
    parser.add_argument("--translator", choices=["rules", "claude", "auto"], default=PipelineConfig.translator,
                        help="rules: offline grammar (default). claude: the API. auto: the API if a key is set")  # fmt: skip
    parser.add_argument("--voice", help="part of a voice name, e.g. Mark or Zira")
    parser.add_argument("--checkpoint", type=Path, default=PipelineConfig.checkpoint)
    parser.add_argument("--gap-ms", type=float, default=buffer_defaults.gap_ms, help="pause that ends a sentence")
    parser.add_argument("--hands-down-ms", type=float, default=buffer_defaults.hands_down_ms)
    parser.add_argument("--min-confidence", type=float, default=buffer_defaults.min_confidence)
    parser.add_argument("--pause-ms", type=int, default=defaults.pause_ms, help="pause that ends a sign")
    parser.add_argument("--smoothing", type=float, default=defaults.smoothing, help="weight of the newest motion reading")
    parser.add_argument("--preview-scale", type=float, default=PipelineConfig.preview_scale)
    parser.add_argument("--torch-threads", type=int, default=PipelineConfig.torch_threads)
    parser.add_argument("--track-width", type=int, default=PipelineConfig.track_max_width, help="max frame width for tracking")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d %(threadName)-10s %(message)s", datefmt="%H:%M:%S")
    config = PipelineConfig(
        source=args.source,
        checkpoint=args.checkpoint,
        segmenter=SegmenterConfig(pause_ms=args.pause_ms, smoothing=args.smoothing),
        buffer=BufferConfig(gap_ms=args.gap_ms, hands_down_ms=args.hands_down_ms, min_confidence=args.min_confidence),
        translator=args.translator,
        voice=args.voice,
        duration_s=args.duration,
        show_window=not args.no_window,
        play_audio=not args.no_audio,
        preview_scale=args.preview_scale,
        torch_threads=args.torch_threads,
        track_max_width=args.track_width,
    )
    print("\n" + Pipeline(config).run())


if __name__ == "__main__":
    main()
