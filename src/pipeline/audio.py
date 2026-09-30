"""Audio output that reports when sound actually reaches the speaker.

One low-latency output stream is opened up front and kept running (playing
silence between sentences). Opening a stream per sentence was measured at
~250 ms on this machine's audio device, plus ~180 ms of device latency at the
default ("high") setting; a persistent "low" stream starts a sentence within one
callback (0-16 ms) plus ~90 ms of device latency.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from src.tts import Audio


class _Sound:
    def __init__(self, samples: np.ndarray):
        self.samples = samples
        self.position = 0
        self.start_ms = float("nan")
        self.started = threading.Event()
        self.done = threading.Event()


class Playback:
    """A sound that has started playing."""

    def __init__(self, sound: _Sound, tail_s: float):
        self._sound = sound
        self._tail_s = tail_s
        self.start_ms = sound.start_ms

    def wait(self, timeout: float | None = None) -> None:
        """Block until the sound has left the speaker."""
        if self._sound.done.wait(timeout):
            time.sleep(self._tail_s)  # the last buffer is still in the device


class AudioPlayer:
    """play() returns once the sound's first buffer is handed to the device, with
    start_ms = that moment plus the device's output latency (PortAudio's DAC
    timestamps, or the stream's reported latency where the host API has none)."""

    def __init__(self, sample_rate: int, device=None, latency: str = "low"):
        import sounddevice

        self.sample_rate = sample_rate
        self._current: _Sound | None = None
        self._stream = sounddevice.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32", device=device, latency=latency, callback=self._fill
        )
        self._stream.start()

    @property
    def latency_s(self) -> float:
        return float(self._stream.latency)

    def _fill(self, outdata, frames, time_info, status) -> None:
        outdata[:] = 0.0
        sound = self._current
        if sound is None:
            return
        if not sound.started.is_set():
            delay = time_info.outputBufferDacTime - time_info.currentTime
            if not 0.0 < delay < 1.0:
                delay = self._stream.latency
            sound.start_ms = (time.perf_counter() + delay) * 1000.0
            sound.started.set()
        chunk = sound.samples[sound.position : sound.position + frames]
        outdata[: len(chunk), 0] = chunk
        sound.position += frames
        if sound.position >= len(sound.samples):
            self._current = None
            sound.done.set()

    def play(self, audio: Audio, start_timeout: float = 2.0) -> Playback:
        samples = np.asarray(audio.samples, np.float32)
        if audio.sample_rate != self.sample_rate:  # engines differ (OneCore 16 kHz, SAPI 22 kHz)
            times = np.arange(int(len(samples) * self.sample_rate / audio.sample_rate)) / self.sample_rate
            samples = np.interp(times, np.arange(len(samples)) / audio.sample_rate, samples).astype(np.float32)
        sound = _Sound(np.ascontiguousarray(samples))
        self._current = sound
        if not sound.started.wait(start_timeout):
            raise RuntimeError("the audio device did not start playing")
        return Playback(sound, self.latency_s)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()
