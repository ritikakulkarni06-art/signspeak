"""SSML -> audio with the speech engines built into Windows.

  * OneCoreEngine - the newer Windows voices (Mark, Zira, David) via WinRT.
  * SapiEngine    - the classic SAPI5 desktop voices via COM; the fallback.

Both run offline with no API key. They are called synchronously (OneCore runs
its own short asyncio loop), so call them from ordinary code or a worker
thread, not from inside a running event loop.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import wave
from dataclasses import dataclass

import numpy as np


@dataclass
class Audio:
    samples: np.ndarray  # mono float32 in [-1, 1]
    sample_rate: int
    ssml: str = ""

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate

    @classmethod
    def from_wav_bytes(cls, data: bytes, ssml: str = "") -> Audio:
        with wave.open(io.BytesIO(data)) as file:
            rate, width, channels = file.getframerate(), file.getsampwidth(), file.getnchannels()
            raw = file.readframes(file.getnframes())
        dtype, scale = {1: (np.uint8, 128.0), 2: (np.int16, 32768.0), 4: (np.int32, 2147483648.0)}[width]
        samples = np.frombuffer(raw, dtype).astype(np.float32)
        if width == 1:
            samples -= 128.0
        samples = samples.reshape(-1, channels).mean(axis=1) / scale
        return cls(samples.astype(np.float32), rate, ssml)

    def trim_silence(self, threshold_db: float = -40.0, keep_lead_ms: float = 10.0, keep_tail_ms: float = 80.0) -> Audio:
        """Drop the silence engines pad around speech. OneCore adds ~110 ms before
        and ~770 ms after every sentence (measured): the lead delays when speech is
        heard, the tail delays the next sentence."""
        if len(self.samples) == 0:
            return self
        window = max(1, self.sample_rate // 100)  # 10 ms envelope
        envelope = np.convolve(np.abs(self.samples), np.ones(window) / window, mode="same")
        audible = np.flatnonzero(envelope > envelope.max() * 10 ** (threshold_db / 20))
        if len(audible) == 0:
            return Audio(np.zeros(0, np.float32), self.sample_rate, self.ssml)
        start = max(0, audible[0] - int(keep_lead_ms * self.sample_rate / 1000))
        end = min(len(self.samples), audible[-1] + int(keep_tail_ms * self.sample_rate / 1000))
        return Audio(self.samples[start:end], self.sample_rate, self.ssml)

    def save(self, path) -> None:
        pcm = (np.clip(self.samples, -1.0, 1.0) * 32767).astype("<i2")
        with wave.open(str(path), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(self.sample_rate)
            file.writeframes(pcm.tobytes())

    def play(self, blocking: bool = True) -> None:
        import sounddevice

        sounddevice.play(self.samples, self.sample_rate, blocking=blocking)


class OneCoreEngine:
    name = "onecore"

    def __init__(self, voice: str | None = None):
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer

        self._synthesizer = SpeechSynthesizer()
        if voice:
            matches = [v for v in SpeechSynthesizer.all_voices if voice.lower() in v.display_name.lower()]
            if not matches:
                raise ValueError(f"no OneCore voice matching {voice!r}; have {self.voices()}")
            self._synthesizer.voice = matches[0]
        self.voice = self._synthesizer.voice.display_name

    @staticmethod
    def voices() -> list[str]:
        from winrt.windows.media.speechsynthesis import SpeechSynthesizer

        return [v.display_name for v in SpeechSynthesizer.all_voices]

    def render(self, ssml: str) -> Audio:
        return Audio.from_wav_bytes(asyncio.run(self._render(ssml)), ssml)

    async def _render(self, ssml: str) -> bytes:
        from winrt.windows.storage.streams import DataReader

        stream = await self._synthesizer.synthesize_ssml_to_stream_async(ssml)
        reader = DataReader(stream.get_input_stream_at(0))
        await reader.load_async(stream.size)
        buffer = bytearray(stream.size)
        reader.read_bytes(buffer)
        return bytes(buffer)


class SapiEngine:
    name = "sapi5"
    _SPEAK_XML = 8  # SVSFIsXML; SSML is recognized from the <speak> root
    _CREATE_FOR_WRITE = 3  # SSFMCreateForWrite

    def __init__(self, voice: str | None = None):
        import comtypes.client

        self._client = comtypes.client
        self._speaker = comtypes.client.CreateObject("SAPI.SpVoice")
        if voice:
            tokens = [t for t in self._speaker.GetVoices() if voice.lower() in t.GetDescription().lower()]
            if not tokens:
                raise ValueError(f"no SAPI5 voice matching {voice!r}")
            self._speaker.Voice = tokens[0]
        self.voice = self._speaker.Voice.GetDescription()

    def render(self, ssml: str) -> Audio:
        handle, path = tempfile.mkstemp(suffix=".wav", prefix="signspeak-tts-")
        os.close(handle)
        try:
            stream = self._client.CreateObject("SAPI.SpFileStream")
            stream.Open(path, self._CREATE_FOR_WRITE)
            self._speaker.AudioOutputStream = stream
            self._speaker.Speak(ssml, self._SPEAK_XML)
            stream.Close()
            with open(path, "rb") as file:
                return Audio.from_wav_bytes(file.read(), ssml)
        finally:
            os.remove(path)


def create_engine(voice: str | None = None, prefer: str | None = None):
    """The first engine that starts: OneCore, then SAPI5 (or `prefer` first)."""
    order = [OneCoreEngine, SapiEngine]
    if prefer:
        order.sort(key=lambda engine: engine.name != prefer)
    errors = []
    for engine in order:
        try:
            return engine(voice)
        except Exception as error:  # noqa: BLE001 - missing bindings, voice, or OS
            errors.append(f"{engine.name}: {error}")
    raise RuntimeError("no speech engine available: " + "; ".join(errors))
