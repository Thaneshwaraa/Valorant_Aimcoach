"""Endpointing: decide when the speaker has stopped.

An energy gate with an adaptive noise floor and a hangover timer. Chosen over
Silero VAD for the MVP because it adds no model download, no ONNX session and
no VRAM, and on a headset microphone it is entirely adequate. If you move to a
noisy open-mic setup, swapping in Silero behind this same interface is the
upgrade path.

The hangover is what makes it usable: without it, every natural pause between
words would end the turn.
"""

from __future__ import annotations

import numpy as np


class EnergyVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 30,
        hangover_ms: int = 700,
        min_speech_ms: int = 300,
        threshold_multiplier: float = 3.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * frame_ms / 1000)
        self.hangover_frames = max(1, hangover_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.threshold_multiplier = threshold_multiplier

        self._noise_floor: float | None = None
        self._speech_frames = 0
        self._silence_frames = 0
        self.triggered = False

    def reset(self) -> None:
        self._speech_frames = 0
        self._silence_frames = 0
        self.triggered = False

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(frame, dtype=np.float64)) + 1e-12))

    def accept(self, frame: np.ndarray) -> bool:
        """Feed one frame. Returns True when the utterance has ended."""
        energy = self._rms(frame)

        if self._noise_floor is None:
            self._noise_floor = energy
        threshold = max(self._noise_floor * self.threshold_multiplier, 0.005)

        if energy > threshold:
            self._speech_frames += 1
            self._silence_frames = 0
            if self._speech_frames >= self.min_speech_frames:
                self.triggered = True
        else:
            # Track the noise floor only while nobody is talking, so a long
            # sentence cannot drag the threshold up above the speaker.
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * energy
            if self.triggered:
                self._silence_frames += 1
                if self._silence_frames >= self.hangover_frames:
                    return True
            else:
                self._speech_frames = 0
        return False
