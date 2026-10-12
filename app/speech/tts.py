"""Text to speech via Piper.

Piper is called as a subprocess rather than through the `piper-tts` pip
package, which is unreliable on Windows. The binary reads text on stdin and
writes raw signed 16-bit mono PCM on stdout, which we hand straight to the
audio device -- no temporary WAV files, no ffmpeg.

Setup (one time):
  1. Download piper_windows_amd64.zip from the Piper releases page and unzip
     it to data/voices/piper/
  2. Download a voice: en_US-ryan-high.onnx plus en_US-ryan-high.onnx.json
     into data/voices/
  3. Point PIPER_EXE and PIPER_VOICE in .env at those files.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from app.config import settings
from app.logging_setup import get_logger

log = get_logger(__name__)

DEFAULT_SAMPLE_RATE = 22050


class PiperUnavailable(RuntimeError):
    pass


class Speaker:
    def __init__(self, exe: str | None = None, voice: str | None = None) -> None:
        self.exe = settings.abs_path(exe or settings.piper_exe)
        self.voice = settings.abs_path(voice or settings.piper_voice)
        self.sample_rate = self._read_sample_rate()

    def available(self) -> bool:
        return self.exe.exists() and self.voice.exists()

    def _read_sample_rate(self) -> int:
        """Voice config carries the true rate; guessing produces chipmunks."""
        cfg = Path(str(self.voice) + ".json")
        if not cfg.exists():
            return DEFAULT_SAMPLE_RATE
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            return int(data.get("audio", {}).get("sample_rate", DEFAULT_SAMPLE_RATE))
        except (json.JSONDecodeError, ValueError, OSError):
            return DEFAULT_SAMPLE_RATE

    def synthesize(self, text: str) -> np.ndarray:
        """Return mono float32 samples in [-1, 1] for one sentence."""
        if not self.available():
            raise PiperUnavailable(
                f"Piper not found. Expected exe at {self.exe} and voice at "
                f"{self.voice}. See the setup notes in app/speech/tts.py."
            )
        if not text.strip():
            return np.zeros(0, dtype=np.float32)

        proc = subprocess.run(
            [
                str(self.exe),
                "--model", str(self.voice),
                "--output-raw",
                "--quiet",
            ],
            input=text.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            raise PiperUnavailable(
                f"piper exited {proc.returncode}: "
                f"{proc.stderr.decode(errors='replace')[:300]}"
            )

        pcm = np.frombuffer(proc.stdout, dtype=np.int16)
        return (pcm.astype(np.float32) / 32768.0).copy()
