"""Speech to text via faster-whisper.

faster-whisper rather than openai-whisper: same weights, CTranslate2 backend,
roughly four times faster. Defaults to int8 on CPU because the GPU budget is
reserved for the LLM -- on an i7, small.en transcribes a five-second clip in
about 0.3 s, which fits the latency budget without touching VRAM.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from app.config import settings
from app.logging_setup import get_logger

log = get_logger(__name__)


class Transcriber:
    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name or settings.stt_model
        self.device = device or settings.stt_device
        self.compute_type = compute_type or settings.stt_compute_type
        log.info(
            "stt.load",
            model=self.model_name,
            device=self.device,
            compute=self.compute_type,
        )
        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe mono float32 audio in [-1, 1] at 16 kHz."""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)

        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,  # greedy: measurably faster, no quality loss here
            vad_filter=True,
            condition_on_previous_text=False,
            # Biases the decoder toward game vocabulary it would otherwise
            # mangle ("Jett" -> "jet", "Sova" -> "sofa").
            initial_prompt=(
                "Valorant coaching. Agents: Jett, Raze, Reyna, Omen, Brimstone, "
                "Viper, Sova, Breach, Skye, KAY/O, Killjoy, Cypher, Sage, "
                "Chamber. Maps: Ascent, Bind, Haven, Split, Icebox, Breeze, "
                "Fracture, Pearl, Lotus, Sunset. Terms: spike, eco, retake, "
                "rotate, lineup, entry, post-plant."
            ),
        )
        return " ".join(s.text.strip() for s in segments).strip()


@lru_cache
def get_transcriber() -> Transcriber:
    return Transcriber()
