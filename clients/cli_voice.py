"""Voice client: microphone in, spoken answer out.

    uv run python clients/cli_voice.py              # hands-free, VAD endpointing
    uv run python clients/cli_voice.py --push       # push-to-talk (press Enter)

Architecture note: speech recognition and synthesis run here, in the client
process, not on the server. Two reasons. Audio never crosses the network, so
the API stays a simple text service; and the sentence-to-speech loop sits next
to the audio device, which is what keeps the response latency low.

The pipeline per turn:
    mic -> VAD -> Whisper -> POST /chat/stream -> sentence splitter -> Piper
    -> speaker, with each sentence spoken while the next is still generating.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time

import httpx
import numpy as np
from rich.console import Console

from app.speech.streaming import clean_for_speech, split_sentences

console = Console()

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_LEN = SAMPLE_RATE * FRAME_MS // 1000
MAX_UTTERANCE_S = 20


class AudioPlayer:
    """Plays synthesized sentences in order, on a background thread.

    A queue rather than direct playback: synthesis of sentence N+1 overlaps
    with playback of sentence N, which is where the perceived speed comes from.
    """

    def __init__(self, sample_rate: int) -> None:
        import sounddevice as sd

        self._sd = sd
        self.sample_rate = sample_rate
        self._q: queue.Queue[np.ndarray | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                self._sd.play(item, self.sample_rate)
                self._sd.wait()
            except Exception as exc:  # noqa: BLE001 - audio device errors vary
                console.print(f"[red]playback failed: {exc}[/red]")

    def play(self, samples: np.ndarray) -> None:
        if samples.size:
            self._q.put(samples)

    def wait(self) -> None:
        while not self._q.empty():
            time.sleep(0.05)

    def close(self) -> None:
        self._q.put(None)


def record_push_to_talk() -> np.ndarray:
    import sounddevice as sd

    console.print("[cyan]Press Enter to start recording...[/cyan]", end="")
    input()
    console.print("[red]recording... press Enter to stop[/red]", end="")

    frames: list[np.ndarray] = []
    stop = threading.Event()

    def _wait_for_enter() -> None:
        input()
        stop.set()

    threading.Thread(target=_wait_for_enter, daemon=True).start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_LEN
    ) as stream:
        while not stop.is_set():
            block, _overflow = stream.read(FRAME_LEN)
            frames.append(block[:, 0].copy())
    console.print()
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


def record_with_vad() -> np.ndarray:
    import sounddevice as sd

    from app.speech.vad import EnergyVAD

    vad = EnergyVAD(sample_rate=SAMPLE_RATE, frame_ms=FRAME_MS)
    frames: list[np.ndarray] = []
    max_frames = MAX_UTTERANCE_S * 1000 // FRAME_MS

    console.print("[cyan]listening...[/cyan]")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=FRAME_LEN
    ) as stream:
        announced = False
        for _ in range(max_frames):
            block, _overflow = stream.read(FRAME_LEN)
            frame = block[:, 0].copy()
            frames.append(frame)
            done = vad.accept(frame)
            if vad.triggered and not announced:
                console.print("[red]  speech detected[/red]")
                announced = True
            if done:
                break

    if not vad.triggered:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(frames)


def run_turn(
    base: str, username: str, session_id: str, text: str, speaker, player
) -> None:
    """Stream the reply and speak it sentence by sentence."""
    payload = {"username": username, "message": text, "session_id": session_id}
    buffer = ""
    first = True
    spoke_at: float | None = None
    t0 = time.perf_counter()

    console.print("[bold magenta]coach[/bold magenta] ", end="")
    with httpx.Client(timeout=180) as client:
        with client.stream("POST", f"{base}/chat/stream", json=payload) as resp:
            resp.raise_for_status()
            for delta in resp.iter_text():
                if not delta:
                    continue
                console.print(delta, end="", highlight=False)
                buffer += delta
                sentences, buffer = split_sentences(buffer, is_first=first)
                for sentence in sentences:
                    first = False
                    if speaker is not None:
                        player.play(speaker.synthesize(clean_for_speech(sentence)))
                        if spoke_at is None:
                            spoke_at = time.perf_counter()

    if buffer.strip() and speaker is not None:
        player.play(speaker.synthesize(clean_for_speech(buffer)))
        if spoke_at is None:
            spoke_at = time.perf_counter()

    console.print()
    if spoke_at is not None:
        console.print(f"[dim]  first audio after {spoke_at - t0:.2f}s[/dim]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--user", default="default")
    parser.add_argument("--push", action="store_true", help="push-to-talk mode")
    parser.add_argument("--no-tts", action="store_true", help="transcribe only")
    args = parser.parse_args()

    try:
        httpx.get(f"{args.base}/health", timeout=5).raise_for_status()
    except httpx.HTTPError:
        console.print(
            f"[red]Cannot reach the API at {args.base}.[/red]\n"
            "Start it with: uv run uvicorn app.main:app"
        )
        return 1

    from app.speech.stt import get_transcriber

    console.print("[dim]loading whisper...[/dim]")
    transcriber = get_transcriber()

    speaker = None
    player = None
    if not args.no_tts:
        from app.speech.tts import Speaker

        speaker = Speaker()
        if not speaker.available():
            console.print(
                "[yellow]Piper not found, running without speech output.[/yellow]\n"
                "[dim]See the setup notes at the top of app/speech/tts.py[/dim]"
            )
            speaker = None
        else:
            player = AudioPlayer(speaker.sample_rate)

    import uuid

    session_id = uuid.uuid4().hex
    console.print("[green]Ready. Ctrl+C to quit.[/green]\n")

    try:
        while True:
            audio = record_push_to_talk() if args.push else record_with_vad()
            if audio.size < SAMPLE_RATE // 2:
                continue

            t0 = time.perf_counter()
            text = transcriber.transcribe(audio, SAMPLE_RATE)
            stt_ms = (time.perf_counter() - t0) * 1000
            if not text.strip():
                continue

            console.print(f"[bold cyan]you[/bold cyan] {text} [dim]({stt_ms:.0f}ms)[/dim]")
            run_turn(args.base, args.user, session_id, text, speaker, player)
            if player is not None:
                player.wait()
    except KeyboardInterrupt:
        console.print("\n[dim]bye[/dim]")
    finally:
        if player is not None:
            player.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
