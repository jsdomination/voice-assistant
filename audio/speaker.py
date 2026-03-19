"""
Speaker: Text → Piper TTS → Lautsprecher
Streaming-Ansatz: Satz für Satz vorlesen für niedrige Latenz
"""

import asyncio
import re
import tempfile
import threading
import wave
from pathlib import Path

import sounddevice as sd
import soundfile as sf
import torch
from piper import PiperVoice

# Piper Modell-Pfad (deutsch) - nach Installation anpassen
PIPER_MODEL = Path("models/de_DE-thorsten-medium.onnx")

VAD_SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512


class Speaker:
    def __init__(self):
        if not PIPER_MODEL.exists():
            print(f"⚠️  Piper-Modell nicht gefunden: {PIPER_MODEL}")
            print("   Download: https://huggingface.co/rhasspy/piper-voices")
            self._voice = None
        else:
            self._voice = PiperVoice.load(str(PIPER_MODEL))
            print("✅ TTS-System bereit")

        # Separate VAD model for barge-in detection
        self._vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self._interrupted = False

    async def speak(self, text: str):
        """
        Text vorlesen. Bricht ab wenn der Nutzer spricht.
        """
        self._interrupted = False
        sentences = self._split_sentences(text)

        for sentence in sentences:
            if sentence.strip():
                if self._interrupted:
                    print("⏹️  TTS abgebrochen (Nutzer spricht)")
                    break
                await self._speak_sentence(sentence.strip())

        if self._interrupted:
            print("⏹️  TTS abgebrochen (Nutzer spricht)")

    async def _speak_sentence(self, text: str):
        """Einzelnen Satz mit Piper synthetisieren und abspielen"""
        if self._voice is None:
            print(f"[TTS Fallback] {text}")
            return

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self._synthesize_to_file, text, tmp_path
            )

            if self._interrupted:
                return

            data, samplerate = sf.read(tmp_path)

            # Start VAD monitoring in background thread
            stop_vad = threading.Event()
            vad_thread = threading.Thread(
                target=self._monitor_speech, args=(stop_vad,), daemon=True
            )
            vad_thread.start()

            try:
                sd.play(data, samplerate)
                # Poll so we can react to interruption quickly
                duration = len(data) / samplerate
                elapsed = 0.0
                poll_interval = 0.05
                while elapsed < duration and sd.get_stream().active:
                    await asyncio.sleep(poll_interval)
                    elapsed += poll_interval
                    if self._interrupted:
                        sd.stop()
                        return
                sd.wait()
            finally:
                stop_vad.set()
                vad_thread.join(timeout=0.5)

        except Exception:
            print(f"[TTS Fallback] {text}")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _monitor_speech(self, stop_event: threading.Event):
        """Background VAD: sets _interrupted when user speaks during playback."""
        try:
            with sd.InputStream(
                samplerate=VAD_SAMPLE_RATE, channels=1, dtype="float32",
            ) as stream:
                consecutive_speech = 0
                while not stop_event.is_set():
                    chunk, _ = stream.read(VAD_CHUNK_SAMPLES)
                    chunk = chunk.flatten()
                    tensor = torch.from_numpy(chunk)
                    prob = self._vad_model(tensor, VAD_SAMPLE_RATE).item()
                    if prob > 0.5:
                        consecutive_speech += 1
                        # Require ~3 consecutive speech chunks (~96ms) to avoid false positives
                        if consecutive_speech >= 3:
                            self._interrupted = True
                            return
                    else:
                        consecutive_speech = 0
        except Exception:
            pass  # Don't crash if mic unavailable during TTS

    def _synthesize_to_file(self, text: str, path: str):
        """Blocking call to Piper — intended for run_in_executor."""
        with wave.open(path, "wb") as wf:
            self._voice.synthesize_wav(text, wf)

    def _split_sentences(self, text: str) -> list[str]:
        """Text in Sätze aufteilen für Streaming-TTS"""
        return re.split(r"(?<=[.!?])\s+", text)
