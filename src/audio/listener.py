"""
AudioListener: Mikrofon → VAD → faster-whisper → Text
"""

from typing import Optional

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512          # Silero VAD erwartet genau 512 Samples bei 16kHz
CHUNK_DURATION = CHUNK_SAMPLES / SAMPLE_RATE  # ~0.032 Sekunden pro Chunk
SILENCE_THRESHOLD = 1.5     # Sekunden Stille → Ende der Aufnahme
MIN_SPEECH_DURATION = 0.5   # Mindest-Sprachzeit damit es verarbeitet wird


class AudioListener:
    def __init__(self, model_size: str = "medium", device: str = "cpu",
                 beam_size: int = 5, vad_filter: bool = True):
        self.beam_size = beam_size
        self.vad_filter = vad_filter

        print("⏳ Lade Whisper-Modell...")
        self.whisper = WhisperModel(
            model_size,
            device=device,
            compute_type="int8"   # Schneller auf CPU
        )

        # Silero VAD
        print("⏳ Lade VAD-Modell...")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )
        self.get_speech_timestamps = utils[0]
        print("✅ Audio-System bereit")

    async def listen(self) -> Optional[str]:
        """
        Nimmt Audio auf bis Stille erkannt wird.
        Gibt transkribierten Text zurück.
        """
        audio_chunks = []
        silence_counter = 0
        speaking = False
        chunk_samples = CHUNK_SAMPLES

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
            while True:
                chunk, _ = stream.read(chunk_samples)
                chunk = chunk.flatten()
                audio_chunks.append(chunk)

                # VAD Check
                tensor = torch.from_numpy(chunk)
                speech_prob = self.vad_model(tensor, SAMPLE_RATE).item()

                if speech_prob > 0.5:
                    speaking = True
                    silence_counter = 0
                    print(".", end="", flush=True)
                elif speaking:
                    silence_counter += CHUNK_DURATION
                    if silence_counter >= SILENCE_THRESHOLD:
                        print()  # Neue Zeile nach den Punkten
                        break

        if not speaking:
            return None

        # Alle Chunks zusammenfügen
        full_audio = np.concatenate(audio_chunks)

        # Mindestlänge prüfen
        if len(full_audio) / SAMPLE_RATE < MIN_SPEECH_DURATION:
            return None

        return self._transcribe(full_audio)

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Transkribiert Audio mit faster-whisper"""
        kwargs = dict(
            language="de",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
        )
        if self.vad_filter:
            kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)

        segments, info = self.whisper.transcribe(audio, **kwargs)

        text = " ".join(seg.text.strip() for seg in segments)
        if text:
            print(f"🗣️  Erkannt: {text}")
        return text if text else None
