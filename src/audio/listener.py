import asyncio
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel


class AsyncRecorder:
    def __init__(self):
        self.model = WhisperModel(
            model_size_or_path="small",
            device="cpu",
            compute_type="int8"
        )

        self.sample_rate = 16000
        self.max_duration = 6  # seconds
        self.silence_threshold = 0.01

        print("🎙️ Whisper model loaded. Ready to listen.")

    async def listen(self) -> str:
        loop = asyncio.get_running_loop()

        try:
            print("🎤 Listening...")

            audio = await loop.run_in_executor(
                None,
                self._record_audio
            )

            if audio is None or len(audio) == 0:
                print("⚠️ No audio captured.")
                return ""

            print("🔍 Transcribing...")

            segments, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    audio,
                    language="en",
                    beam_size=5
                )
            )

            text = " ".join(seg.text for seg in segments).strip()

            if not text:
                print("⚠️ No speech detected.")
                return ""

            print(f"📝 Heard: {text}")
            return text

        except Exception as e:
            print(f"❌ Whisper error: {e}")
            return ""

    def _record_audio(self):
        try:
            recording = sd.rec(
                int(self.max_duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32"
            )
            sd.wait()

            audio = np.squeeze(recording)

            # Normalize
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

            # Silence detection
            if np.mean(np.abs(audio)) < self.silence_threshold:
                return None

            return audio

        except Exception as e:
            print(f"❌ Microphone error: {e}")
            return None
