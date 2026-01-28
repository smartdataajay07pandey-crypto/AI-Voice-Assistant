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
        self.duration = 5  # seconds

        print("🎙️ Whisper model loaded. Ready to listen.")

    async def listen(self) -> str:
        loop = asyncio.get_event_loop()

        try:
            print("🎤 Listening...")

            audio = await loop.run_in_executor(
                None,
                self._record_audio
            )

            print("🔍 Transcribing...")

            segments, info = await loop.run_in_executor(
                None,
                lambda: self.model.transcribe(
                    audio,
                    task="transcribe",
                    language="en",
                    beam_size=5
                )
            )

            text = " ".join(segment.text for segment in segments).strip()
            return text

        except Exception as e:
            print(f"Whisper error: {e}")
            return ""

    def _record_audio(self):
        recording = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        return np.squeeze(recording)
