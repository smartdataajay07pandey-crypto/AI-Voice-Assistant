import asyncio
import sounddevice as sd
import soundfile as sf
import os


class AsyncPlayer:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def play_audio(self, file_path: str):
        """Plays an audio file asynchronously (WAV/MP3)."""
        loop = asyncio.get_event_loop()

        try:
            async with self._lock:
                await loop.run_in_executor(
                    None,
                    self._play_blocking,
                    file_path
                )

        except Exception as e:
            print(f"🔊 Playback error: {e}")

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    def _play_blocking(self, file_path: str):
        data, samplerate = sf.read(file_path, dtype="float32")
        sd.play(data, samplerate)
        sd.wait()
