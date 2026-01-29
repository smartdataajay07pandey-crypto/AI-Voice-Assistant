import asyncio
import sounddevice as sd
import soundfile as sf
import os
import numpy as np


class AsyncPlayer:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def play_audio(self, file_path: str):
        """Play an audio file asynchronously (WAV/MP3)."""
        loop = asyncio.get_running_loop()

        if not os.path.exists(file_path):
            print(f"🔊 Audio file not found: {file_path}")
            return

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
            # Remove temp file only after successful attempt
            try:
                os.remove(file_path)
            except OSError:
                pass

    def _play_blocking(self, file_path: str):
        data, samplerate = sf.read(file_path, dtype="float32")

        # Ensure correct shape
        if data.ndim == 1:
            data = np.expand_dims(data, axis=1)

        # Normalize (safety)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        sd.play(data, samplerate)
        sd.wait()
