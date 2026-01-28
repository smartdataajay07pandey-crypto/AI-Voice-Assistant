import asyncio
import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import queue
import sys

from faster_whisper import WhisperModel

# ==========================
# CONFIG
# ==========================
SAMPLE_RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5
WHISPER_MODEL = "small"
TEMP_DIR = "./temp"

# 🔧 Silence threshold (tuned for most laptop mics)
SILENCE_THRESHOLD = 0.003

os.makedirs(TEMP_DIR, exist_ok=True)

# ==========================
# SILENCE DETECTION (DEBUG)
# ==========================
def is_silent(audio: np.ndarray) -> bool:
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"🔎 Audio RMS: {rms:.6f}")
    return rms < SILENCE_THRESHOLD


# ==========================
# AUDIO RECORDER
# ==========================
class Recorder:
    def __init__(self):
        self.q = queue.Queue()

    def _callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

    def record(self, seconds=5):
        print("🎙️  Recording... speak now")

        frames = []
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            callback=self._callback,
        ):
            for _ in range(int(SAMPLE_RATE / 1024 * seconds)):
                frames.append(self.q.get())

        audio = np.concatenate(frames, axis=0)

        if is_silent(audio):
            print("🔇 Silence detected — skipping\n")
            return None

        path = os.path.join(TEMP_DIR, "input.wav")
        sf.write(path, audio, SAMPLE_RATE)
        print("🎙️  Recording saved")
        return path


# ==========================
# SPEAKER
# ==========================
class Speaker:
    def speak(self, wav_path: str):
        data, sr = sf.read(wav_path, dtype="float32")
        sd.play(data, sr)
        sd.wait()


# ==========================
# SIMPLE LOCAL TTS (WAV)
# ==========================
def simple_tts(text: str) -> str:
    print(f"🤖 Assistant says: {text}")

    duration = min(3 + len(text) * 0.03, 8)
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
    audio = 0.2 * np.sin(2 * np.pi * 220 * t)

    path = os.path.join(TEMP_DIR, "speech.wav")
    sf.write(path, audio, SAMPLE_RATE)
    return path


# ==========================
# MOCK LLM
# ==========================
def llm_reply(user_text: str) -> str:
    user_text = user_text.lower()

    if "your name" in user_text:
        return "I am your local voice assistant."
    if "time" in user_text:
        import datetime
        return f"The current time is {datetime.datetime.now().strftime('%H:%M')}."
    return f"You said: {user_text}"


# ==========================
# MAIN LOOP
# ==========================
async def main():
    print("\n🚀 Voice Assistant Demo Started")
    print("Speak clearly. Say 'exit' to quit.\n")

    recorder = Recorder()
    speaker = Speaker()

    print("⏳ Loading Whisper model...")
    whisper = WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type="int8"
    )
    print("✅ Whisper loaded\n")

    while True:
        wav_path = recorder.record(RECORD_SECONDS)
        if wav_path is None:
            continue

        segments, info = whisper.transcribe(wav_path)

        if info.no_speech_prob > 0.6:
            print("🤫 Whisper detected no speech\n")
            continue

        spoken_text = " ".join(seg.text.strip() for seg in segments).strip()

        if len(spoken_text) < 3:
            print("⚠️ Ignoring short / hallucinated output\n")
            continue

        print(f"🗣️  YOU SAID: \"{spoken_text}\"")

        if spoken_text.lower() in {"exit", "quit", "stop"}:
            print("👋 Exit command detected")
            break

        response = llm_reply(spoken_text)
        tts_path = simple_tts(response)
        speaker.speak(tts_path)

        print()

    print("🛑 Assistant stopped")


# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
