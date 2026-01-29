import asyncio
import time
import traceback

from src.audio.listener import AsyncRecorder
from src.audio.speaker import AsyncPlayer
from src.brain.engine import AssistantEngine
from src.utils.helpers import initialize_env, ensure_dirs

# ==========================
# DEBUG CONFIG
# ==========================
DEBUG = True

# Whisper hallucination / noise filters
MIN_TEXT_LEN = 3
GARBAGE_INPUTS = {
    "you",
    "you.",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "hello",
    "hi",
}


def log(stage: str, message: str):
    if DEBUG:
        print(f"[DEBUG][{stage}] {message}")


# ==========================
# TTS + AUDIO OUTPUT
# ==========================
async def speak(engine, player, text: str):
    audio_file = "temp/speech.mp3"

    try:
        log("TTS", f"Input text → TTS: {text}")

        async with engine.client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=text,
        ) as response:
            await response.stream_to_file(audio_file)

        log("TTS", f"Audio file saved: {audio_file}")

        await player.play_audio(audio_file)
        log("SPEAKER", "Playback finished")

    except Exception as e:
        log("TTS/SPEAKER", f"ERROR: {e}")
        traceback.print_exc()


# ==========================
# MAIN LOOP
# ==========================
async def main():
    # ----- SYSTEM INIT -----
    log("SYSTEM", "Initializing environment")
    initialize_env()
    ensure_dirs()

    recorder = AsyncRecorder()
    player = AsyncPlayer()
    engine = AssistantEngine()

    log("SYSTEM", "All components initialized")

    # ----- AUDIO SELF TEST -----
    log("HEALTH", "Running TTS + Speaker self-check")
    await speak(engine, player, "System check complete. Audio output is working.")

    print("\n Assistant is online. Speak now...\n")

    try:
        while True:
            # ==========================
            # MIC + STT
            # ==========================
            log("MIC", "Listening...")
            start_time = time.time()

            try:
                user_input = await recorder.listen()
            except Exception as e:
                log("STT", f"Recorder error: {e}")
                traceback.print_exc()
                continue

            if not user_input:
                log("STT", "Empty transcription")
                continue

            clean_input = user_input.strip()

            # ---- STT FILTERING ----
            if len(clean_input) < MIN_TEXT_LEN:
                log("STT", f"Ignored short input: '{clean_input}'")
                continue

            if clean_input.lower() in GARBAGE_INPUTS:
                log("STT", f"Ignored garbage input: '{clean_input}'")
                continue

            log(
                "STT",
                f"Transcription accepted in {time.time() - start_time:.2f}s"
            )
            print(f"\nYou → {clean_input}\n")

            # ==========================
            # LLM STREAMING
            # ==========================
            response_buffer = ""

            try:
                log("LLM", "Sending input to LLM")

                async for chunk in engine.generate_response(clean_input):
                    log("LLM_STREAM", f"Chunk: {chunk}")
                    response_buffer += chunk

                    if response_buffer.endswith((".", "?", "!")):
                        log("PIPELINE", "Sentence complete → TTS")
                        await speak(engine, player, response_buffer)
                        response_buffer = ""

            except Exception as e:
                log("LLM", f"ERROR: {e}")
                traceback.print_exc()
                continue

            # ==========================
            # FINAL TTS FLUSH
            # ==========================
            if response_buffer.strip():
                log("PIPELINE", "Flushing remaining text → TTS")
                await speak(engine, player, response_buffer)

    except KeyboardInterrupt:
        print("\n Assistant shutting down safely...")
    except Exception as e:
        log("SYSTEM", f"FATAL ERROR: {e}")
        traceback.print_exc()


# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    asyncio.run(main())