import asyncio
from src.audio.listener import AsyncRecorder
from src.audio.speaker import AsyncPlayer
from src.brain.engine import AssistantEngine
from src.utils.helpers import initialize_env, ensure_dirs


async def speak(engine, player, text: str):
    """Convert text to speech and play it asynchronously."""
    audio_file = "temp/speech.mp3"

    async with engine.client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text,
    ) as response:
        await response.stream_to_file(audio_file)

    await player.play_audio(audio_file)


async def main():
    # 1. Setup
    initialize_env()
    ensure_dirs()

    recorder = AsyncRecorder()
    player = AsyncPlayer()
    engine = AssistantEngine()

    print(" Assistant is online. How can I help?")

    try:
        while True:
            # 2. Listen (non-blocking)
            user_input = await recorder.listen()

            if not user_input:
                continue

            print(f"You: {user_input}")

            # 3. Stream LLM response + sentence-level TTS
            response_buffer = ""

            async for chunk in engine.generate_response(user_input):
                response_buffer += chunk

                # Speak once we reach a natural pause
                if response_buffer.endswith((".", "?", "!")):
                    await speak(engine, player, response_buffer)
                    response_buffer = ""

            # Speak any remaining text
            if response_buffer.strip():
                await speak(engine, player, response_buffer)

    except KeyboardInterrupt:
        print("\n Assistant shutting down safely...")


if __name__ == "__main__":
    asyncio.run(main())
