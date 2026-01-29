# 🎙️ Async Voice Assistant (Python)

A fully asynchronous, local-first **AI Voice Assistant** built with Python.  
It supports **speech-to-text (STT)**, **retrieval-augmented generation (RAG)**, **streaming LLM responses**, and **text-to-speech (TTS)** with sentence-level playback.

This project is designed with **clean architecture**, **async safety**, and **modular components**, making it easy to extend with memory, barge-in, and custom skills.

---

## ✨ Features

- 🎧 **Speech-to-Text (STT)** using `faster-whisper`
- 🧠 **LLM Brain** with OpenAI (streaming responses)
- 📚 **RAG (Retrieval-Augmented Generation)** using ChromaDB
- 🔊 **Text-to-Speech (TTS)** using OpenAI TTS (`tts-1`)
- ⚡ **Fully Async Architecture**
- 🧩 Clean separation of Listener / Speaker / Engine
- 🗂️ Local vector database for contextual grounding
- 🔁 Continuous listening loop
- 🛑 Graceful shutdown with `Ctrl + C`

---

## 🏗️ Project Architecture

## 🧱 System Architecture

```mermaid
flowchart TD
    Mic[🎤 Microphone] --> Listener[AsyncRecorder<br/>Whisper STT]

    Listener -->|Text| MainLoop[Main Event Loop<br/>asyncio]

    MainLoop -->|User Query| Engine[AssistantEngine]

    Engine -->|Similarity Search| VectorDB[(ChromaDB)]
    VectorDB -->|Context| Engine

    Engine -->|Token Stream| MainLoop

    MainLoop -->|Sentence Buffer| TTS[OpenAI TTS<br/>tts-1]

    TTS -->|Audio File| Speaker[AsyncPlayer]

    Speaker -->|Playback| Output[🔊 Speaker]
