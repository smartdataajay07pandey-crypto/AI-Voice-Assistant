import os
import asyncio
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

class AssistantEngine:
    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Initialize the vector store connection
        self.vector_db = Chroma(
            persist_directory="./database/chroma_db",
            embedding_function=OpenAIEmbeddings()
        )

    async def get_context(self, user_query: str):
        """Searches the local database for relevant text chunks."""
        # We run this in a thread to prevent blocking the async loop
        loop = asyncio.get_event_loop()
        docs = await loop.run_in_executor(
            None, 
            self.vector_db.similarity_search, 
            user_query, 
            3 # Top 3 snippets
        )
        return "\n".join([d.page_content for d in docs])

    async def generate_response(self, user_input: str):
        context = await self.get_context(user_input)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, friendly voice assistant. "
                    "Answer naturally and concisely for spoken output."
                )
            },
            {
                "role": "system",
                "content": f"Relevant context:\n{context if context else 'No context found.'}"
            },
            {
                "role": "user",
                "content": user_input
            }
        ]

        stream = await self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content
