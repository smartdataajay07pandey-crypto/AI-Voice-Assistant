import os
from dotenv import load_dotenv

def initialize_env():
    """Loads environment variables and checks for API keys."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(" MISSING API KEY: Please set OPENAI_API_KEY in your .env file.")
    return api_key

def ensure_dirs():
    """Ensures necessary project directories exist."""
    directories = ['./data', './database', './temp']
    for d in directories:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")