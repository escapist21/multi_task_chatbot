import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from a .env file
load_dotenv()

# Debug flag (set DEBUG=1 to enable verbose logs)
DEBUG = os.environ.get("DEBUG", "0").lower() in ("1", "true", "yes", "on")

def dprint(msg: str) -> None:
    if DEBUG:
        try:
            print(msg)
        except Exception:
            pass

# Initialise OpenAI client
# Make sure your OPENAI_API_KEY is set in your .env file or environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# Define task configurations with model, temperature, etc.
TASK_CONFIG = {
    "Generic Assistant": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Chat with Document": {"model": "gpt-4.1", "temperature": 0.0},
    "Summarisation": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Translation": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Text Classification": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Table Question Answering": {"model": "gpt-4.1", "temperature": 0.0},
    "Sentence Similarity": {"model": "gpt-4.1-mini", "temperature": 0.0},
}
