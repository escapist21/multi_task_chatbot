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
# Prefer .env or environment variable; UI can set at runtime.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    # Ensure the key is available to other modules in this process
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    dprint("OPENAI_API_KEY loaded from environment.")
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    dprint("OPENAI_API_KEY not set at startup; waiting for UI input or .env.")
    client = None

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


def set_openai_api_key(new_key: str) -> str:
    """Update the OpenAI API key at runtime and reinitialize the client.

    Returns a short status message suitable for UI display.
    """
    global OPENAI_API_KEY, client
    try:
        key = (new_key or "").strip()
        if not key:
            return "Error: API key cannot be empty."
        os.environ["OPENAI_API_KEY"] = key
        OPENAI_API_KEY = key
        client = OpenAI(api_key=OPENAI_API_KEY)
        dprint("OPENAI_API_KEY updated at runtime.")
        return "API key updated. Session has been reset."
    except Exception as e:
        return f"Error updating API key: {e}"
