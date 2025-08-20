import os
from getpass import getpass
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
    dprint("OPENAI_API_KEY not found in environment; prompting via getpass.")
    try:
        OPENAI_API_KEY = getpass("Enter your OPENAI_API_KEY (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        OPENAI_API_KEY = None
    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is required. Set it in your .env, environment, or enter when prompted."
        )

# Ensure the key is available to other modules in this process
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
dprint("OPENAI_API_KEY set in process environment.")

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
