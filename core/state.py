# Shared application state and session management

# Global variables to hold state across Gradio interactions
vector_store_id: str | None = None
assistant_id: str | None = None
thread_id: str | None = None


def reset_session() -> str:
    """Resets the assistant and thread, forcing recreation on the next message."""
    global assistant_id, thread_id
    assistant_id = None
    thread_id = None
    print("Session reset. New assistant and thread will be created.")
    return "Session has been reset."
