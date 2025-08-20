from __future__ import annotations

from typing import Any, List, Tuple

from openai.types.beta.assistant_create_params import ToolResources

from config.settings import client, TASK_CONFIG
from config.prompts import SYS_PROMPTS
from core import state


def chat_fn(message: str, history: List[Tuple[str, str]], task: str, enabled_tools: List[str]) -> tuple[str, List[Tuple[str, str]]]:
    """Main chat function that uses the Assistants API."""
    # --- 1. Create or Update Assistant ---
    if not state.assistant_id:
        print("No assistant found. Creating a new one...")
        instructions = SYS_PROMPTS.get(task, "You are a helpful assistant.")
        cfg = TASK_CONFIG.get(task, {"model": "gpt-4o-mini"})

        assistant_tools: List[dict[str, Any]] = []
        if "Web Search" in enabled_tools:
            assistant_tools.append({"type": "search"})

        # Initialize tool_resources as None, with the correct type hint
        tool_resources: ToolResources | None = None
        if "File Search" in enabled_tools:
            assistant_tools.append({"type": "file_search"})
            if state.vector_store_id:
                tool_resources = ToolResources(file_search={"vector_store_ids": [state.vector_store_id]})
            else:
                print("Warning: File Search is enabled, but no files have been uploaded.")

        try:
            assistant = client.beta.assistants.create(
                name="Multi-Task Chatbot",
                instructions=instructions,
                tools=assistant_tools,
                model=cfg["model"],
                tool_resources=tool_resources,
            )
            state.assistant_id = assistant.id
            print(
                f"Created new Assistant (ID: {state.assistant_id}) for task '{task}' with tools: {enabled_tools}"
            )
        except Exception as e:
            print(f"Error creating assistant: {e}")
            history.append((message, f"Error: Could not create the assistant. {e}"))
            return "", history

    # --- 2. Create a Thread ---
    if not state.thread_id:
        print("No thread found. Creating a new one...")
        try:
            thread = client.beta.threads.create()
            state.thread_id = thread.id
            print(f"Created new Thread (ID: {state.thread_id})")
        except Exception as e:
            print(f"Error creating thread: {e}")
            history.append((message, f"Error: Could not create the conversation thread. {e}"))
            return "", history

    # --- 3. Add User's Message to the Thread ---
    try:
        client.beta.threads.messages.create(
            thread_id=state.thread_id,
            role="user",
            content=message,
        )
    except Exception as e:
        print(f"Error adding message to thread: {e}")
        history.append((message, f"Error: Could not process your message. {e}"))
        return "", history

    # --- 4. Run the Assistant and Poll for Completion ---
    try:
        print(f"Running Assistant {state.assistant_id} on Thread {state.thread_id}...")
        run = client.beta.threads.runs.create_and_poll(
            thread_id=state.thread_id,
            assistant_id=state.assistant_id,
        )
    except Exception as e:
        print(f"Error during assistant run: {e}")
        history.append((message, f"Error: The assistant failed to run. {e}"))
        return "", history

    # --- 5. Retrieve and Display the Response ---
    if run.status == "completed":
        messages = client.beta.threads.messages.list(thread_id=state.thread_id)
        assistant_message = messages.data[0]

        reply_parts: List[str] = []
        for content_block in assistant_message.content:
            if content_block.type == "text":
                reply_parts.append(content_block.text.value)

        reply = "".join(reply_parts)

        history.append((message, reply))
        return "", history
    else:
        print(f"Run failed with status: {run.status}")
        error_message = f"Run failed with status: {run.status}. Please try again."
        if getattr(run, "last_error", None):
            error_message += f" Details: {run.last_error.message}"
        history.append((message, error_message))
        return "", history
