from __future__ import annotations

from typing import Iterator, List, Tuple, Dict, Any

import config.settings as settings
from config.prompts import SYS_PROMPTS
from utils.chat_format import (
    messages_to_openai,
    messages_append_user,
    ensure_last_assistant_message,
    append_to_last_assistant,
)


def responses_stream_chat(
    message: str,
    history_messages: List[Dict[str, Any]],
    task: str,
) -> Iterator[tuple[str, List[Dict[str, Any]]]]:
    """Stream tokens using Chat Completions API (no tools), messages-based.

    Returns yields suitable for Gradio Chatbot(type="messages"): ("", messages_list)
    """
    # Guard: require OpenAI client
    if settings.client is None:
        work_messages: List[Dict[str, Any]] = messages_append_user(list(history_messages or []), message)
        work_messages = ensure_last_assistant_message(work_messages)
        work_messages[-1]["content"] = (
            "Error: OpenAI API key is not set. Please enter your key under '0. API Key' and click 'Set API Key'."
        )
        yield "", work_messages
        return
    # Prepare system instruction and model
    instructions = SYS_PROMPTS.get(task, "You are a helpful assistant.")
    cfg = settings.TASK_CONFIG.get(task, {"model": "gpt-4o-mini"})
    model = cfg.get("model", "gpt-4o-mini")

    # Working copy: add user message and a placeholder assistant
    work_messages: List[Dict[str, Any]] = messages_append_user(list(history_messages or []), message)
    work_messages = ensure_last_assistant_message(work_messages)
    # Immediate placeholder so UI shows activity
    try:
        yield "", list(work_messages)
    except Exception:
        pass

    # Build OpenAI chat payload
    oa_messages = messages_to_openai(work_messages, system_instruction=instructions)
    try:
        stream = settings.client.chat.completions.create(
            model=model,
            # Exclude the placeholder assistant for API call; last element is the assistant placeholder
            messages=oa_messages[:-1],
            stream=True,
        )
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta
                delta_text = getattr(delta, "content", None) or ""
            except Exception:
                delta_text = ""
            if delta_text:
                # Debug: log small snippet of delta
                settings.dprint(f"[responses_stream] delta({len(delta_text)}): {delta_text[:40]!r}")
                work_messages = append_to_last_assistant(work_messages, delta_text)
                yield "", list(work_messages)

        # Done: nothing else to fetch; accumulated content is in last assistant message
        yield "", work_messages
    except Exception as e:
        work_messages = append_to_last_assistant(work_messages, f"Error: Streaming failed. {e}")
        yield "", work_messages
