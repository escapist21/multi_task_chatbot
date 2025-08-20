from __future__ import annotations

from typing import List, Dict, Any

ChatMessage = Dict[str, Any]


def messages_append_user(messages: List[ChatMessage], text: str) -> List[ChatMessage]:
    msgs = list(messages or [])
    msgs.append({"role": "user", "content": text or ""})
    return msgs


def messages_append_assistant(messages: List[ChatMessage], text: str) -> List[ChatMessage]:
    msgs = list(messages or [])
    msgs.append({"role": "assistant", "content": text or ""})
    return msgs


def messages_to_openai(messages: List[ChatMessage], system_instruction: str | None = None) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if system_instruction:
        out.append({"role": "system", "content": system_instruction})
    for m in messages or []:
        role = m.get("role")
        content = m.get("content")
        # Flatten content to string
        if isinstance(content, list):
            parts: List[str] = []
            for c in content:
                if isinstance(c, dict):
                    val = c.get("text") or c.get("content") or c.get("value")
                    if isinstance(val, str):
                        parts.append(val)
            content = "".join(parts)
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        out.append({"role": role, "content": content})
    return out


def ensure_last_assistant_message(messages: List[ChatMessage]) -> List[ChatMessage]:
    msgs = list(messages or [])
    if not msgs or msgs[-1].get("role") != "assistant":
        msgs.append({"role": "assistant", "content": ""})
    return msgs


def append_to_last_assistant(messages: List[ChatMessage], delta_text: str) -> List[ChatMessage]:
    msgs = ensure_last_assistant_message(messages)
    try:
        msgs[-1]["content"] = (msgs[-1].get("content") or "") + (delta_text or "")
    except Exception:
        # Fallback: replace content as string
        prev = msgs[-1].get("content")
        if not isinstance(prev, str):
            prev = str(prev) if prev is not None else ""
        msgs[-1]["content"] = prev + (delta_text or "")
    return msgs


def extract_text_blocks_from_assistant(openai_message: Any) -> str:
    # For Assistants API message objects: concatenate text blocks
    try:
        reply_parts: List[str] = []
        for content_block in openai_message.content:
            if getattr(content_block, "type", None) == "text":
                text_obj = getattr(content_block, "text", None)
                val = getattr(text_obj, "value", None)
                if isinstance(val, str):
                    reply_parts.append(val)
        return "".join(reply_parts)
    except Exception:
        return ""


def is_messages_list(value: Any) -> bool:
    """Quickly validate a messages list: each item is a dict with role/content strings."""
    if not isinstance(value, list):
        return False
    for m in value:
        if not isinstance(m, dict):
            return False
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant", "system"):
            return False
        if not (isinstance(content, str) or isinstance(content, list) or content is None):
            return False
    return True


def sanitize_messages(messages: Any) -> List[ChatMessage]:
    """Coerce possibly-messy Chatbot history into [{role, content}] messages.

    - Converts tuples or other types into valid dicts when possible.
    - Flattens content lists to strings.
    - Filters invalid entries.
    """
    out: List[ChatMessage] = []
    if not isinstance(messages, list):
        return out
    for item in messages:
        role: str | None = None
        content: Any = None
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            # best-effort: assume (user, assistant) pair-style; map into two messages
            u, a = item
            if isinstance(u, str) and u:
                out.append({"role": "user", "content": u})
            if isinstance(a, str) and a:
                out.append({"role": "assistant", "content": a})
            continue
        else:
            # skip
            continue

        if role not in ("user", "assistant", "system"):
            continue
        # flatten content
        if isinstance(content, list):
            parts: List[str] = []
            for c in content:
                if isinstance(c, dict):
                    val = c.get("text") or c.get("content") or c.get("value")
                    if isinstance(val, str):
                        parts.append(val)
            content = "".join(parts)
        if not isinstance(content, str):
            content = str(content) if content is not None else ""
        out.append({"role": role, "content": content})
    return out
