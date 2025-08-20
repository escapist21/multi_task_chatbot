from __future__ import annotations

from typing import Iterator, List, Tuple

from config.settings import client, TASK_CONFIG
from config.prompts import SYS_PROMPTS


def responses_stream_chat(
    message: str,
    history: List[Tuple[str, str]],
    task: str,
) -> Iterator[tuple[str, List[Tuple[str, str]]]]:
    """Stream tokens using Chat Completions API (no tools).

    Yields (textbox_value, history) tuples for Gradio.
    """
    # Prepare system instruction and model
    instructions = SYS_PROMPTS.get(task, "You are a helpful assistant.")
    cfg = TASK_CONFIG.get(task, {"model": "gpt-4o-mini"})
    model = cfg.get("model", "gpt-4o-mini")

    # Working copy: add placeholder assistant turn
    work_history = list(history)
    work_history.append((message, ""))
    # Immediate placeholder so UI shows activity
    try:
        yield "", list(work_history)
    except Exception:
        pass

    # Build chat messages from history
    messages: list[dict] = [{"role": "system", "content": instructions}]
    for u, a in history:
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
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
                try:
                    print(f"[responses_stream] delta({len(delta_text)}): {delta_text[:40]!r}")
                except Exception:
                    pass
                user_msg, current_reply = work_history[-1]
                work_history[-1] = (user_msg, current_reply + delta_text)
                yield "", list(work_history)

        # Done: nothing else to fetch; chat.completions stream ends with full text accumulated
        yield "", work_history
    except Exception as e:
        user_msg, _ = work_history[-1]
        work_history[-1] = (user_msg, f"Error: Streaming failed. {e}")
        yield "", work_history
