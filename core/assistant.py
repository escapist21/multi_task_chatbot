from __future__ import annotations

from typing import Any, Iterator, List, Tuple

from openai.types.beta.assistant_create_params import ToolResources

from config.settings import client, TASK_CONFIG
from config.prompts import SYS_PROMPTS
from core import state
from core.responses_chat import responses_stream_chat


def _messages_to_pairs(messages: List[Any]) -> List[Tuple[str, str]]:
    """Convert Gradio Chatbot messages-format to list of (user, assistant) pairs.

    Expects a list of dicts like {"role": "user"|"assistant", "content": str}.
    Tolerates missing assistant replies by pairing with "".
    """
    pairs: List[Tuple[str, str]] = []
    pending_user: str | None = None
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        # content may be a list (rich); join text if needed
        if isinstance(content, list):
            # try to extract plain text parts
            text_parts: List[str] = []
            for c in content:
                if isinstance(c, dict):
                    val = c.get("text") or c.get("content") or c.get("value")
                    if isinstance(val, str):
                        text_parts.append(val)
            content = "".join(text_parts) if text_parts else ""
        if not isinstance(content, str):
            content = str(content) if content is not None else ""

        if role == "user":
            if pending_user is not None:
                # previous user without assistant, close it with empty assistant
                pairs.append((pending_user, ""))
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                # unexpected assistant-first; treat as empty user
                pairs.append(("", content))
            else:
                pairs.append((pending_user, content))
                pending_user = None
        else:
            # ignore other roles for pairing
            continue
    if pending_user is not None:
        pairs.append((pending_user, ""))
    return pairs


def _pairs_to_messages(pairs: List[Tuple[str, str]]) -> List[dict]:
    """Convert (user, assistant) pairs to Gradio Chatbot messages-format list of dicts."""
    msgs: List[dict] = []
    for u, a in pairs:
        if u is not None:
            msgs.append({"role": "user", "content": u})
        if a is not None and a != "":
            msgs.append({"role": "assistant", "content": a})
    return msgs


def _ensure_assistant_and_thread(task: str, enabled_tools: List[str], history: List[Tuple[str, str]], message: str) -> tuple[bool, List[Tuple[str, str]]]:
    """Ensure assistant and thread exist. Returns (ok, history)."""
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
            return False, history

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
            return False, history

    return True, history


def chat_fn(message: str, history: List[Tuple[str, str]], task: str, enabled_tools: List[str]) -> tuple[str, List[Tuple[str, str]]]:
    """Non-streaming chat function that uses the Assistants API and returns once completed."""
    ok, history = _ensure_assistant_and_thread(task, enabled_tools, history, message)
    if not ok:
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


def chat_fn_streaming(
    message: str, history: List[Tuple[str, str]], task: str, enabled_tools: List[str]
) -> Iterator[tuple[str, List[Tuple[str, str]]]]:
    """Streaming chat function using Assistants API streaming.

    Yields progressive updates to the last assistant message in history.
    """
    ok, history = _ensure_assistant_and_thread(task, enabled_tools, history, message)
    if not ok:
        yield "", history
        return

    # Add the user's message and prime the assistant reply in history
    try:
        client.beta.threads.messages.create(
            thread_id=state.thread_id,
            role="user",
            content=message,
        )
    except Exception as e:
        print(f"Error adding message to thread: {e}")
        history.append((message, f"Error: Could not process your message. {e}"))
        yield "", history
        return

    # Prepare a working copy of history with a placeholder assistant reply
    work_history = list(history)
    work_history.append((message, ""))

    # True token-by-token streaming via Assistants API
    try:
        print(f"Streaming Assistant {state.assistant_id} on Thread {state.thread_id}...")
        # Emit an immediate placeholder so the UI shows progress
        try:
            user_msg, current_reply = work_history[-1]
            if not current_reply:
                work_history[-1] = (user_msg, "...")
                yield "", list(work_history)
                # replace placeholder on first real delta
                work_history[-1] = (user_msg, "")
        except Exception:
            pass

        with client.beta.threads.runs.stream(
            thread_id=state.thread_id, assistant_id=state.assistant_id
        ) as stream:
            for event in stream:
                # Some SDKs expose `event.event` instead of `event.type`
                etype = getattr(event, "type", None) or getattr(event, "event", None)
                try:
                    print(f"[assist_stream] event: {etype}")
                    if etype is None:
                        print(f"[assist_stream] event class: {event.__class__.__name__}")
                        # Print a shortened repr to avoid flooding
                        er = repr(event)
                        if len(er) > 300:
                            er = er[:300] + "..."
                        print(f"[assist_stream] event repr: {er}")
                except Exception:
                    pass

                # Try to extract textual delta from multiple shapes
                delta_text = ""
                # 1) response.output_text.delta (Responses-style)
                if etype == "response.output_text.delta":
                    delta_obj = getattr(event, "delta", None)
                    delta_text = getattr(delta_obj, "value", "") or ""
                # 2) thread.message.delta (Assistants-style)
                elif etype == "thread.message.delta":
                    # Based on repr: ThreadMessageDelta(data=MessageDeltaEvent(..., delta=MessageDelta(content=[TextDeltaBlock(... text=TextDelta(value='...'))])))
                    try:
                        data_obj = getattr(event, "data", None)
                        msg_delta = getattr(data_obj, "delta", None)
                        content_list = getattr(msg_delta, "content", None)
                        if content_list and isinstance(content_list, (list, tuple)):
                            parts = []
                            for block in content_list:
                                try:
                                    if getattr(block, "type", None) == "text":
                                        text_obj = getattr(block, "text", None)
                                        val = getattr(text_obj, "value", None)
                                        if isinstance(val, str) and val:
                                            parts.append(val)
                                except Exception:
                                    continue
                            delta_text = "".join(parts)
                        else:
                            delta_text = ""
                    except Exception:
                        delta_text = ""
                # 3) Generic response.delta/message.delta fallbacks
                elif etype in ("response.delta", "message.delta", "run.step.delta"):
                    delta_obj = getattr(event, "delta", None)
                    delta_text = (
                        getattr(delta_obj, "value", None)
                        or getattr(delta_obj, "text", None)
                        or (delta_obj if isinstance(delta_obj, str) else "")
                    ) or ""

                # 4) Fallback: dict-like payloads with nested text
                elif etype is None:
                    try:
                        # If event behaves like a dict, try common shapes
                        if isinstance(event, dict):
                            # e.g., {"delta": {"value": "..."}}
                            d = event.get("delta") or {}
                            delta_text = d.get("value") or d.get("text") or ""
                            if not delta_text:
                                data = event.get("data") or {}
                                # Try nested content blocks
                                content = data.get("content") or []
                                for block in content:
                                    val = (
                                        block.get("text", {}).get("value")
                                        if isinstance(block.get("text"), dict)
                                        else block.get("value")
                                    )
                                    if isinstance(val, str) and val:
                                        delta_text = val
                                        break
                    except Exception:
                        pass

                if delta_text:
                    user_msg, current_reply = work_history[-1]
                    work_history[-1] = (user_msg, current_reply + delta_text)
                    yield "", list(work_history)

            # Final run status
            run = stream.get_final_run()
    except Exception as e:
        print(f"Error during streaming: {e}")
        work_history[-1] = (work_history[-1][0], f"Error: The assistant failed to stream. {e}")
        yield "", work_history
        return

    if run.status == "completed":
        # Ensure final text is complete (in case some tokens weren't emitted as deltas)
        try:
            messages = client.beta.threads.messages.list(thread_id=state.thread_id)
            assistant_message = messages.data[0]
            reply_parts: List[str] = []
            for content_block in assistant_message.content:
                if content_block.type == "text":
                    reply_parts.append(content_block.text.value)
            final_reply = "".join(reply_parts)
            work_history[-1] = (work_history[-1][0], final_reply)
        except Exception as e:
            print(f"Error fetching final message after stream: {e}")
        yield "", work_history
    else:
        err = f"Run failed with status: {run.status}."
        if getattr(run, "last_error", None):
            err += f" Details: {run.last_error.message}"
        work_history[-1] = (work_history[-1][0], err)
        yield "", work_history


def chat_entry(
    message: str,
    history: List[Any],
    task: str,
    enabled_tools: List[str],
    stream: bool,
):
    """Entry point used by the UI. If stream=True, yields streaming updates.

    Note: For Gradio streaming, this function itself must be a generator that
    yields output tuples matching the outputs spec. Returning a generator object
    (instead of yielding) causes a ValueError about output arity.
    """
    # Normalize incoming history: detect messages-format dicts
    # Treat empty history as messages-format because UI Chatbot uses type="messages".
    history_is_messages = (not history) or isinstance(history[0], dict)
    history_pairs: List[Tuple[str, str]] = _messages_to_pairs(history) if history_is_messages else list(history or [])

    if stream:
        # If no tools are enabled, use the simpler Responses API streaming path
        if not enabled_tools:
            for _, out_pairs in responses_stream_chat(message, history_pairs, task):
                # Always return messages for Chatbot(type="messages")
                out_msgs = _pairs_to_messages(out_pairs)
                yield "", out_msgs
            return
        # Otherwise, use Assistants streaming (supports tools)
        for _, out_pairs in chat_fn_streaming(message, history_pairs, task, enabled_tools):
            out_msgs = _pairs_to_messages(out_pairs)
            yield "", out_msgs
        return
    # Non-streaming path
    _, out_pairs = chat_fn(message, history_pairs, task, enabled_tools)
    out_msgs = _pairs_to_messages(out_pairs)
    return "", out_msgs
