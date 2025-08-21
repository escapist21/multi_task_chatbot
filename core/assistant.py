from __future__ import annotations

from typing import Any, Iterator, List, Tuple
import json

from openai.types.beta.assistant_create_params import ToolResources

import config.settings as settings
from config.prompts import SYS_PROMPTS
from core import state
from core.responses_chat import responses_stream_chat
from utils.chat_format import (
    messages_append_user,
    messages_append_assistant,
    ensure_last_assistant_message,
    append_to_last_assistant,
    extract_text_blocks_from_assistant,
    sanitize_messages,
)


def _ensure_assistant_and_thread(task: str, enabled_tools: List[str], history_messages: List[dict], message: str) -> tuple[bool, List[dict]]:
    """Ensure assistant and thread exist. Returns (ok, messages)."""
    # Guard: require OpenAI client
    if settings.client is None:
        msgs = messages_append_user(list(history_messages or []), message)
        msgs = messages_append_assistant(
            msgs,
            "Error: OpenAI API key is not set. Please enter your key under '0. API Key' and click 'Set API Key'.",
        )
        return False, msgs
    # --- 1. Create or Update Assistant ---
    if not state.assistant_id:
        settings.dprint("No assistant found. Creating a new one...")
        instructions = SYS_PROMPTS.get(task, "You are a helpful assistant.")
        cfg = settings.TASK_CONFIG.get(task, {"model": "gpt-4o-mini"})

        assistant_tools: List[dict[str, Any]] = []
        if "Web Search" in enabled_tools:
            # Register a function tool for web search; the model can request it when needed
            assistant_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for up-to-date information and return a brief, source-linked summary.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query to look up on the web.",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Maximum number of results to include (1-10).",
                                    "minimum": 1,
                                    "maximum": 10,
                                    "default": 5,
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

        # Initialize tool_resources as None, with the correct type hint
        tool_resources: ToolResources | None = None
        if "File Search" in enabled_tools:
            assistant_tools.append({"type": "file_search"})
            if state.vector_store_id:
                tool_resources = ToolResources(file_search={"vector_store_ids": [state.vector_store_id]})
            else:
                print("Warning: File Search is enabled, but no files have been uploaded.")

        try:
            assistant = settings.client.beta.assistants.create(
                name="Multi-Task Chatbot",
                instructions=instructions,
                tools=assistant_tools,
                model=cfg["model"],
                tool_resources=tool_resources,
            )
            state.assistant_id = assistant.id
            settings.dprint(
                f"Created new Assistant (ID: {state.assistant_id}) for task '{task}' with tools: {enabled_tools}"
            )
        except Exception as e:
            print(f"Error creating assistant: {e}")
            msgs = messages_append_user(list(history_messages or []), message)
            msgs = messages_append_assistant(msgs, f"Error: Could not create the assistant. {e}")
            return False, msgs

    # --- 2. Create a Thread ---
    if not state.thread_id:
        settings.dprint("No thread found. Creating a new one...")
        try:
            thread = settings.client.beta.threads.create()
            state.thread_id = thread.id
            settings.dprint(f"Created new Thread (ID: {state.thread_id})")
        except Exception as e:
            print(f"Error creating thread: {e}")
            msgs = messages_append_user(list(history_messages or []), message)
            msgs = messages_append_assistant(msgs, f"Error: Could not create the conversation thread. {e}")
            return False, msgs

    return True, history_messages


def chat_fn(message: str, history_messages: List[dict], task: str, enabled_tools: List[str]) -> tuple[str, List[dict]]:
    """Non-streaming chat (Assistants API); returns messages for Gradio Chatbot(type="messages")."""
    ok, history_messages = _ensure_assistant_and_thread(task, enabled_tools, history_messages, message)
    if not ok:
        return "", history_messages

    # --- 3. Add User's Message to the Thread ---
    try:
        settings.client.beta.threads.messages.create(
            thread_id=state.thread_id,
            role="user",
            content=message,
        )
    except Exception as e:
        print(f"Error adding message to thread: {e}")
        msgs = messages_append_user(list(history_messages or []), message)
        msgs = messages_append_assistant(msgs, f"Error: Could not process your message. {e}")
        return "", msgs

    # --- 4. Run the Assistant and Poll for Completion ---
    try:
        settings.dprint(f"Running Assistant {state.assistant_id} on Thread {state.thread_id}...")
        run = settings.client.beta.threads.runs.create(
            thread_id=state.thread_id,
            assistant_id=state.assistant_id,
        )

        # Handle function tool-calls loop
        while True:
            status = getattr(run, "status", None)
            if status == "completed":
                break
            if status == "requires_action":
                try:
                    tool_calls = run.required_action.submit_tool_outputs.tool_calls  # type: ignore[attr-defined]
                except Exception:
                    tool_calls = []

                tool_outputs: List[dict[str, str]] = []
                for call in tool_calls:
                    try:
                        fname = getattr(call, "function", None).name  # type: ignore[union-attr]
                        fargs_json = getattr(call, "function", None).arguments  # type: ignore[union-attr]
                    except Exception:
                        fname = None
                        fargs_json = None
                    if fname == "web_search":
                        try:
                            args = json.loads(fargs_json or "{}")
                        except Exception:
                            args = {}
                        query = args.get("query", "")
                        max_results = args.get("max_results", 5)
                        # Execute Tavily search
                        try:
                            from utils.web_search import tavily_search_summarize

                            output_text = tavily_search_summarize(query=query, max_results=max_results)
                        except Exception as err:
                            output_text = f"Error performing web search: {err}"
                        tool_outputs.append({
                            "tool_call_id": call.id,
                            "output": output_text,
                        })
                # Submit tool outputs and poll until next state
                if tool_outputs:
                    run = settings.client.beta.threads.runs.submit_tool_outputs_and_poll(
                        thread_id=state.thread_id,
                        run_id=run.id,
                        tool_outputs=tool_outputs,
                    )
                    continue
            # Fallback: poll until completion or next action required
            run = settings.client.beta.threads.runs.retrieve(
                thread_id=state.thread_id,
                run_id=run.id,
            )
    except Exception as e:
        print(f"Error during assistant run: {e}")
        msgs = messages_append_user(list(history_messages or []), message)
        msgs = messages_append_assistant(msgs, f"Error: The assistant failed to run. {e}")
        return "", msgs

    # --- 5. Retrieve and Display the Response ---
    if run.status == "completed":
        msgs_list = list(history_messages or [])
        msgs_list = messages_append_user(msgs_list, message)
        try:
            thread_messages = settings.client.beta.threads.messages.list(thread_id=state.thread_id)
            latest = thread_messages.data[0]
            final_reply = extract_text_blocks_from_assistant(latest)
            msgs_list = messages_append_assistant(msgs_list, final_reply)
        except Exception as e:
            print(f"Error fetching final message: {e}")
            msgs_list = messages_append_assistant(msgs_list, "")
        return "", msgs_list
    else:
        settings.dprint(f"Run failed with status: {run.status}")
        error_message = f"Run failed with status: {run.status}. Please try again."
        if getattr(run, "last_error", None):
            error_message += f" Details: {run.last_error.message}"
        msgs_list = list(history_messages or [])
        msgs_list = messages_append_user(msgs_list, message)
        msgs_list = messages_append_assistant(msgs_list, error_message)
        return "", msgs_list


def chat_fn_streaming(
    message: str, history_messages: List[dict], task: str, enabled_tools: List[str]
) -> Iterator[tuple[str, List[dict]]]:
    """Streaming chat function using Assistants API streaming.

    Yields progressive updates to the last assistant message in history.
    """
    ok, history_messages = _ensure_assistant_and_thread(task, enabled_tools, history_messages, message)
    if not ok:
        yield "", history_messages
        return

    # Add the user's message and prime the assistant reply in history
    try:
        settings.client.beta.threads.messages.create(
            thread_id=state.thread_id,
            role="user",
            content=message,
        )
    except Exception as e:
        print(f"Error adding message to thread: {e}")
        msgs = messages_append_user(list(history_messages or []), message)
        msgs = messages_append_assistant(msgs, f"Error: Could not process your message. {e}")
        yield "", msgs
        return

    # Prepare a working copy of history with a placeholder assistant reply
    work_messages = messages_append_user(list(history_messages or []), message)
    work_messages = ensure_last_assistant_message(work_messages)

    # True token-by-token streaming via Assistants API
    try:
        settings.dprint(f"Streaming Assistant {state.assistant_id} on Thread {state.thread_id}...")
        # Emit an immediate placeholder so the UI shows progress
        try:
            # Emit placeholder assistant if empty
            if not (work_messages and work_messages[-1].get("role") == "assistant" and work_messages[-1].get("content")):
                tmp = list(work_messages)
                tmp[-1]["content"] = "..."
                yield "", tmp
                work_messages[-1]["content"] = ""
        except Exception:
            pass

        with settings.client.beta.threads.runs.stream(
            thread_id=state.thread_id, assistant_id=state.assistant_id
        ) as stream:
            for event in stream:
                # Some SDKs expose `event.event` instead of `event.type`
                etype = getattr(event, "type", None) or getattr(event, "event", None)
                try:
                    settings.dprint(f"[assist_stream] event: {etype}")
                    if etype is None:
                        settings.dprint(f"[assist_stream] event class: {event.__class__.__name__}")
                        # Print a shortened repr to avoid flooding
                        er = repr(event)
                        if len(er) > 300:
                            er = er[:300] + "..."
                        settings.dprint(f"[assist_stream] event repr: {er}")
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
                    work_messages = append_to_last_assistant(work_messages, delta_text)
                    yield "", list(work_messages)

            # Final run status
            run = stream.get_final_run()
    except Exception as e:
        print(f"Error during streaming: {e}")
        work_messages = append_to_last_assistant(work_messages, f"Error: The assistant failed to stream. {e}")
        yield "", work_messages
        return

    if run.status == "completed":
        # Ensure final text is complete (in case some tokens weren't emitted as deltas)
        try:
            thread_messages = settings.client.beta.threads.messages.list(thread_id=state.thread_id)
            assistant_message = thread_messages.data[0]
            final_reply = extract_text_blocks_from_assistant(assistant_message)
            # Replace last assistant content with final
            work_messages[-1]["content"] = final_reply
        except Exception as e:
            settings.dprint(f"Error fetching final message after stream: {e}")
        yield "", work_messages
    else:
        err = f"Run failed with status: {run.status}."
        if getattr(run, "last_error", None):
            err += f" Details: {run.last_error.message}"
        work_messages = append_to_last_assistant(work_messages, err)
        yield "", work_messages


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
    # Messages-only model: sanitize incoming history for robustness
    history_messages: List[dict] = sanitize_messages(history)

    if stream:
        # If no tools are enabled, use the simpler Responses API streaming path
        if not enabled_tools:
            for _, out_messages in responses_stream_chat(message, history_messages, task):
                yield "", out_messages
            return
        # If Web Search is enabled, fall back to non-streaming (function tool requires action loop)
        if "Web Search" in enabled_tools:
            _, out_messages = chat_fn(message, history_messages, task, enabled_tools)
            yield "", out_messages
            return
        # Otherwise, use Assistants streaming (supports file_search), messages-based
        for _, out_messages in chat_fn_streaming(message, history_messages, task, enabled_tools):
            yield "", out_messages
        return
    # Non-streaming path
    _, out_messages = chat_fn(message, history_messages, task, enabled_tools)
    return "", out_messages
