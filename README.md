# Multi-Task Chatbot with File and Web Search
### Powered by OpenAI Assistants API
### Built with Gradio

## Features
- Configure from a list of pre-selected tasks
	- Text Classification
	- Question Answering
	- Table Question Answering
	- Sentence Similarity
	- Translation
	- Summarisation
	- Chat with Document
- Tools
	- Web Search
	- File Search
- Support for multiple file upload to vectorstore


## Quickstart

1) Prerequisites
   - Python 3.12+
   - An OpenAI API key in environment: `OPENAI_API_KEY`

2) Install
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3) Run
```bash
python main.py
```
The app starts at http://127.0.0.1:7860 and a temporary public Gradio link.


## Architecture

The codebase is modular. Primary modules:

- `config/`
  - `settings.py`: OpenAI client, global settings.
  - `prompts.py`: System prompts per task.
- `core/`
  - `assistant.py`: Assistants API path (tools-enabled). Streaming and non-streaming.
  - `responses_chat.py`: Chat Completions API path (no-tools). Streaming.
  - `state.py`: Session-scoped IDs (assistant, thread, vector store).
- `ui/`
  - `components.py`: Gradio UI wiring. Uses `gr.Chatbot(type="messages")`.
- `utils/`
  - `chat_format.py`: Helpers for messages model (append, stream deltas, sanitize, extract text).


## Messages model (canonical)

Internally and in the UI, chat history is a list of dicts:
```json
[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
```

- UI Chatbot is configured with `type="messages"` and expects this shape.
- All back-end paths accept and return messages. No legacy `(user, assistant)` tuples remain.
- `utils/chat_format.py` provides:
  - `messages_append_user()` / `messages_append_assistant()`
  - `ensure_last_assistant_message()` and `append_to_last_assistant()` for streaming
  - `sanitize_messages()` for robustness against malformed histories
  - `extract_text_blocks_from_assistant()` to flatten Assistants message blocks


## Streaming

Two streaming paths are supported:

- No-tools path: OpenAI Chat Completions API
  - Implemented in `core/responses_chat.py`
  - True token-by-token streaming; yields updated messages as tokens arrive.

- Tools path: OpenAI Assistants API
  - Implemented in `core/assistant.py`
  - Parses stream events (e.g., `thread.message.delta`, `response.output_text.delta`).
  - True token-by-token streaming; final message is reconciled with the thread.

Both paths yield messages lists compatible with Gradio `Chatbot(type="messages")`.


## Tools

- Web Search
  - Enable via the UI toggle. Assistant is created with `{type: "search"}` tool.

- File Search
  - Upload files in the UI. A vector store is created and attached as tool resources.
  - Assistant is created/updated with `{type: "file_search"}` and `vector_store_ids`.
  - After a successful upload, the UI automatically enables File Search and switches the Task to "Chat with Document".
  - Tip: You can manually change the Task at any time using the Task selector in the right panel.

Sessions can be reset from the UI; this recreates assistant and thread and reattaches tools as needed.


## Environment

Required:

- `OPENAI_API_KEY` — your OpenAI key.

Optional:

- `DEBUG` — set to `1`/`true` to enable verbose logs.


## Debugging

Enable verbose, gated logs using the `DEBUG` flag (only debug lines are affected; errors still print):

```bash
# Option A: one-off
export DEBUG=1
python main.py

# Option B: helper script
bash scripts/run_debug.sh
```

Notes:
- `DEBUG` affects both paths (Assistants and Chat Completions).
- Stream traces like `[assist_stream] event:` and `[responses_stream] delta(...)` appear only when DEBUG is enabled.


## Troubleshooting

- Only final message appears during streaming
  - Ensure you’re using the latest code; both paths now stream token-by-token.

- "messages format" warnings in Gradio
  - The UI uses `type="messages"`. If you see issues, confirm inputs/outputs are messages lists. The app sanitizes history via `sanitize_messages()`.

- File Search returns no results
  - Verify files are uploaded and the vector store ID is attached in logs.


## Development notes

- Internal history is messages-only. Avoid reintroducing tuple pairs.
- Keep tool output human-readable; `extract_text_blocks_from_assistant()` flattens Assistants content blocks.
- Streaming updates always mutate the last assistant message in-place to render tokens live.

