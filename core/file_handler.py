from __future__ import annotations

import time
from typing import List, Tuple, Any

import gradio as gr

from config.settings import client
from core.state import reset_session
import os
from pathlib import Path


def upload_files(files: List[os.PathLike | str] | None) -> Tuple[str, Any, Any]:
    """Uploads files to OpenAI and adds them to a vector store.

    Returns a tuple for Gradio outputs:
    (upload_status_text, tools_checkbox_value, task_radio_value)

    On success: enables File Search and switches task to Document Question Answering.
    On no files/error: leaves tool/task unchanged using gr.update().
    """
    # Use a local import to avoid circular import of state at module import time
    from core import state

    if not files:
        return (
            "No files selected. Please upload at least one file.",
            gr.update(),
            gr.update(),
        )

    try:
        # Guard: OpenAI client must be initialized
        if client is None:
            return (
                "Error: OpenAI API key is not set. Please enter your key under '0. API Key' and click 'Set API Key'.",
                gr.update(),
                gr.update(),
            )
        # Create a vector store
        vs = client.vector_stores.create(name=f"chatbot_store_{int(time.time())}")
        state.vector_store_id = vs.id

        # Normalize incoming paths from Gradio (may be str or PathLike objects)
        paths: List[Path] = []
        for f in files:
            try:
                if isinstance(f, (str, bytes, os.PathLike)):
                    p = Path(f)
                else:
                    name_attr = getattr(f, "name", None)
                    p = Path(name_attr) if isinstance(name_attr, str) else None
                if p is not None:
                    paths.append(p)
            except Exception:
                continue

        if not paths:
            return (
                "No valid file paths were provided.",
                gr.update(),
                gr.update(),
            )

        # Open files and ensure they are always closed
        file_streams = []
        try:
            for p in paths:
                file_streams.append(open(str(p), "rb"))

            print(
                f"Uploading {len(paths)} files to new Vector Store: {state.vector_store_id}"
            )

            # Upload and poll
            file_batch = client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=state.vector_store_id, files=file_streams
            )
        finally:
            for stream in file_streams:
                try:
                    stream.close()
                except Exception:
                    pass

        # Reset current assistant to force recreation with new vector store
        reset_session()

        return (
            f"Uploaded {len(paths)} files. Vector Store Status: {file_batch.status}",
            ["File Search"],
            "Chat with Document",
        )

    except Exception as e:
        print(f"Error during file upload: {e}")
        return (
            f"An error occurred: {e}",
            gr.update(),
            gr.update(),
        )
