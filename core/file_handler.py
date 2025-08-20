from __future__ import annotations

import time
from typing import List

from config.settings import client
from core.state import reset_session
import os


def upload_files(files: List[os.PathLike] | None) -> str:
    """Uploads files to OpenAI and adds them to a vector store."""
    # Use a local import to avoid circular import of state at module import time
    from core import state

    if not files:
        return "No files selected. Please upload at least one file."

    try:
        # Create a vector store
        vs = client.vector_stores.create(name=f"chatbot_store_{int(time.time())}")
        state.vector_store_id = vs.id

        file_paths = [f.name for f in files]
        file_streams = [open(path, "rb") for path in file_paths]

        print(f"Uploading {len(file_paths)} files to new Vector Store: {state.vector_store_id}")

        # Upload and poll
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=state.vector_store_id, files=file_streams
        )

        # Close file streams
        for stream in file_streams:
            stream.close()

        # Reset current assistant to force recreation with new vector store
        reset_session()

        return f"Uploaded {len(files)} files. Vector Store Status: {file_batch.status}"

    except Exception as e:
        print(f"Error during file upload: {e}")
        return f"An error occurred: {e}"
