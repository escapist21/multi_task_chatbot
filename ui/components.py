from __future__ import annotations

import gradio as gr

from config.settings import TASK_CONFIG
from core.assistant import chat_fn
from core.file_handler import upload_files
from core.state import reset_session


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks() as demo:
        gr.Markdown("# Multi-Task Chatbot with Web & File Search")
        gr.Markdown("Powered by OpenAI Assistants API")

        with gr.Row():
            # --- Left Column (Chat Interface) ---
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=600)
                user_input = gr.Textbox(placeholder="Type your message...", lines=1, label="", scale=10)

            # --- Right Column (Controls) ---
            with gr.Column(scale=1):
                gr.Markdown("### 1. Configure Task")
                task_select = gr.Radio(
                    choices=list(TASK_CONFIG.keys()),
                    label="Task",
                    info="Choose the primary task for the assistant.",
                    value="Question Answering",
                )

                gr.Markdown("### 2. Enable Tools")
                tool_select = gr.CheckboxGroup(
                    choices=["Web Search", "File Search"],
                    label="Tools",
                    info="Enable tools for the assistant.",
                    value=["Web Search"],
                )

                gr.Markdown("### 3. Upload Files (for File Search)")
                with gr.Column():
                    file_upload = gr.File(label="Upload documents", file_count="multiple", type="filepath")
                    upload_btn = gr.Button("Upload to Vector Store")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)

                gr.Markdown("### 4. Session Control")
                reset_btn = gr.Button("Reset Session")
                reset_status = gr.Textbox(label="Session Status", interactive=False)

        # --- Event Listeners ---
        user_input.submit(
            fn=chat_fn, inputs=[user_input, chatbot, task_select, tool_select], outputs=[user_input, chatbot]
        )

        upload_btn.click(fn=upload_files, inputs=[file_upload], outputs=[upload_status])

        task_select.change(fn=reset_session, inputs=None, outputs=[reset_status])
        tool_select.change(fn=reset_session, inputs=None, outputs=[reset_status])
        reset_btn.click(fn=reset_session, inputs=None, outputs=[reset_status])

    return demo
