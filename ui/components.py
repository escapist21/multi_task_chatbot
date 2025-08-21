from __future__ import annotations
from re import T

import gradio as gr

from config.settings import TASK_CONFIG, set_openai_api_key
import config.settings as settings
from core.assistant import chat_entry
from core.file_handler import upload_files
from core.state import reset_session


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(
        title="Syntra",
        theme=gr.themes.Default(
            font=[
                gr.themes.GoogleFont("Rubik"),
                "ui-sans-serif",
                "system-ui",
                "sans-serif",
            ],
            primary_hue=gr.themes.colors.emerald,
        ),
    ) as demo:
        gr.Markdown("# Syntra")
        gr.Markdown("## Unified AI Assistant for Search and Tasks")
        gr.Markdown("Powered by OpenAI API")

        with gr.Row():
            # --- Left Column (Chat Interface) ---
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    type="messages",
                    show_copy_button=True,
                    show_copy_all_button=True,
                )
                user_input = gr.Textbox(
                    placeholder="Type your message...",
                    lines=1,
                    label="",
                    scale=10,
                    submit_btn=True,
                )

            # --- Controls Left: API Key + Configure Task ---
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 0. API Key")
                with gr.Column():
                    api_key_box = gr.Textbox(
                        label="OpenAI API Key",
                        placeholder="sk-...",
                        type="password",
                    )
                    set_key_btn = gr.Button("Set API Key")
                    api_key_status = gr.Textbox(
                        label="API Key Status", interactive=False
                    )

                gr.Markdown("### 1. Configure Task")
                task_select = gr.Radio(
                    choices=list(TASK_CONFIG.keys()),
                    label="Task",
                    info="Choose the primary task for the assistant.",
                    value="Generic Assistant",
                )

            # --- Controls Right: Tools + Upload + Session ---
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 2. Enable Tools")
                tool_select = gr.CheckboxGroup(
                    choices=["Web Search", "File Search"],
                    label="Tools",
                    info="Enable tools for the assistant.",
                    value=[],
                )

                gr.Markdown("### 3. Upload Files (for File Search)")
                with gr.Column():
                    file_upload = gr.File(
                        label="Upload documents",
                        file_count="multiple",
                        type="filepath",
                        height=150,
                    )
                    upload_btn = gr.Button("Upload to Vector Store")
                    upload_status = gr.Textbox(label="Upload Status", interactive=False)

                gr.Markdown("### 4. Session Control")
                reset_btn = gr.Button("Reset Session")
                reset_status = gr.Textbox(label="Session Status", interactive=False)

                # Hidden state: streaming is enabled by default
                stream_default = gr.State(True)

        # --- Event Listeners ---
        def apply_api_key(key: str):
            msg = set_openai_api_key(key)
            reset_msg = reset_session()
            return msg, reset_msg

        def on_tools_change(selected_tools: list[str] | None):
            try:
                settings.dprint(f"Tools toggled: {selected_tools}")
            except Exception:
                pass
            return reset_session()

        user_input.submit(
            fn=chat_entry,
            inputs=[user_input, chatbot, task_select, tool_select, stream_default],
            outputs=[user_input, chatbot],
        )

        # On successful upload, auto-enable File Search and switch task to Document QA
        upload_btn.click(
            fn=upload_files,
            inputs=[file_upload],
            outputs=[upload_status, tool_select, task_select],
        )

        task_select.change(fn=reset_session, inputs=None, outputs=[reset_status])
        tool_select.change(
            fn=on_tools_change, inputs=[tool_select], outputs=[reset_status]
        )
        reset_btn.click(fn=reset_session, inputs=None, outputs=[reset_status])

        set_key_btn.click(
            fn=apply_api_key,
            inputs=[api_key_box],
            outputs=[api_key_status, reset_status],
        )

    # Enable queuing so generator outputs stream incrementally in the UI
    demo.queue()
    return demo
