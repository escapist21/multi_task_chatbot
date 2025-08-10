import gradio as gr
from openai import OpenAI
import os
from dotenv import load_dotenv
import time

# Import the specific types to satisfy the type checker
from openai.types.beta.assistant_create_params import ToolResources

# Load environment variables from a .env file
load_dotenv()

# Initialise OpenAI client
# Make sure your OPENAI_API_KEY is set in your .env file or environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in your .env file or as an environment variable.")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- CONFIGURATIONS ---

# Define task configurations with model, temperature, etc.
TASK_CONFIG = {
    "Text Classification": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Question Answering": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Table Question Answering": {"model": "gpt-4.1", "temperature": 0.0},
    "Sentence Similarity": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Translation": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Summarisation": {"model": "gpt-4.1-mini", "temperature": 0.0},
    "Document Question Answering": {"model": "gpt-4.1", "temperature": 0.0}
}

# System prompts that define the assistant's behavior for each task
SYS_PROMPTS = {
    "Text Classification": "You are an expert text classifier.",
    "Question Answering": "You are a helpful question answering assistant. Use the search tool if you don't know the answer. Mention the information source and date.",
    "Table Question Answering": "You are a helpful question answering assistant especially capable of summarising data in tables.",
    "Sentence Similarity": "You are an expert in detecting similar sentences.",
    "Translation": "Translate the user's text as precisely as possible. Try to maintain semantic coherence.",
    "Summarisation": "Summarize the provided content concisely. If the file_search tool is enabled using the document. If no word limit or format (e.g., bullet points, paragraph) is provided, ask the user for clarification.",
    "Document Question Answering": "You are an expert at answering questions based on the provided files. Use the file_search tool to find relevant information within the documents. Always follow the word limit and the tone suggested by the user. If no word limit is provided ask the user for it. If no tone is provided ask the user for the purpose and derive the tone from there."
}

# --- STATE MANAGEMENT ---

# Global variables to hold state across Gradio interactions
vector_store_id = None
assistant_id = None
thread_id = None

# --- CORE FUNCTIONS ---

def reset_session():
    """Resets the assistant and thread, forcing recreation on the next message."""
    global assistant_id, thread_id
    assistant_id = None
    thread_id = None
    print("Session reset. New assistant and thread will be created.")
    return "Session has been reset."

def upload_files(files):
    """Uploads files to OpenAI and adds them to a vector store."""
    global vector_store_id

    if not files:
        return "No files selected. Please upload at least one file."

    try:
        # Vector Stores API is at the top level, not under .beta
        vs = client.vector_stores.create(name=f"chatbot_store_{int(time.time())}")
        vector_store_id = vs.id
        
        file_paths = [f.name for f in files]
        file_streams = [open(path, "rb") for path in file_paths]
        
        print(f"Uploading {len(file_paths)} files to new Vector Store: {vector_store_id}")
        
        # Use the file_batches API for robust uploading and polling.
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store_id, files=file_streams
        )

        # Close the file streams after upload
        for stream in file_streams:
            stream.close()

        # After uploading, the current assistant is outdated. Resetting forces recreation.
        reset_session()

        return f"Uploaded {len(files)} files. Vector Store Status: {file_batch.status}"

    except Exception as e:
        print(f"Error during file upload: {e}")
        return f"An error occurred: {e}"


def chat_fn(message, history, task, enabled_tools):
    """Main chat function that uses the Assistants API."""
    global assistant_id, thread_id, vector_store_id

    # --- 1. Create or Update Assistant ---
    if not assistant_id:
        print("No assistant found. Creating a new one...")
        instructions = SYS_PROMPTS.get(task, "You are a helpful assistant.")
        cfg = TASK_CONFIG.get(task, {"model": "gpt-4o-mini"})

        assistant_tools = []
        if "Web Search" in enabled_tools:
            assistant_tools.append({"type": "search"})

        # Initialize tool_resources as None, with the correct type hint
        tool_resources: ToolResources | None = None
        if "File Search" in enabled_tools:
            assistant_tools.append({"type": "file_search"})
            if vector_store_id:
                # Instantiate the specific class required by the API
                tool_resources = ToolResources(
                    file_search={"vector_store_ids": [vector_store_id]}
                )
            else:
                print("Warning: File Search is enabled, but no files have been uploaded.")

        try:
            assistant = client.beta.assistants.create(
                name="Multi-Task Chatbot",
                instructions=instructions,
                tools=assistant_tools,
                model=cfg["model"],
                tool_resources=tool_resources
            )
            assistant_id = assistant.id
            print(f"Created new Assistant (ID: {assistant_id}) for task '{task}' with tools: {enabled_tools}")
        except Exception as e:
            print(f"Error creating assistant: {e}")
            history.append((message, f"Error: Could not create the assistant. {e}"))
            return "", history

    # --- 2. Create a Thread ---
    if not thread_id:
        print("No thread found. Creating a new one...")
        try:
            thread = client.beta.threads.create()
            thread_id = thread.id
            print(f"Created new Thread (ID: {thread_id})")
        except Exception as e:
            print(f"Error creating thread: {e}")
            history.append((message, f"Error: Could not create the conversation thread. {e}"))
            return "", history

    # --- 3. Add User's Message to the Thread ---
    try:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=message
        )
    except Exception as e:
        print(f"Error adding message to thread: {e}")
        history.append((message, f"Error: Could not process your message. {e}"))
        return "", history

    # --- 4. Run the Assistant and Poll for Completion ---
    try:
        print(f"Running Assistant {assistant_id} on Thread {thread_id}...")
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
        )
    except Exception as e:
        print(f"Error during assistant run: {e}")
        history.append((message, f"Error: The assistant failed to run. {e}"))
        return "", history

    # --- 5. Retrieve and Display the Response ---
    if run.status == 'completed':
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        assistant_message = messages.data[0]
        
        reply_parts = []
        for content_block in assistant_message.content:
            # Check the type of the content block before accessing its attributes
            if content_block.type == 'text':
                reply_parts.append(content_block.text.value)
        
        reply = "".join(reply_parts)
        
        history.append((message, reply))
        return "", history
    else:
        print(f"Run failed with status: {run.status}")
        error_message = f"Run failed with status: {run.status}. Please try again."
        if run.last_error:
            error_message += f" Details: {run.last_error.message}"
        history.append((message, error_message))
        return "", history


# --- GRADIO UI ---

# Removed the theme argument to be compatible with older Gradio versions.
# For themes to work, run: pip install --upgrade gradio
with gr.Blocks() as demo:
    gr.Markdown("# Multi-Task Chatbot with Web & File Search")
    gr.Markdown("Powered by OpenAI Assistants API")

    with gr.Row():
        # --- Left Column (Chat Interface) ---
        with gr.Column(scale=3):
            # Removed render=False to fix the layout
            chatbot = gr.Chatbot(label="Conversation", height=600)
            user_input = gr.Textbox(placeholder="Type your message...", lines=1, label="", scale=10)
        
        # --- Right Column (Controls) ---
        with gr.Column(scale=1):
            gr.Markdown("### 1. Configure Task")
            task_select = gr.Radio(
                choices=list(TASK_CONFIG.keys()),
                label="Task",
                info="Choose the primary task for the assistant.",
                value="Question Answering"
            )

            gr.Markdown("### 2. Enable Tools")
            tool_select = gr.CheckboxGroup(
                choices=["Web Search", "File Search"],
                label="Tools",
                info="Enable tools for the assistant.",
                value=["Web Search"]
            )
            
            gr.Markdown("### 3. Upload Files (for File Search)")
            with gr.Column():
                file_upload = gr.File(
                    label="Upload documents",
                    file_count="multiple",
                    type="filepath"
                )
                upload_btn = gr.Button("Upload to Vector Store")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)
            
            gr.Markdown("### 4. Session Control")
            reset_btn = gr.Button("Reset Session")
            reset_status = gr.Textbox(label="Session Status", interactive=False)

    # --- Event Listeners ---
    
    user_input.submit(
        fn=chat_fn, 
        inputs=[user_input, chatbot, task_select, tool_select], 
        outputs=[user_input, chatbot]
    )
    
    upload_btn.click(
        fn=upload_files, 
        inputs=[file_upload], 
        outputs=[upload_status]
    )
    
    task_select.change(fn=reset_session, inputs=None, outputs=[reset_status])
    tool_select.change(fn=reset_session, inputs=None, outputs=[reset_status])
    reset_btn.click(fn=reset_session, inputs=None, outputs=[reset_status])



if __name__ == "__main__":
    demo.launch(share=True, debug=True)