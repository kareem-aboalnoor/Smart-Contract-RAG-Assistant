"""
ui.py — Gradio Web Interface
Provides a user-friendly UI with Chat, Upload, and Summarize tabs.
"""
import os
import gradio as gr
from chains import call_agent
from ingestion import ingest_document
from vector_store import add_documents, clear_vectorstore, load_vectorstore
from summarization import summarize_document
from guardrails import get_safety_disclaimer


def chat_fn(message, history):
    """Handle chat messages with conversation history."""
    if not message.strip():
        return "", history

    # Build history text for context
    history_text = ""
    if history:
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                history_text += f"User: {content}\n"
            elif role == "assistant":
                history_text += f"Assistant: {content}\n"

    answer = call_agent(message, history_text)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return "", history


def upload_fn(file_obj):
    """Handle file upload and ingestion."""
    if file_obj is None:
        return "No file selected."
    filename = os.path.basename(file_obj.name)
    try:
        chunks, _ = ingest_document(file_obj.name, filename)
        num_added = add_documents(chunks)
        return f"✅ Added '{filename}' ({num_added} chunks) to the knowledge base."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def summarize_fn(file_obj):
    """Handle document summarization."""
    if file_obj is None:
        return "No file selected."
    filename = os.path.basename(file_obj.name)
    return summarize_document(file_obj.name, filename)


def clear_fn():
    """Clear the knowledge base."""
    return clear_vectorstore()


def build_ui():
    """Build and return the Gradio interface."""
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Document Q&A Assistant"
    ) as demo:

        gr.Markdown("# 📄 Document Q&A Assistant")
        gr.Markdown(
            "Upload a **PDF** or **DOCX** document, then ask questions about it. "
            "Powered by RAG with FAISS + Groq LLaMA 3.3 70B."
        )

        with gr.Tabs():
            # === Chat Tab ===
            with gr.Tab("💬 Chat"):
                chatbot = gr.Chatbot(height=450, type="messages")
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about your documents...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_chat_btn = gr.Button("🗑️ Clear Chat")
                gr.Examples(
                    examples=[
                        "What is this document about?",
                        "Summarize the key points",
                        "What are the main clauses in this contract?",
                        "List all important dates mentioned",
                        "What are the risks described?",
                        "Explain the conclusion section",
                    ],
                    inputs=msg,
                    label="Example Questions"
                )
                msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])
                send_btn.click(chat_fn, [msg, chatbot], [msg, chatbot])
                clear_chat_btn.click(
                    lambda: ("", []), None, [msg, chatbot], queue=False
                )

            # === Upload Tab ===
            with gr.Tab("📤 Upload"):
                gr.Markdown("Upload a PDF or DOCX to add it to the knowledge base.")
                file_input = gr.File(file_types=[".pdf", ".docx"])
                upload_btn = gr.Button("📤 Upload & Process", variant="secondary")
                upload_output = gr.Textbox(label="Status", interactive=False)
                upload_btn.click(upload_fn, file_input, upload_output)
                gr.Markdown("---")
                clear_kb_btn = gr.Button("🗑️ Clear Knowledge Base", variant="stop")
                clear_kb_output = gr.Textbox(label="Clear Status", interactive=False)
                clear_kb_btn.click(clear_fn, None, clear_kb_output)

            # === Summarize Tab ===
            with gr.Tab("📋 Summarize"):
                gr.Markdown("Upload a document to get an AI-generated summary.")
                sum_file = gr.File(file_types=[".pdf", ".docx"])
                sum_btn = gr.Button("📋 Summarize")
                sum_output = gr.Textbox(label="Summary", lines=12, interactive=False)
                sum_btn.click(summarize_fn, sum_file, sum_output)

        gr.Markdown("---")
        gr.Markdown(get_safety_disclaimer())

    return demo


def launch_ui(share=False):
    """Initialize vectorstore and launch the Gradio UI."""
    load_vectorstore()
    demo = build_ui()
    demo.launch(share=share, server_port=7860)
    return demo
