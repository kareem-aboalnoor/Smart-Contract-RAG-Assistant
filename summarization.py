"""
summarization.py — Document Summarization
Uses the LLM to generate concise summaries of uploaded documents.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from ingestion import load_document

SUMMARY_PROMPT = """You are a document summarization expert. 
Provide a clear, concise summary of the following document.
Structure your summary with:
1. **Overview** — What the document is about
2. **Key Points** — The most important information
3. **Conclusions** — Main takeaways

Document content:
{text}"""


def get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )


def summarize_text(text: str, max_chars: int = 20000) -> str:
    """Summarize raw text using the LLM."""
    llm = get_llm()
    truncated = text[:max_chars]
    try:
        response = llm.invoke([
            SystemMessage(content="Summarize the following document concisely."),
            HumanMessage(content=SUMMARY_PROMPT.format(text=truncated))
        ])
        return response.content
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return "⏳ Rate limit reached. Please wait a few minutes and try again."
        return f"Error during summarization: {str(e)}"


def summarize_document(file_path: str, filename: str) -> str:
    """Load and summarize a document file."""
    try:
        text = load_document(file_path)
        if not text.strip():
            return "Document is empty or contains no extractable text."
        return summarize_text(text)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            return "⏳ Rate limit reached. Please wait a few minutes."
        return f"Error: {str(e)}"
