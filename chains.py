"""
chains.py — RAG Chains & LangGraph Agent
Implements the retrieve-respond pipeline with source citations and guard-rails.
"""
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from guardrails import check_query_safety
from vector_store import get_retriever

# ============================================================
# System Prompt — instructs LLM to cite sources
# ============================================================
SYSTEM_PROMPT = """You are a Document Q&A Assistant. You help users understand 
and answer questions about their uploaded documents.

Rules:
1. Base your answers ONLY on the provided document context below.
2. Always cite the source document for each piece of information using [Source: filename].
3. If the context does not contain enough information, say so honestly.
4. If no documents have been uploaded yet, tell the user to upload a document first.
5. You may engage in brief, friendly conversation (greetings, etc).
6. For questions unrelated to the documents, politely redirect:
   "I can only help with questions about your uploaded documents."

{context_section}

Conversation History:
{chat_history}"""


# ============================================================
# LangGraph Agent State
# ============================================================
class AgentState(TypedDict):
    question: str
    chat_history: str
    documents: List[Document]
    answer: str


def _get_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=LLM_TEMPERATURE
    )


# ============================================================
# Graph Nodes
# ============================================================
def retrieve_node(state: AgentState) -> dict:
    """Retrieve relevant document chunks from the vector store."""
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    # Filter out system placeholder documents
    real_docs = [
        d for d in docs
        if d.metadata.get("source") != "system_init"
        and "System initialized" not in d.page_content
        and "Upload a document to get started" not in d.page_content
    ]
    return {"documents": real_docs}


def respond_node(state: AgentState) -> dict:
    """Generate an answer with source citations."""
    documents = state.get("documents", [])
    chat_history = state.get("chat_history", "")

    if documents:
        parts = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", "?")
            parts.append(f"[Source: {source}, Chunk {chunk_idx}]:\n{doc.page_content}")
        context_section = "Relevant Document Excerpts:\n" + "\n\n".join(parts)
    else:
        context_section = "(No documents uploaded yet. Please upload a PDF or DOCX file first.)"

    system_msg = SYSTEM_PROMPT.format(
        context_section=context_section,
        chat_history=chat_history or "(New conversation)"
    )

    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=state["question"])
    ])
    return {"answer": response.content}


# ============================================================
# Build the LangGraph Agent
# ============================================================
def build_agent():
    """Build and compile the LangGraph RAG agent."""
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("respond", respond_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)
    return workflow.compile()


# Compile the agent
agent = build_agent()


# ============================================================
# Public API
# ============================================================
def call_agent(question: str, chat_history: str = "") -> str:
    """
    Run the RAG agent with guard-rails.
    Returns the answer string with source citations.
    """
    # Guard-rails check
    safety = check_query_safety(question)
    if not safety.is_safe:
        return safety.reason

    try:
        result = agent.invoke({
            "question": question,
            "chat_history": chat_history
        })
        return result["answer"]
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            return "⏳ Rate limit reached. Please wait a few minutes and try again."
        if "401" in err or "invalid_api_key" in err.lower():
            return "🔑 Invalid API key. Please check your GROQ_API_KEY in .env file."
        return f"Error: {err}"


def create_rag_chain():
    """Create a simple RAG chain for LangServe (Runnable interface)."""
    from langchain_core.runnables import RunnableLambda

    def rag_invoke(input_dict):
        question = input_dict.get("question", input_dict.get("input", ""))
        history = input_dict.get("chat_history", "")
        return {"answer": call_agent(question, history)}

    return RunnableLambda(rag_invoke)
