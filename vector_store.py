"""
vector_store.py — FAISS Vector Store Management
Handles initialization, loading, saving, and querying the vector store.
"""
import os
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import (
    EMBEDDING_MODEL, SAVE_DIR, INDEX_PATH, CONFIG_JSON_PATH,
    SUMMARY_PATH, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVER_K, LLM_MODEL
)

# Module-level state
_embeddings = None
_vectorstore = None
_retriever = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        print(f"[VectorStore] Loading embedding model: {EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def initialize_vectorstore():
    """Create a fresh vectorstore with a placeholder document."""
    global _vectorstore, _retriever
    embeddings = get_embeddings()
    placeholder = Document(
        page_content="System initialized. Upload a document to get started.",
        metadata={"source": "system_init"}
    )
    _vectorstore = FAISS.from_documents([placeholder], embeddings)
    _retriever = _vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    save_vectorstore()
    save_pipeline_config(total_chunks=1, init_mode="placeholder")
    print("[VectorStore] Initialized with placeholder document.")
    return _vectorstore


def load_vectorstore():
    """Load an existing vectorstore from disk, or create a new one."""
    global _vectorstore, _retriever
    embeddings = get_embeddings()

    if os.path.exists(INDEX_PATH):
        print(f"[VectorStore] Loading index from: {INDEX_PATH}")
        _vectorstore = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        _retriever = _vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        print("[VectorStore] Index loaded successfully.")
    else:
        print("[VectorStore] No existing index found. Creating new one.")
        initialize_vectorstore()

    return _vectorstore


def save_vectorstore():
    """Save the vectorstore to disk."""
    global _vectorstore
    if _vectorstore is None:
        raise RuntimeError("Vectorstore not initialized.")
    os.makedirs(SAVE_DIR, exist_ok=True)
    _vectorstore.save_local(INDEX_PATH)
    print(f"[VectorStore] Index saved to: {INDEX_PATH}")


def add_documents(chunks):
    """Add document chunks to the vectorstore and update retriever."""
    global _vectorstore, _retriever
    if _vectorstore is None:
        load_vectorstore()
    _vectorstore.add_documents(chunks)
    save_vectorstore()
    _retriever = _vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    return len(chunks)


def clear_vectorstore():
    """Reset the vectorstore to initial state."""
    initialize_vectorstore()
    return "Knowledge base cleared successfully."


def get_retriever():
    global _retriever
    if _retriever is None:
        load_vectorstore()
    return _retriever


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        load_vectorstore()
    return _vectorstore


def save_pipeline_config(total_chunks=1, init_mode="placeholder"):
    """Save pipeline configuration to JSON."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    config = {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": LLM_MODEL,
        "retriever_k": RETRIEVER_K,
        "total_chunks": total_chunks,
        "init_mode": init_mode,
        "provider": "groq"
    }
    with open(CONFIG_JSON_PATH, "w") as f:
        json.dump(config, f, indent=2)
