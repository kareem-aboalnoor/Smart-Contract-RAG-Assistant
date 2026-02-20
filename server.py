"""
server.py — FastAPI + LangServe Backend
Provides REST API endpoints and LangServe chain serving.
"""
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import UPLOAD_DIR, SUPPORTED_EXTENSIONS
from chains import call_agent, create_rag_chain
from ingestion import ingest_document
from vector_store import add_documents, clear_vectorstore, load_vectorstore
from summarization import summarize_document

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Document Q&A Assistant API",
    description="RAG-based document Q&A with LangChain, FAISS, and Groq",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request/Response Models
# ============================================================
class ChatRequest(BaseModel):
    question: str
    chat_history: str = ""

class ChatResponse(BaseModel):
    answer: str

class StatusResponse(BaseModel):
    status: str
    message: str


# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
async def root():
    return {"message": "Document Q&A Assistant API is running.", "docs": "/docs"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat with your documents using RAG."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    answer = call_agent(request.question, request.chat_history)
    return ChatResponse(answer=answer)


@app.post("/api/upload", response_model=StatusResponse)
async def upload_endpoint(file: UploadFile = File(...)):
    """Upload a PDF or DOCX file to the knowledge base."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Use: {SUPPORTED_EXTENSIONS}"
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        chunks, _ = ingest_document(file_path, file.filename)
        num_added = add_documents(chunks)

        return StatusResponse(
            status="success",
            message=f"Added '{file.filename}' ({num_added} chunks) to knowledge base."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize")
async def summarize_endpoint(file: UploadFile = File(...)):
    """Upload a file and get an AI-generated summary."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format.")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        summary = summarize_document(file_path, file.filename)
        return {"filename": file.filename, "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear", response_model=StatusResponse)
async def clear_endpoint():
    """Clear the knowledge base."""
    result = clear_vectorstore()
    return StatusResponse(status="success", message=result)


# ============================================================
# LangServe Route
# ============================================================
def setup_langserve(app: FastAPI):
    """Add LangServe routes for the RAG chain."""
    try:
        from langserve import add_routes
        rag_chain = create_rag_chain()
        add_routes(app, rag_chain, path="/rag")
        print("[Server] LangServe route added at /rag")
    except ImportError:
        print("[Server] Warning: langserve not installed. Skipping /rag route.")
    except Exception as e:
        print(f"[Server] Warning: Could not setup LangServe: {e}")


# ============================================================
# Startup Event
# ============================================================
@app.on_event("startup")
async def startup():
    load_vectorstore()
    setup_langserve(app)
    print("[Server] API ready at http://127.0.0.1:8000")
    print("[Server] Docs at http://127.0.0.1:8000/docs")
