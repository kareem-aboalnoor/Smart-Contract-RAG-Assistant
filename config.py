"""
config.py — Centralized Configuration
All project settings in one place.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# API Keys
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ============================================================
# Model Configuration
# ============================================================
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-mpnet-base-v2"
LLM_TEMPERATURE = 0.3

# ============================================================
# Chunking Configuration
# ============================================================
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ============================================================
# Retriever Configuration
# ============================================================
RETRIEVER_K = 4

# ============================================================
# File Paths
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_index")
INDEX_PATH = os.path.join(SAVE_DIR, "faiss_index")
CONFIG_JSON_PATH = os.path.join(SAVE_DIR, "pipeline_config.json")
SUMMARY_PATH = os.path.join(SAVE_DIR, "doc_summary.txt")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# ============================================================
# Server Configuration
# ============================================================
FASTAPI_HOST = "127.0.0.1"
FASTAPI_PORT = 8000
GRADIO_PORT = 7860

# ============================================================
# Supported File Types
# ============================================================
SUPPORTED_EXTENSIONS = [".pdf", ".docx"]
