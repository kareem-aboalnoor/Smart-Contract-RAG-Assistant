# 📄 Chat With Your PDF — RAG Document Assistant

An AI-powered **Retrieval-Augmented Generation (RAG)** application that lets you upload PDF or DOCX files and chat with them using natural language. Built with **LangChain**, **LangGraph**, **FAISS**, **FastAPI**, **LangServe**, **Groq (LLaMA 3.3 70B)**, and **Gradio**.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.3.1-green)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal)
![Groq](https://img.shields.io/badge/LLM-LLaMA%203.3%2070B-orange)
![Gradio](https://img.shields.io/badge/UI-Gradio-yellow)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 💬 **Chat with Documents** | Ask questions and get accurate answers with **source citations** |
| 📤 **Upload Documents** | Upload PDF or DOCX files to build a searchable knowledge base |
| 📋 **Summarize Documents** | Get AI-generated summaries of any uploaded document |
| 🔍 **RAG Pipeline** | FAISS vector search retrieves relevant chunks before answering |
| 🛡️ **Guard-Rails** | Blocks prompt injection and enforces safety policies |
| 🌐 **FastAPI + LangServe** | REST API backend with LangServe chain serving |
| 📊 **Evaluation Pipeline** | Built-in metrics: retrieval quality, latency, guardrail tests |
| 🏗️ **Modular Codebase** | Clean, structured Python modules for easy extension |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Frontend (Gradio UI)               │
│            Tabs: [Chat] [Upload] [Summarize]          │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│            Backend (FastAPI + LangServe)              │
│   /api/chat  /api/upload  /api/summarize  /rag       │
└──────────────────────┬───────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌──────────┐
   │ Guard-Rails│ │ RAG Agent│ │Summarizer│
   │ (Safety)   │ │(LangGraph)│ │  (LLM)   │
   └────────────┘ └─────┬────┘ └──────────┘
                        │
              ┌─────────┼─────────┐
              ▼                   ▼
       ┌────────────┐     ┌────────────┐
       │  Retriever  │     │ Groq LLM   │
       │   (FAISS)   │     │(LLaMA 3.3) │
       └──────┬─────┘     └────────────┘
              │
       ┌──────┴──────┐
       │  Embeddings  │
     (all-mpnet-base-v2)│
       └─────────────┘
```

---

## 📁 Project Structure

```
Chat-With-Your-PDF-main/
│
│── Source Code (Python Modules)
├── config.py                    # Centralized configuration
├── ingestion.py                 # Document loading & chunking (PDF/DOCX)
├── vector_store.py              # FAISS vector store management
├── chains.py                    # LangGraph RAG agent + source citations
├── guardrails.py                # Input safety & prompt injection blocking
├── summarization.py             # Document summarization
├── evaluation.py                # Evaluation pipeline with metrics
├── server.py                    # FastAPI + LangServe backend
├── ui.py                        # Gradio web interface
├── main.py                      # Application entry point
│
│── Configuration
├── requirements.txt             # Python dependencies
├── .env                         # API key (secret — not uploaded to git)
├── .env.example                 # API key template
├── .gitignore                   # Git ignore rules
│
│── Documentation
├── README.md                    # This file
├── explanation_source_code.md   # Detailed code explanation (Arabic)
├── explanation_notebook1.md     # Notebook 1 explanation (Arabic)
├── explanation_notebook2.md     # Notebook 2 explanation (Arabic)
│
│── Original Notebooks (Colab)
├── 01_build_rag_pipeline.ipynb  # Build FAISS index (Colab version)
└── 02_gradio_ui.ipynb           # Gradio UI (Colab version)
```

---

## 🚀 Quick Start (Local — Python 3.10)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure API Key

Create a `.env` file in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free API key from [console.groq.com](https://console.groq.com).

### Step 3: Run the Application

```bash
# Option 1: Gradio UI only (recommended)
python main.py

# Option 2: FastAPI server only
python main.py --api

# Option 3: Both API + UI together
python main.py --both

# Option 4: Run evaluation pipeline
python main.py --evaluate

# Option 5: Evaluate with a specific test document
python main.py --evaluate path/to/document.pdf

# Option 6: Gradio with public share link
python main.py --share
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM** | Groq — LLaMA 3.3 70B Versatile |
| **Embeddings** | Sentence Transformers — all-MiniLM-L6-v2 |
| **Vector Store** | FAISS (Facebook AI Similarity Search) |
| **Framework** | LangChain + LangGraph |
| **API Server** | FastAPI + LangServe |
| **UI** | Gradio |
| **Document Parsing** | PyPDF, pdfplumber, python-docx |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/chat` | Send a question, get an answer with citations |
| `POST` | `/api/upload` | Upload a PDF/DOCX to the knowledge base |
| `POST` | `/api/summarize` | Upload a file and get a summary |
| `POST` | `/api/clear` | Clear the knowledge base |
| `POST` | `/rag/invoke` | LangServe RAG chain endpoint |
| `GET` | `/docs` | Interactive API documentation (Swagger) |

---

## 📋 Configuration

All settings are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model for generation |
| `EMBEDDING_MODEL` | `all-mpnet-base-v2` | Model for text embeddings |
| `CHUNK_SIZE` | `500` | Characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `RETRIEVER_K` | `4` | Number of chunks to retrieve |
| `FASTAPI_PORT` | `8000` | FastAPI server port |
| `GRADIO_PORT` | `7860` | Gradio UI port |

---

## 📊 Evaluation

Run the evaluation pipeline to test:
- **Guard-Rail Effectiveness** — Prompt injection blocking accuracy
- **Retrieval Quality** — Chunks retrieved per question
- **Semantic Similarity** — Cosine similarity between queries and context
- **Answer Quality** — Response time, citation rate, error rate

```bash
python main.py --evaluate
python main.py --evaluate test_document.pdf
```

Results are saved to `evaluation_report.md`.

---

## 🛡️ Guard-Rails

- **Prompt injection blocking** — Detects and blocks injection patterns
- **Off-topic filtering** — Redirects non-document questions
- **Source grounding** — Answers cite specific document sources
- **Safety disclaimer** — Displayed in the UI

---

## 📝 Example Questions

After uploading a document:

- *"What is this document about?"*
- *"Summarize the key points"*
- *"What are the main clauses in this contract?"*
- *"List all important dates mentioned"*
- *"What are the risks described in the document?"*
