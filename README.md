# Smart Contract Summary & Q&A Assistant

A modular RAG (Retrieval-Augmented Generation) pipeline for uploading and chatting with legal documents (contracts, insurance policies, reports).

---

## Features

- Upload PDF or DOCX documents
- Automatic chunking & embedding into a local vector store (Chroma)
- Semantic search-based retrieval
- LLM-powered Q&A with source citations (uses local Llama via Ollama)
- Conversation history
- Guardrails (safety + grounding checks)
- Optional document summarization
- Evaluation pipeline with RAGAS-style metrics
- Clean Gradio UI (Upload tab + Chat tab)
- FastAPI + LangServe microservice backend

---

## Prerequisites

### 1. Install Ollama
```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Then pull the Llama model
ollama pull llama3.2
```

### 2. Python Environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the App

### Option A: All-in-one (Gradio UI only)
```bash
python app/ui/gradio_app.py
```

### Option B: Backend + Frontend separately
```bash
# Terminal 1 — Start FastAPI/LangServe backend
uvicorn app.server:app --reload --port 8000

# Terminal 2 — Start Gradio UI
python app/ui/gradio_app.py
```

Then open: http://localhost:7860

---

## Project Structure

```
smart_contract_assistant/
├── app/
│   ├── ingestion/          # PDF/DOCX parsing, chunking, embedding
│   │   ├── __init__.py
│   │   ├── document_loader.py
│   │   ├── chunker.py
│   │   └── embedder.py
│   ├── retrieval/          # Vector store + semantic retrieval
│   │   ├── __init__.py
│   │   └── retriever.py
│   ├── llm/                # LangChain LLM chain (Ollama/Llama)
│   │   ├── __init__.py
│   │   ├── chain.py
│   │   └── prompts.py
│   ├── guardrails/         # Safety and grounding checks
│   │   ├── __init__.py
│   │   └── guardrails.py
│   ├── summarization/      # Map-reduce summarization
│   │   ├── __init__.py
│   │   └── summarizer.py
│   ├── evaluation/         # Retrieval & answer quality metrics
│   │   ├── __init__.py
│   │   └── evaluator.py
│   ├── ui/
│   │   └── gradio_app.py   # Gradio frontend
│   ├── server.py           # FastAPI + LangServe backend
│   └── config.py           # Central configuration
├── data/                   # Uploaded documents & Chroma DB
├── tests/                  # Unit tests
├── requirements.txt
└── README.md
```

---

## Configuration

Edit `app/config.py` to change:
- Ollama model name (`llama3.2` by default)
- Chunk size / overlap
- Number of retrieved chunks (top-k)
- Vector store path
- Embedding model

---

## Evaluation

Run the evaluation suite against a set of test questions:
```bash
python -m app.evaluation.evaluator
```

Produces metrics: Faithfulness, Answer Relevance, Context Recall.
