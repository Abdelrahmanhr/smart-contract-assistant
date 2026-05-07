"""
server.py — FastAPI + LangServe backend for the Smart Contract Assistant.

Endpoints:
  POST /ingest          — Upload and ingest a PDF or DOCX file
  POST /chat            — Ask a question about the uploaded document
  POST /summarize       — Get an executive summary of the uploaded document
  GET  /health          — Health check
  GET  /docs            — Swagger UI (auto-generated)

LangServe routes:
  /chain/invoke         — Direct LangChain chain invocation
  /chain/stream         — Streaming chain invocation
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from pydantic import BaseModel

from app.config import API_HOST, API_PORT, UPLOAD_DIR
from app.guardrails import apply_guardrails
from app.ingestion import (
    add_to_vector_store,
    build_vector_store,
    chunk_documents,
    load_document,
)
from app.ingestion.embedder import load_vector_store
from app.llm.chain import ConversationalRAGChain, get_llm
from app.retrieval import get_retriever
from app.summarization import summarize_document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Contract Assistant",
    description="Upload legal documents and chat with them using a local Llama model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state (single-process; fine for local/workshop use) ───────────────
_vector_store = None
_rag_chain: Optional[ConversationalRAGChain] = None
_chat_history: List = []
_ingested_files: List[str] = []


def _get_or_reload_chain() -> ConversationalRAGChain:
    """Return the active RAG chain or raise 404 if no document is ingested."""
    global _vector_store, _rag_chain

    if _rag_chain is None:
        vs = load_vector_store()
        if vs is None:
            raise HTTPException(
                status_code=404,
                detail="No document has been ingested yet. Please upload a file first.",
            )
        _vector_store = vs
        retriever = get_retriever(vs)
        _rag_chain = ConversationalRAGChain(retriever=retriever)

    return _rag_chain


# ─── Request / Response models ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    use_guardrails: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    guardrail_info: Dict[str, Any]
    standalone_question: str


class SummarizeResponse(BaseModel):
    summary: str
    files: List[str]


class HealthResponse(BaseModel):
    status: str
    ingested_files: List[str]
    vector_store_loaded: bool


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    vs = load_vector_store()
    return HealthResponse(
        status="ok",
        ingested_files=_ingested_files,
        vector_store_loaded=vs is not None,
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX file.
    The file is parsed, chunked, embedded, and stored in the Chroma vector DB.
    """
    global _vector_store, _rag_chain, _ingested_files

    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Upload a PDF or DOCX.",
        )

    # Save to a temp file then to the uploads directory
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("Received file: %s", save_path)

    try:
        docs = load_document(save_path)
        chunks = chunk_documents(docs)

        if _vector_store is None:
            _vector_store = build_vector_store(chunks)
        else:
            _vector_store = add_to_vector_store(chunks)

        # Rebuild chain with updated vector store
        retriever = get_retriever(_vector_store)
        _rag_chain = ConversationalRAGChain(retriever=retriever)
        _ingested_files.append(file.filename)

        return {
            "message": f"Successfully ingested '{file.filename}'.",
            "pages_loaded": len(docs),
            "chunks_created": len(chunks),
            "filename": file.filename,
        }

    except Exception as exc:
        logger.error("Ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question about the ingested documents."""
    global _chat_history

    chain = _get_or_reload_chain()

    result = chain.run(
        question=request.question,
        chat_history=_chat_history,
    )

    answer = result["answer"]
    source_docs = result["sources"]
    guardrail_info: Dict = {}

    if request.use_guardrails:
        answer, guardrail_info = apply_guardrails(
            question=request.question,
            answer=answer,
            source_docs=source_docs,
        )

    # Update conversation history
    _chat_history = ConversationalRAGChain.build_history(
        _chat_history, request.question, answer
    )

    # Serialise source metadata
    sources_meta = [
        {
            "filename": d.metadata.get("filename", ""),
            "page": d.metadata.get("page", ""),
            "chunk_index": d.metadata.get("chunk_index", ""),
            "snippet": d.page_content[:200],
        }
        for d in source_docs
    ]

    return ChatResponse(
        answer=answer,
        sources=sources_meta,
        guardrail_info=guardrail_info,
        standalone_question=result.get("standalone_question", request.question),
    )


@app.post("/reset_history")
def reset_history():
    """Clear the conversation history."""
    global _chat_history
    _chat_history = []
    return {"message": "Conversation history cleared."}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize():
    """Generate an executive summary of all ingested documents."""
    from app.ingestion.embedder import load_vector_store as _load

    vs = _load()
    if vs is None:
        raise HTTPException(
            status_code=404,
            detail="No documents ingested yet.",
        )

    # Retrieve all stored documents from Chroma
    all_docs_data = vs._collection.get(include=["documents", "metadatas"])
    from langchain_core.documents import Document

    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(
            all_docs_data["documents"], all_docs_data["metadatas"]
        )
    ]

    summary = summarize_document(docs)

    return SummarizeResponse(summary=summary, files=_ingested_files)


# ─── LangServe route (optional programmatic access) ───────────────────────────

def _langserve_handler(inputs: Dict) -> Dict:
    """Thin wrapper so LangServe can invoke the RAG chain."""
    chain = _get_or_reload_chain()
    result = chain.run(
        question=inputs.get("question", ""),
        chat_history=[],
    )
    return {"answer": result["answer"]}


add_routes(
    app,
    RunnableLambda(_langserve_handler),
    path="/chain",
)

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app.server:app", host=API_HOST, port=API_PORT, reload=True)
