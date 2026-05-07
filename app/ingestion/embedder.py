"""
ingestion/embedder.py

Creates and manages the Chroma vector store.
Uses local SentenceTransformer embeddings — no API key required.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    VECTOR_STORE_COLLECTION,
)

logger = logging.getLogger(__name__)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFace embedding model (local, no API key needed).
    The model is downloaded on first use and cached locally by sentence-transformers.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(
    chunks: List[Document],
    collection_name: str = VECTOR_STORE_COLLECTION,
    persist_directory: str = str(CHROMA_DIR),
) -> Chroma:
    """
    Embed a list of document chunks and store them in a (persistent) Chroma DB.

    Args:
        chunks:           Chunked Documents from the ingestion pipeline.
        collection_name:  Chroma collection name.
        persist_directory: Directory where Chroma persists its data.

    Returns:
        A Chroma vector store instance ready for similarity search.
    """
    if not chunks:
        raise ValueError("Cannot build a vector store from an empty chunk list.")

    logger.info(
        "Embedding %d chunks into Chroma collection '%s' …",
        len(chunks),
        collection_name,
    )

    embeddings = _get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    logger.info(
        "Vector store built: %d vectors in collection '%s' at '%s'.",
        len(chunks),
        collection_name,
        persist_directory,
    )
    return vector_store


def add_to_vector_store(
    chunks: List[Document],
    collection_name: str = VECTOR_STORE_COLLECTION,
    persist_directory: str = str(CHROMA_DIR),
) -> Chroma:
    """
    Add new chunks to an existing Chroma collection (incremental ingestion).
    If the collection doesn't exist yet, it will be created.

    Args:
        chunks:           New chunked Documents to add.
        collection_name:  Chroma collection name.
        persist_directory: Directory where Chroma persists its data.

    Returns:
        The updated Chroma vector store.
    """
    if not chunks:
        raise ValueError("No chunks provided to add_to_vector_store.")

    embeddings = _get_embeddings()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vector_store.add_documents(chunks)

    logger.info(
        "Added %d new chunks to collection '%s'.",
        len(chunks),
        collection_name,
    )
    return vector_store


def load_vector_store(
    collection_name: str = VECTOR_STORE_COLLECTION,
    persist_directory: str = str(CHROMA_DIR),
) -> Optional[Chroma]:
    """
    Load an existing Chroma vector store from disk.

    Returns:
        The Chroma instance, or None if the collection is empty / not found.
    """
    embeddings = _get_embeddings()
    try:
        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        count = store._collection.count()
        if count == 0:
            logger.warning(
                "Chroma collection '%s' exists but is empty.", collection_name
            )
            return None
        logger.info(
            "Loaded Chroma collection '%s' (%d vectors).", collection_name, count
        )
        return store
    except Exception as exc:
        logger.error("Failed to load Chroma vector store: %s", exc)
        return None
