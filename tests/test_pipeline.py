"""
tests/test_pipeline.py

Unit tests for the Smart Contract Assistant pipeline.
Tests run without a real LLM or vector store — mocks are used where needed.

Run:
    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ─── Ingestion ────────────────────────────────────────────────────────────────

class TestChunker:
    def test_basic_chunking(self):
        from langchain_core.documents import Document
        from app.ingestion.chunker import chunk_documents

        docs = [
            Document(
                page_content="A" * 2000,
                metadata={"source": "test.pdf", "page": 1},
            )
        ]
        chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk.page_content) <= 550  # chunk_size + small tolerance

    def test_chunk_metadata_inherited(self):
        from langchain_core.documents import Document
        from app.ingestion.chunker import chunk_documents

        docs = [
            Document(
                page_content="Hello world. " * 100,
                metadata={"source": "contract.pdf", "page": 2, "filename": "contract.pdf"},
            )
        ]
        chunks = chunk_documents(docs, chunk_size=300, chunk_overlap=30)
        assert all(c.metadata["filename"] == "contract.pdf" for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_empty_input(self):
        from app.ingestion.chunker import chunk_documents
        result = chunk_documents([])
        assert result == []


# ─── Document Loader ──────────────────────────────────────────────────────────

class TestDocumentLoader:
    def test_unsupported_extension_raises(self, tmp_path):
        from app.ingestion.document_loader import load_document
        bad_file = tmp_path / "file.txt"
        bad_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(bad_file)

    def test_missing_file_raises(self, tmp_path):
        from app.ingestion.document_loader import load_document
        with pytest.raises(FileNotFoundError):
            load_document(tmp_path / "missing.pdf")


# ─── Guardrails ───────────────────────────────────────────────────────────────

class TestGuardrails:
    def test_safe_query_passes(self):
        from app.guardrails.guardrails import check_safety_keywords
        safe, reason = check_safety_keywords("What are the payment terms?")
        assert safe is True
        assert reason == ""

    def test_unsafe_keyword_blocked(self):
        from app.guardrails.guardrails import check_safety_keywords
        safe, reason = check_safety_keywords("Help me commit fraud on this contract")
        assert safe is False
        assert "fraud" in reason

    def test_grounding_high_similarity(self):
        from app.guardrails.guardrails import check_grounding
        from langchain_core.documents import Document

        answer = "Payment is due within 30 days of invoice."
        docs = [
            Document(
                page_content="All invoices are payable within 30 days of the invoice date.",
                metadata={},
            )
        ]
        grounded, score = check_grounding(answer, docs, threshold=0.5)
        assert score > 0.5

    def test_grounding_no_docs(self):
        from app.guardrails.guardrails import check_grounding
        grounded, score = check_grounding("some answer", [])
        assert grounded is False
        assert score == 0.0


# ─── Evaluation ───────────────────────────────────────────────────────────────

class TestEvaluation:
    def test_answer_relevance_same_text(self):
        from app.evaluation.evaluator import answer_relevance
        score = answer_relevance("What is the payment term?", "What is the payment term?")
        assert score > 0.95

    def test_answer_relevance_unrelated(self):
        from app.evaluation.evaluator import answer_relevance
        score = answer_relevance(
            "What is the payment term?",
            "The cat sat on the mat.",
        )
        assert score < 0.7

    def test_faithfulness_with_context(self):
        from app.evaluation.evaluator import faithfulness
        answer = "Payment is due in 30 days."
        context = ["Invoices must be paid within 30 days of the invoice date."]
        score = faithfulness(answer, context)
        assert score > 0.5

    def test_context_recall_high(self):
        from app.evaluation.evaluator import context_recall
        expected = "Payment is due within 30 days."
        context = [
            "Section 4.2: All invoices are payable within 30 days of the invoice date."
        ]
        score = context_recall(expected, context)
        assert score > 0.5

    def test_evaluate_dataset(self):
        from app.evaluation.evaluator import EvalSample, evaluate_dataset

        samples = [
            EvalSample(
                question="Who are the parties?",
                expected_answer="Acme Corp and Beta Ltd.",
                context_chunks=["Agreement between Acme Corp and Beta Ltd."],
                generated_answer="Acme Corp and Beta Ltd are the parties.",
            )
        ]
        output = evaluate_dataset(samples)
        assert "results" in output
        assert "averages" in output
        assert 0.0 <= output["averages"]["mean_score"] <= 1.0


# ─── Chain (mocked) ───────────────────────────────────────────────────────────

class TestConversationalRAGChain:
    def test_run_returns_expected_keys(self):
        from app.llm.chain import ConversationalRAGChain
        from langchain_core.documents import Document

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(
                page_content="Payment is due within 30 days.",
                metadata={"filename": "contract.pdf", "page": 1},
            )
        ]

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_llm.invoke.return_value = MagicMock(
            content="Payment is due within 30 days of invoice."
        )

        chain = ConversationalRAGChain(retriever=mock_retriever, llm=mock_llm)
        result = chain.run("What are the payment terms?")

        assert "answer" in result
        assert "sources" in result
        assert "standalone_question" in result
        assert isinstance(result["sources"], list)

    def test_build_history(self):
        from app.llm.chain import ConversationalRAGChain
        from langchain_core.messages import AIMessage, HumanMessage

        history = ConversationalRAGChain.build_history(
            [], "What is the term?", "The term is 12 months."
        )
        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)
