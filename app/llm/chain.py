"""
llm/chain.py

Builds and runs the conversational RAG chain.

Flow:
  1. If there is chat history → condense the follow-up into a standalone question.
  2. Retrieve relevant chunks from the vector store.
  3. Feed context + question to the Llama LLM via Ollama.
  4. Return the answer text and the source documents used.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_ollama import ChatOllama

from app.config import LLM_MAX_TOKENS, LLM_TEMPERATURE, OLLAMA_BASE_URL, OLLAMA_MODEL
from app.llm.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT

logger = logging.getLogger(__name__)


# ─── LLM singleton ────────────────────────────────────────────────────────────

def get_llm(
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
) -> ChatOllama:
    """
    Return a ChatOllama instance pointing at the local Ollama server.
    Raises a clear error if Ollama is not reachable.
    """
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=temperature,
        num_predict=max_tokens,
    )


# ─── Context formatter ────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """Convert retrieved documents to a single context string with page refs."""
    parts: list[str] = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        ref = f"[Source {i} | {meta.get('filename', 'unknown')} | page {meta.get('page', '?')}]"
        parts.append(f"{ref}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ─── Standalone-question condenser ────────────────────────────────────────────

def _build_condenser(llm: ChatOllama):
    """Chain that rewrites a follow-up question into a standalone question."""
    return CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()


# ─── Main RAG chain ───────────────────────────────────────────────────────────

class ConversationalRAGChain:
    """
    Stateless conversational RAG chain.

    The caller is responsible for maintaining chat_history between turns.
    This keeps the chain itself simple and easily testable.
    """

    def __init__(self, retriever: BaseRetriever, llm: ChatOllama | None = None):
        self.retriever = retriever
        self.llm = llm or get_llm()
        self._condenser = _build_condenser(self.llm)
        logger.info(
            "ConversationalRAGChain ready (model=%s).", self.llm.model
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _condense_question(
        self, question: str, chat_history: List[BaseMessage]
    ) -> str:
        """Rewrite a follow-up question using chat history, if needed."""
        if not chat_history:
            return question
        try:
            condensed = self._condenser.invoke(
                {"question": question, "chat_history": chat_history}
            )
            logger.debug("Condensed question: '%s' → '%s'", question, condensed)
            return condensed
        except Exception as exc:
            logger.warning("Question condensing failed, using original: %s", exc)
            return question

    def _retrieve(self, standalone_question: str) -> List[Document]:
        """Run semantic retrieval for the standalone question."""
        docs = self.retriever.invoke(standalone_question)
        logger.debug("Retrieved %d chunks for question.", len(docs))
        return docs

    def _generate_answer(
        self,
        question: str,
        context_docs: List[Document],
        chat_history: List[BaseMessage],
    ) -> str:
        """Call the LLM with context and return the answer string."""
        context_str = _format_docs(context_docs)
        messages = QA_PROMPT.format_messages(
            context=context_str,
            question=question,
            chat_history=chat_history,
        )
        response = self.llm.invoke(messages)
        return response.content

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        chat_history: List[BaseMessage] | None = None,
    ) -> Dict:
        """
        Answer a question given an optional conversation history.

        Args:
            question:     The user's question.
            chat_history: List of HumanMessage / AIMessage objects.

        Returns:
            A dict with keys:
                - "answer":   The LLM's answer string.
                - "sources":  List of source Document objects used.
                - "standalone_question": The (possibly condensed) question.
        """
        history = chat_history or []

        standalone = self._condense_question(question, history)
        source_docs = self._retrieve(standalone)

        if not source_docs:
            answer = (
                "I could not find relevant information in the uploaded document "
                "to answer your question. Please try rephrasing."
            )
        else:
            answer = self._generate_answer(standalone, source_docs, history)

        return {
            "answer": answer,
            "sources": source_docs,
            "standalone_question": standalone,
        }

    @staticmethod
    def build_history(
        previous_history: List[BaseMessage],
        human_question: str,
        ai_answer: str,
    ) -> List[BaseMessage]:
        """
        Append the latest turn to the conversation history.

        Args:
            previous_history: Existing history list.
            human_question:   The question just asked.
            ai_answer:        The answer just produced.

        Returns:
            Updated history list.
        """
        return previous_history + [
            HumanMessage(content=human_question),
            AIMessage(content=ai_answer),
        ]
