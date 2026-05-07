"""
ui/gradio_app.py

Gradio web interface for the Smart Contract Assistant.

Tabs:
  1. 📄 Upload & Ingest  — upload PDF/DOCX, trigger ingestion, optional summary
  2. 💬 Chat             — conversational Q&A with source citations
  3. 📊 Evaluation       — run the built-in evaluation demo

Run standalone:
    python app/ui/gradio_app.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gradio as gr
from langchain_core.messages import AIMessage, HumanMessage

from app.config import GRADIO_PORT, GRADIO_SHARE
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

# ─── Global app state ─────────────────────────────────────────────────────────

class AppState:
    vector_store = None
    rag_chain: ConversationalRAGChain | None = None
    chat_history = []          # LangChain message objects
    ingested_files: list[str] = []


STATE = AppState()


# ─── Ingestion logic ──────────────────────────────────────────────────────────

def ingest_file(file_obj, progress=gr.Progress()):
    """
    Called when a user uploads a file in the Upload tab.
    Returns status text and enables the chat tab.
    """
    if file_obj is None:
        return "⚠️ No file selected.", gr.update(interactive=False)

    file_path = Path(file_obj.name)
    filename = file_path.name
    progress(0, desc="Reading document …")

    try:
        docs = load_document(file_path)
        progress(0.3, desc=f"Loaded {len(docs)} page(s) …")

        chunks = chunk_documents(docs)
        progress(0.5, desc=f"Created {len(chunks)} chunk(s) …")

        progress(0.6, desc="Embedding chunks (this may take a minute) …")
        if STATE.vector_store is None:
            # Try to load an existing store first
            existing = load_vector_store()
            if existing is not None:
                STATE.vector_store = existing
                STATE.vector_store = add_to_vector_store(chunks)
            else:
                STATE.vector_store = build_vector_store(chunks)
        else:
            STATE.vector_store = add_to_vector_store(chunks)

        progress(0.9, desc="Building retrieval chain …")
        retriever = get_retriever(STATE.vector_store)
        STATE.rag_chain = ConversationalRAGChain(retriever=retriever)
        STATE.ingested_files.append(filename)
        progress(1.0, desc="Done!")

        status = (
            f"✅ **{filename}** ingested successfully!\n\n"
            f"- Pages loaded: {len(docs)}\n"
            f"- Chunks created: {len(chunks)}\n"
            f"- Files in store: {', '.join(STATE.ingested_files)}\n\n"
            "Switch to the **💬 Chat** tab to start asking questions."
        )
        return status, gr.update(interactive=True)

    except Exception as exc:
        logger.error("Ingestion error: %s", exc, exc_info=True)
        return f"❌ Ingestion failed: {exc}", gr.update(interactive=False)


def run_summarize():
    """Generate an executive summary of all ingested documents."""
    if STATE.vector_store is None:
        return "⚠️ Please upload a document first."

    try:
        all_data = STATE.vector_store._collection.get(
            include=["documents", "metadatas"]
        )
        from langchain_core.documents import Document

        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(all_data["documents"], all_data["metadatas"])
        ]
        summary = summarize_document(docs)
        return f"### 📋 Executive Summary\n\n{summary}"
    except Exception as exc:
        return f"❌ Summarisation failed: {exc}"


# ─── Chat logic ───────────────────────────────────────────────────────────────

def chat(user_message: str, history: list, use_guardrails: bool):
    """
    Process a user message and return (updated_history, sources_text).
    history is the Gradio-format list of [user_msg, bot_msg] pairs.
    """
    if not user_message.strip():
        return history, ""

    if STATE.rag_chain is None:
        # Try loading from disk
        vs = load_vector_store()
        if vs is None:
            history.append([user_message, "⚠️ Please upload a document first."])
            return history, ""
        STATE.vector_store = vs
        retriever = get_retriever(vs)
        STATE.rag_chain = ConversationalRAGChain(retriever=retriever)

    try:
        result = STATE.rag_chain.run(
            question=user_message,
            chat_history=STATE.chat_history,
        )

        answer = result["answer"]
        source_docs = result["sources"]

        if use_guardrails:
            answer, guardrail_info = apply_guardrails(
                question=user_message,
                answer=answer,
                source_docs=source_docs,
            )
        else:
            guardrail_info = {}

        # Update LangChain history
        STATE.chat_history = ConversationalRAGChain.build_history(
            STATE.chat_history, user_message, answer
        )

        # Build sources panel text
        if source_docs:
            sources_lines = ["**📚 Sources used:**\n"]
            for i, doc in enumerate(source_docs, 1):
                meta = doc.metadata
                sources_lines.append(
                    f"**[{i}]** `{meta.get('filename', '?')}` "
                    f"— Page {meta.get('page', '?')}\n"
                    f"> {doc.page_content[:300].strip()}…\n"
                )
            sources_text = "\n".join(sources_lines)
        else:
            sources_text = "*No relevant chunks retrieved.*"

        history.append([user_message, answer])
        return history, sources_text

    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        history.append([user_message, f"❌ Error: {exc}"])
        return history, ""


def clear_chat():
    """Reset conversation history."""
    STATE.chat_history = []
    return [], ""


# ─── Evaluation logic ─────────────────────────────────────────────────────────

def run_evaluation_demo():
    """Run the built-in evaluation demo and return a formatted report."""
    from app.evaluation.evaluator import (
        EvalSample,
        evaluate_dataset,
    )
    import io, contextlib

    samples = [
        EvalSample(
            question="What are the payment terms?",
            expected_answer="Payment is due within 30 days of invoice.",
            context_chunks=[
                "Section 4.2: All invoices are payable within 30 days of the invoice date.",
                "Late payments will incur a 2% monthly interest charge.",
            ],
            generated_answer="According to Section 4.2, payment is due within 30 days of the invoice date.",
        ),
        EvalSample(
            question="Who are the parties to the agreement?",
            expected_answer="Acme Corp and Beta Ltd.",
            context_chunks=[
                "This Agreement is entered into between Acme Corp ('Vendor') and Beta Ltd ('Client').",
            ],
            generated_answer="The parties are Acme Corp (Vendor) and Beta Ltd (Client).",
        ),
        EvalSample(
            question="What is the governing law?",
            expected_answer="The agreement is governed by the laws of New York.",
            context_chunks=[
                "Section 12: Termination clause applies after 30 days notice.",
            ],
            generated_answer="I could not find specific information about governing law.",
        ),
    ]

    output = evaluate_dataset(samples)
    avgs = output["averages"]

    lines = ["## 📊 Evaluation Report (Demo Dataset)\n"]
    for r in output["results"]:
        lines.append(f"**Q:** {r.question}")
        lines.append(f"- Answer Relevance: `{r.answer_relevance:.3f}`")
        lines.append(f"- Faithfulness:     `{r.faithfulness:.3f}`")
        lines.append(f"- Context Recall:   `{r.context_recall:.3f}`")
        lines.append(f"- **Mean Score:**   `{r.mean_score:.3f}`\n")

    lines.append("---")
    lines.append("### Averages")
    lines.append(f"| Metric | Score |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Answer Relevance | `{avgs['answer_relevance']:.3f}` |")
    lines.append(f"| Faithfulness     | `{avgs['faithfulness']:.3f}` |")
    lines.append(f"| Context Recall   | `{avgs['context_recall']:.3f}` |")
    lines.append(f"| **Mean Score**   | `{avgs['mean_score']:.3f}` |")

    return "\n".join(lines)


# ─── Gradio UI ────────────────────────────────────────────────────────────────

CSS = """
#chatbot { height: 500px; overflow-y: auto; }
.sources-box { font-size: 0.85em; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Smart Contract Assistant",
    theme=gr.themes.Soft(primary_hue="blue"),
    css=CSS,
) as demo:

    gr.Markdown(
        """
        # 📜 Smart Contract Assistant
        Upload a PDF or DOCX legal document, then ask questions about it in natural language.
        Powered by **LangChain + Local Llama (Ollama)** — all processing stays on your machine.
        """
    )

    with gr.Tabs():

        # ── Tab 1: Upload & Ingest ────────────────────────────────────────────
        with gr.Tab("📄 Upload & Ingest"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_input = gr.File(
                        label="Upload PDF or DOCX",
                        file_types=[".pdf", ".docx", ".doc"],
                        type="filepath",
                    )
                    ingest_btn = gr.Button("🚀 Ingest Document", variant="primary")
                    summarize_btn = gr.Button("📋 Generate Summary", variant="secondary")

                with gr.Column(scale=2):
                    ingest_status = gr.Markdown("*Upload a file and click Ingest.*")
                    summary_output = gr.Markdown(label="Summary")

            # Wire up
            ingest_btn.click(
                fn=ingest_file,
                inputs=[upload_input],
                outputs=[ingest_status, summarize_btn],
            )
            summarize_btn.click(
                fn=run_summarize,
                inputs=[],
                outputs=[summary_output],
            )

        # ── Tab 2: Chat ───────────────────────────────────────────────────────
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        elem_id="chatbot",
                        bubble_full_width=False,
                        
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Your question",
                            placeholder="e.g. What are the payment terms?",
                            scale=4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat")
                        guardrails_toggle = gr.Checkbox(
                            label="Enable Guardrails",
                            value=True,
                        )

                with gr.Column(scale=1, elem_classes=["sources-box"]):
                    sources_display = gr.Markdown(
                        label="📚 Sources",
                        value="*Sources will appear here after your first question.*",
                    )

            # Wire up
            def _send(message, history, guardrails):
                return chat(message, history, guardrails)

            send_btn.click(
                fn=_send,
                inputs=[msg_input, chatbot, guardrails_toggle],
                outputs=[chatbot, sources_display],
            ).then(lambda: "", outputs=msg_input)

            msg_input.submit(
                fn=_send,
                inputs=[msg_input, chatbot, guardrails_toggle],
                outputs=[chatbot, sources_display],
            ).then(lambda: "", outputs=msg_input)

            clear_btn.click(
                fn=clear_chat,
                outputs=[chatbot, sources_display],
            )

        # ── Tab 3: Evaluation ─────────────────────────────────────────────────
        with gr.Tab("📊 Evaluation"):
            gr.Markdown(
                """
                ### RAG Quality Evaluation
                Measures three metrics on a built-in demo dataset:
                - **Answer Relevance** — Does the answer address the question?
                - **Faithfulness** — Is the answer grounded in the retrieved context?
                - **Context Recall** — Do the retrieved chunks cover the expected answer?

                All metrics use local embeddings (cosine similarity) — no API key needed.
                """
            )
            eval_btn = gr.Button("▶️ Run Evaluation Demo", variant="primary")
            eval_output = gr.Markdown("*Click the button above to run the evaluation.*")

            eval_btn.click(
                fn=run_evaluation_demo,
                outputs=[eval_output],
            )

# ─── Launch ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )
