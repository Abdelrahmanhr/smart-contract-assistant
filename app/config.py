"""
config.py — Central configuration for the Smart Contract Assistant.
Edit this file to change models, paths, and pipeline parameters.
"""

from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
UPLOAD_DIR = DATA_DIR / "uploads"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ─── LLM (Ollama / local Llama) ───────────────────────────────────────────────
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_MODEL: str = "llama3.2"          # change to "llama3", "mistral", etc.
LLM_TEMPERATURE: float = 0.0            # 0 = deterministic / factual
LLM_MAX_TOKENS: int = 1024

# ─── Embeddings ───────────────────────────────────────────────────────────────
# Local SentenceTransformer model — no API key required
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ─── Chunking ─────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 800           # characters per chunk
CHUNK_OVERLAP: int = 150        # overlap between chunks

# ─── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = 5        # number of chunks returned per query
RETRIEVAL_SCORE_THRESHOLD: float = 0.30   # min similarity score (0–1)

# ─── Guardrails ───────────────────────────────────────────────────────────────
GUARDRAIL_SIMILARITY_THRESHOLD: float = 0.25   # min score to consider grounded
UNSAFE_KEYWORDS: list[str] = [
    "kill", "harm", "illegal", "fraud", "money laundering",
    "drug", "weapon", "exploit",
]

# ─── Summarization ────────────────────────────────────────────────────────────
SUMMARIZATION_CHUNK_SIZE: int = 3000   # larger chunks for summarization
SUMMARIZATION_MAX_TOKENS: int = 512

# ─── Vector Store ─────────────────────────────────────────────────────────────
VECTOR_STORE_COLLECTION: str = "contracts"

# ─── Server ───────────────────────────────────────────────────────────────────
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# ─── UI ───────────────────────────────────────────────────────────────────────
GRADIO_PORT: int = 7860
GRADIO_SHARE: bool = True
