# =============================================================================
# CONFIGURATION — central place for models, paths, and the knowledge-base sources
# =============================================================================
# Keeping every "magic value" here (instead of scattered through the code) makes
# the system easy to re-point at a different domain, model, or corpus. This is a
# deliberate design choice: the multi-agent *architecture* in agents.py is
# domain-agnostic; only this file and the bundled data know about "finance".
# =============================================================================

import os
from pathlib import Path

from dotenv import load_dotenv

# Load GOOGLE_API_KEY (and any overrides) from a local .env file if present.
load_dotenv()

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"

# The structured dataset analysed by the Data-Analyst agent.
COMPANIES_CSV = DATA_DIR / "companies.csv"

# -----------------------------------------------------------------------------
# Models (defaults match the course reference; override via .env)
# -----------------------------------------------------------------------------
# We keep Gemini Flash as the reasoning model: it is fast, cheap, has a free
# tier, and is the exact family used in the course's reference agent, so our
# results are directly comparable to the baseline.
# Default to gemini-2.0-flash: a fast, tool-capable model with a comparatively
# generous free-tier daily quota. Override via LLM_MODEL in .env. (Note: some
# newer aliases like gemini-flash-latest carry a very small free daily limit.)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.0-flash")

# Client-side pacing to respect free-tier per-minute limits during eval runs.
LLM_REQUESTS_PER_SECOND = float(os.environ.get("LLM_REQUESTS_PER_SECOND", "0.25"))

# Embedding backend for the RAG vector store.
#   "huggingface" (default) — local sentence-transformers model. Free, offline,
#                  unlimited: it decouples retrieval from the Gemini free-tier
#                  embedding quota (only ~100 req/min, ~1k/day), which cannot
#                  embed a multi-hundred-chunk corpus. This makes the whole
#                  project reproducible with nothing but a chat API key.
#   "google"     — Gemini embeddings (matches the course reference). Works for a
#                  small corpus but hits the free-tier quota on a large one.
EMBEDDINGS_BACKEND = os.environ.get("EMBEDDINGS_BACKEND", "huggingface")

# Model id used by the selected backend.
HF_EMBEDDING_MODEL = os.environ.get("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "models/gemini-embedding-001")

# Deterministic by default: temperature=0 makes routing and answers reproducible,
# which matters for the evaluation harness.
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0"))

# -----------------------------------------------------------------------------
# Knowledge-base corpus for the Document-Q&A (RAG) agent
# -----------------------------------------------------------------------------
# Real English Wikipedia pages. We mix major public companies with core finance
# concepts so the RAG agent can answer both "what does Apple do?" and
# "what is a price-earnings ratio?". These slugs map to:
#   https://en.wikipedia.org/wiki/<slug>
WIKI_PAGES = [
    # --- Companies ---
    "Apple_Inc.",
    "Microsoft",
    "Amazon_(company)",
    "Alphabet_Inc.",
    "Nvidia",
    "Tesla,_Inc.",
    "Meta_Platforms",
    # --- Finance concepts ---
    "Market_capitalization",
    "Price–earnings_ratio",
    "Initial_public_offering",
    "Exchange-traded_fund",
]

WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/wiki/"

# -----------------------------------------------------------------------------
# Retrieval / chunking parameters (RAG agent)
# -----------------------------------------------------------------------------
CHUNK_SIZE = 1024        # characters per chunk
CHUNK_OVERLAP = 128      # overlap to preserve context across chunk boundaries
RETRIEVER_K = 4          # number of chunks returned per search

# -----------------------------------------------------------------------------
# Agent routing
# -----------------------------------------------------------------------------
# Canonical names of the specialist agents the supervisor can dispatch to.
AGENT_DOCUMENT_QA = "document_qa"
AGENT_DATA_ANALYST = "data_analyst"
AGENT_WEB_RESEARCH = "web_research"
FINISH = "FINISH"

SPECIALISTS = [AGENT_DOCUMENT_QA, AGENT_DATA_ANALYST, AGENT_WEB_RESEARCH]

# Safety valve: maximum supervisor hops before we force a final answer, so a
# confused router can never loop forever.
MAX_SUPERVISOR_STEPS = 6


def require_api_key() -> None:
    """Fail fast with a friendly message if the Google API key is missing."""
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set.\n"
            "1. Get a free key at https://aistudio.google.com/apikey\n"
            "2. Copy .env.example to .env and paste your key there, or\n"
            "   export GOOGLE_API_KEY=... in your shell."
        )
