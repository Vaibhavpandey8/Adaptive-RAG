"""
Configuration for LangChain Adaptive-RAG Application.
Uses LangChain Google GenAI (ChatGoogleGenerativeAI + GoogleGenerativeAIEmbeddings).
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "data" / "uploads"
STORAGE_DIR = BASE_DIR / "storage"
FRONTEND_DIR = BASE_DIR / "frontend"

for d in (UPLOADS_DIR, STORAGE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Load .env
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001").strip()

# Chunking Hyperparameters (RecursiveCharacterTextSplitter)
CHUNK_SIZE = 550
CHUNK_OVERLAP = 90

# Retrieval Hyperparameters
TOP_K_SINGLE = 4
TOP_K_MULTI_HOP = 3
MAX_SUB_QUERIES = 3
RRF_K = 60

# Router & Guardrail Thresholds
ROUTER_CONFIDENCE_THRESHOLD = 0.70
MIN_VECTOR_SIMILARITY = 0.68
MIN_BM25_SCORE = 4.0

_embeddings_cache = None


def get_llm(temperature: float = 0.0, max_tokens: int = 1024) -> ChatGoogleGenerativeAI:
    """Returns a LangChain ChatGoogleGenerativeAI instance."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing in .env")
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Returns a singleton LangChain GoogleGenerativeAIEmbeddings instance."""
    global _embeddings_cache
    if _embeddings_cache is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing in .env")
        _embeddings_cache = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GEMINI_API_KEY,
        )
    return _embeddings_cache
