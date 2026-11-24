# Adaptive-RAG Chatbot (LangChain LCEL + FastAPI)

Clean **Adaptive Retrieval-Augmented Generation** web app built with **LangChain LCEL**, **FastAPI**, **Google Gemini**, and **HTML/CSS/JS** (Light & Dark mode).

## Run Locally

```powershell
cd V:\MTECH\PLACEMENT\AdaptiveRAG
python run.py
```

Open **http://127.0.0.1:8000** (`/`, `/benchmark`, `/architecture`).

## LangChain Architecture (6 Clean Modules)

1. `backend/config.py` — `ChatGoogleGenerativeAI` + `GoogleGenerativeAIEmbeddings`.
2. `backend/ingestion.py` — Loads user-uploaded PDF/MD/TXT into LangChain `Document` objects and chunks with `RecursiveCharacterTextSplitter`.
3. `backend/retriever.py` — Hybrid search combining `BM25Okapi` (sparse) + persisted LangChain `InMemoryVectorStore` (dense) via **Reciprocal Rank Fusion (RRF)**.
4. `backend/router.py` — 2-Stage Hybrid Router using fast heuristic/BM25 probe + LangChain `ChatPromptTemplate | llm.with_structured_output(RouteDecision)` + low-confidence fallback.
5. `backend/pipeline.py` — LCEL chains for Conversational Follow-up Rewriting, Route A (Direct LLM), Route B (Single-Hop + CRAG Reranker & Retry + Abstention Guardrail), and Route C (Multi-Hop Decomposition & Synthesis).
6. `backend/evaluator.py` — Self-contained 9-question benchmark comparing No-RAG vs Naive Single-Hop RAG vs Adaptive RAG.
