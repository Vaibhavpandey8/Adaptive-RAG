"""
Simple FastAPI server for Adaptive-RAG Chatbot.
Pages:
  - /              -> Clean Minimal Chatbot
  - /benchmark     -> Separate Benchmark Evaluation Page
  - /architecture  -> Separate Architecture & Routing Page
"""

import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.config import EMBEDDING_MODEL, FRONTEND_DIR, LLM_MODEL, TOP_K_SINGLE, UPLOADS_DIR
from backend.evaluator import load_cached_eval, run_benchmark
from backend.ingestion import load_all_documents
from backend.pipeline import AdaptiveRAGPipeline
from backend.retriever import HybridRetriever

retriever = HybridRetriever()
pipeline = AdaptiveRAGPipeline(retriever)


def refresh_index():
    chunks, manifest, corpus_hash = load_all_documents()
    retriever.build(chunks, manifest, corpus_hash)
    return manifest, len(chunks)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    manifest, num_chunks = refresh_index()
    print(f"[AdaptiveRAG] Ready: {len(manifest)} docs, {num_chunks} chunks.")
    yield


app = FastAPI(title="Adaptive-RAG Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_cache(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path in ("/", "/benchmark", "/architecture"):
        response.headers["Cache-Control"] = "no-store"
    return response


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    history: List[Dict[str, str]] = Field(default_factory=list)
    force_route: Optional[Literal["auto", "A", "B", "C"]] = "auto"
    retriever_mode: Literal["hybrid", "bm25", "vector"] = "hybrid"
    top_k: int = Field(default=TOP_K_SINGLE, ge=1, le=8)


class EvalRequest(BaseModel):
    retriever_mode: Literal["hybrid", "bm25", "vector"] = "hybrid"


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "documents_indexed": len(retriever.manifest),
        "chunks_indexed": len(retriever.chunks),
    }


@app.get("/api/documents")
def list_documents():
    return {
        "documents": retriever.manifest,
        "total_chunks": len(retriever.chunks),
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, and .md files are supported.")

    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", Path(file.filename).name)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    (UPLOADS_DIR / safe_name).write_bytes(content)
    manifest, total_chunks = refresh_index()
    return {"message": f"Indexed {safe_name}", "documents": manifest, "total_chunks": total_chunks}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    target = UPLOADS_DIR / Path(filename).name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Only user-uploaded documents can be deleted.")
    target.unlink()
    manifest, total_chunks = refresh_index()
    return {"message": f"Deleted {target.name}", "documents": manifest, "total_chunks": total_chunks}


@app.post("/api/query")
def query_rag(req: QueryRequest):
    forced = None if req.force_route == "auto" else req.force_route
    return pipeline.run(
        query=req.query.strip(),
        history=req.history,
        force_route=forced,
        retriever_mode=req.retriever_mode,
        top_k=req.top_k,
    )


@app.get("/api/eval/latest")
def get_latest_eval():
    return load_cached_eval()


@app.post("/api/eval/run")
def execute_eval(req: EvalRequest):
    return run_benchmark(retriever_mode=req.retriever_mode)


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_chat():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/benchmark")
def serve_benchmark():
    return FileResponse(str(FRONTEND_DIR / "benchmark.html"))


@app.get("/architecture")
def serve_architecture():
    return FileResponse(str(FRONTEND_DIR / "architecture.html"))
