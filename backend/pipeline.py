"""
LangChain LCEL Adaptive-RAG Execution Pipeline:
  1. Conversational Query Rewriter Chain (ChatPromptTemplate | LLM.with_structured_output)
  2. 2-Stage Hybrid Router (Route A / Route B / Route C)
  3. Hybrid Search (BM25Okapi + LangChain InMemoryVectorStore + RRF)
  4. Corrective Reranker Chain (CRAG) + 1 Auto-Retry + Safe Abstention Guardrail
  5. Multi-Hop Decomposition & Synthesis Chains (Route C)
"""

import time
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from backend.config import (
    MAX_SUB_QUERIES,
    MIN_BM25_SCORE,
    MIN_VECTOR_SIMILARITY,
    TOP_K_MULTI_HOP,
    TOP_K_SINGLE,
    get_llm,
)
from backend.retriever import HybridRetriever
from backend.router import TwoStageHybridRouter

STRATEGY_LABELS = {
    "A": "Direct LLM",
    "B": "Single-Hop RAG",
    "C": "Multi-Hop RAG",
}


# ---------------------------------------------------------------------------
# Pydantic Schemas for LangChain Structured Outputs
# ---------------------------------------------------------------------------
class StandaloneQuery(BaseModel):
    standalone_query: str = Field(description="Rewritten self-contained query")
    was_rewritten: bool = Field(description="True only if history was needed to resolve follow-up references")


class ChunkScore(BaseModel):
    id: int = Field(description="Zero-based candidate chunk ID")
    score: float = Field(ge=0.0, le=10.0, description="Relevance score from 0 to 10")


class RerankResult(BaseModel):
    scores: List[ChunkScore]


class SubQueries(BaseModel):
    sub_queries: List[str] = Field(description="2 to 3 concise independent search sub-queries")


# ---------------------------------------------------------------------------
# LangChain ChatPromptTemplates (LCEL)
# ---------------------------------------------------------------------------
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given recent conversation history and the latest user message, rewrite the latest message "
        "into a clear standalone question ONLY if it depends on prior turns (e.g., 'it', 'they', 'what about L5'). "
        "If already standalone, return it unchanged with was_rewritten=false."
    ),
    ("human", "History:\n{history}\n\nLatest User Message: {query}"),
])

RERANK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict RAG passage reranker. Score each candidate chunk from 0 to 10 on how directly "
        "it helps answer the user's question (0 = completely irrelevant, 10 = directly answers it)."
    ),
    ("human", "Question: {query}\n\nCandidate Passages:\n{passages}"),
])

REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's search query using alternative keywords and synonyms to improve "
        "document retrieval recall. Return ONLY the rewritten search query."
    ),
    ("human", "{query}"),
])

DIRECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a concise, helpful AI assistant. Answer clearly in 3-5 sentences using general knowledge."),
    ("human", "{query}"),
])

SINGLE_HOP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a precise document assistant. Answer ONLY using the provided context.\n"
        "Rules:\n"
        "1. Cite exact source filenames inline like [filename.pdf].\n"
        "2. If the context does NOT answer the question, start your reply with '[INSUFFICIENT_CONTEXT]' "
        "followed by one polite sentence stating the uploaded documents do not cover it. Never invent facts.\n"
        "3. Keep the response clean, direct, and easy to read."
    ),
    ("human", "Context:\n{context}\n\nQuestion: {query}\nAnswer:"),
])

DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        f"Break the complex user question into 2 to {MAX_SUB_QUERIES} concise, independent search sub-queries "
        "needed to gather all facts across uploaded documents."
    ),
    ("human", "{query}"),
])

MULTI_HOP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior document advisor synthesizing evidence retrieved across multiple search hops.\n"
        "Rules:\n"
        "1. Address EVERY part of the user's question clearly using concise bullet points.\n"
        "2. Cite exact source filenames inline like [filename.md].\n"
        "3. Base every rule, threshold, and number strictly on the retrieved context; never invent details."
    ),
    (
        "human",
        "Original Question: {query}\nSub-Queries Explored: {sub_queries}\n\n"
        "Retrieved Context:\n{context}\n\nAnswer:"
    ),
])


class AdaptiveRAGPipeline:
    """Orchestrates LangChain LCEL chains for Adaptive-RAG."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.router = TwoStageHybridRouter(retriever)

        llm_strict = get_llm(temperature=0.0, max_tokens=1024)
        llm_fast = get_llm(temperature=0.0, max_tokens=260)
        llm_chat = get_llm(temperature=0.2, max_tokens=512)
        parser = StrOutputParser()

        # Reusable LangChain LCEL Chains
        self.contextualize_chain = CONTEXTUALIZE_PROMPT | llm_fast.with_structured_output(StandaloneQuery)
        self.rerank_chain = RERANK_PROMPT | llm_fast.with_structured_output(RerankResult)
        self.rewrite_chain = REWRITE_PROMPT | llm_fast | parser
        self.direct_chain = DIRECT_PROMPT | llm_chat | parser
        self.single_hop_chain = SINGLE_HOP_PROMPT | llm_strict | parser
        self.decompose_chain = DECOMPOSE_PROMPT | llm_fast.with_structured_output(SubQueries)
        self.multi_hop_chain = MULTI_HOP_PROMPT | llm_strict | parser

    def _contextualize_query(
        self, query: str, history: List[Dict[str, str]]
    ) -> Tuple[str, bool, float]:
        if not history:
            return query, False, 0.0

        t0 = time.perf_counter()
        hist_text = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')[:300]}"
            for m in history[-6:]
        )
        try:
            res: StandaloneQuery = self.contextualize_chain.invoke({"history": hist_text, "query": query})
            rewritten = res.standalone_query.strip() or query
            was_rewritten = bool(res.was_rewritten) and rewritten.lower() != query.lower()
            return rewritten, was_rewritten, round((time.perf_counter() - t0) * 1000, 1)
        except Exception:
            return query, False, round((time.perf_counter() - t0) * 1000, 1)

    def _rerank_and_grade(
        self, query: str, chunks: List[Dict], keep_k: int
    ) -> Tuple[List[Dict], float]:
        if not chunks:
            return [], 0.0

        t0 = time.perf_counter()
        passages = "\n---\n".join(
            f"ID={i} | Source={c['source']} | Section={c['section']}\n{c['content'][:420]}"
            for i, c in enumerate(chunks)
        )
        try:
            res: RerankResult = self.rerank_chain.invoke({"query": query, "passages": passages})
            score_map = {item.id: float(item.score) for item in res.scores}
            scored = []
            for i, c in enumerate(chunks):
                rel = score_map.get(i, 5.0)
                if rel >= 4.0:
                    scored.append({**c, "rerank_score": round(rel, 1)})
            scored.sort(key=lambda x: (x["rerank_score"], x.get("rrf_score", 0.0)), reverse=True)
            return scored[:keep_k], round((time.perf_counter() - t0) * 1000, 1)
        except Exception:
            return chunks[:keep_k], round((time.perf_counter() - t0) * 1000, 1)

    def _format_context(self, chunks: List[Dict]) -> str:
        return "\n\n".join(
            f"[{i}] Source: {c['source']} | Section: {c['section']}\n{c['content']}"
            for i, c in enumerate(chunks, start=1)
        )

    def _passes_fast_threshold(self, chunks: List[Dict], retriever_mode: str) -> bool:
        if not chunks:
            return False
        best_vec = max(c.get("vector_score", 0.0) for c in chunks)
        best_bm25 = max(c.get("bm25_score", 0.0) for c in chunks)
        if retriever_mode == "bm25":
            return best_bm25 >= MIN_BM25_SCORE
        if retriever_mode == "vector":
            return best_vec >= MIN_VECTOR_SIMILARITY
        return (best_vec >= MIN_VECTOR_SIMILARITY) or (best_bm25 >= MIN_BM25_SCORE * 1.5)

    def _run_route_a(self, query: str) -> Dict:
        t0 = time.perf_counter()
        answer = self.direct_chain.invoke({"query": query}).strip()
        gen_ms = round((time.perf_counter() - t0) * 1000, 1)
        return {
            "answer": answer,
            "retrieved_chunks": [],
            "sub_queries": [],
            "retrieval_steps": 0,
            "crag_retry": False,
            "guardrail_triggered": False,
            "citations": [],
            "retrieval_ms": 0.0,
            "rerank_ms": 0.0,
            "generation_ms": gen_ms,
        }

    def _run_route_b(self, query: str, retriever_mode: str, top_k: int) -> Dict:
        if not self.retriever.documents:
            return {
                "answer": "No documents are uploaded yet. Upload a PDF, TXT, or MD file from the sidebar to search your documents.",
                "retrieved_chunks": [],
                "sub_queries": [],
                "retrieval_steps": 0,
                "crag_retry": False,
                "guardrail_triggered": True,
                "citations": [],
                "retrieval_ms": 0.0,
                "rerank_ms": 0.0,
                "generation_ms": 0.0,
            }

        t_ret = time.perf_counter()
        candidates = self.retriever.retrieve(query, top_k=min(top_k + 2, 7), mode=retriever_mode)
        ret_ms = round((time.perf_counter() - t_ret) * 1000, 1)
        retrieval_steps = 1
        crag_retry = False
        rerank_ms = 0.0

        if self._passes_fast_threshold(candidates, retriever_mode):
            chunks, rerank_ms = self._rerank_and_grade(query, candidates, keep_k=top_k)
        else:
            chunks = []

        # Corrective RAG (CRAG) single auto-retry with rewritten search query
        if not chunks:
            crag_retry = True
            try:
                retry_query = self.rewrite_chain.invoke({"query": query}).strip('" ') or query
            except Exception:
                retry_query = query
            t_ret2 = time.perf_counter()
            retry_candidates = self.retriever.retrieve(retry_query, top_k=min(top_k + 2, 7), mode=retriever_mode)
            ret_ms = round(ret_ms + (time.perf_counter() - t_ret2) * 1000, 1)
            retrieval_steps = 2

            if self._passes_fast_threshold(retry_candidates, retriever_mode):
                chunks, rr2_ms = self._rerank_and_grade(query, retry_candidates, keep_k=top_k)
                rerank_ms = round(rerank_ms + rr2_ms, 1)

        if not chunks:
            return {
                "answer": (
                    "I could not find relevant information in the uploaded documents to answer "
                    "this reliably. Try uploading the relevant file or rephrasing your question."
                ),
                "retrieved_chunks": [],
                "sub_queries": [],
                "retrieval_steps": retrieval_steps,
                "crag_retry": crag_retry,
                "guardrail_triggered": True,
                "citations": [],
                "retrieval_ms": ret_ms,
                "rerank_ms": rerank_ms,
                "generation_ms": 0.0,
            }

        t_gen = time.perf_counter()
        answer = self.single_hop_chain.invoke({
            "context": self._format_context(chunks),
            "query": query,
        }).strip()
        gen_ms = round((time.perf_counter() - t_gen) * 1000, 1)

        guardrail_hit = answer.startswith("[INSUFFICIENT_CONTEXT]")
        if guardrail_hit:
            answer = answer.replace("[INSUFFICIENT_CONTEXT]", "").strip()

        citations = [] if guardrail_hit else list(dict.fromkeys(c["source"] for c in chunks))
        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "sub_queries": [],
            "retrieval_steps": retrieval_steps,
            "crag_retry": crag_retry,
            "guardrail_triggered": guardrail_hit,
            "citations": citations,
            "retrieval_ms": ret_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": gen_ms,
        }

    def _run_route_c(self, query: str, retriever_mode: str) -> Dict:
        if not self.retriever.documents:
            return {
                "answer": "No documents are uploaded yet. Upload one or more PDF, TXT, or MD files from the sidebar to run multi-hop reasoning.",
                "retrieved_chunks": [],
                "sub_queries": [],
                "retrieval_steps": 0,
                "crag_retry": False,
                "guardrail_triggered": True,
                "citations": [],
                "retrieval_ms": 0.0,
                "rerank_ms": 0.0,
                "generation_ms": 0.0,
            }

        t_dec = time.perf_counter()
        try:
            sq_res: SubQueries = self.decompose_chain.invoke({"query": query})
            sub_queries = [s.strip() for s in sq_res.sub_queries if s.strip()][:MAX_SUB_QUERIES] or [query]
        except Exception:
            sub_queries = [query]
        dec_ms = round((time.perf_counter() - t_dec) * 1000, 1)

        t_ret = time.perf_counter()
        merged_by_id: Dict[str, Dict] = {}
        for hop_idx, sq in enumerate(sub_queries, start=1):
            for c in self.retriever.retrieve(sq, top_k=TOP_K_MULTI_HOP, mode=retriever_mode):
                cid = c["chunk_id"]
                tagged = {**c, "matched_hop": hop_idx, "matched_sub_query": sq}
                if cid not in merged_by_id or tagged["rrf_score"] > merged_by_id[cid]["rrf_score"]:
                    merged_by_id[cid] = tagged

        ret_ms = round((time.perf_counter() - t_ret) * 1000, 1)
        candidates = sorted(merged_by_id.values(), key=lambda x: x.get("rrf_score", 0.0), reverse=True)[:8]
        chunks, rerank_ms = self._rerank_and_grade(query, candidates, keep_k=6)

        if not chunks:
            return {
                "answer": (
                    "Even after breaking your question into multiple search steps, the uploaded documents "
                    "do not contain enough evidence to answer it reliably."
                ),
                "retrieved_chunks": [],
                "sub_queries": sub_queries,
                "retrieval_steps": len(sub_queries),
                "crag_retry": False,
                "guardrail_triggered": True,
                "citations": [],
                "retrieval_ms": ret_ms,
                "rerank_ms": rerank_ms,
                "generation_ms": dec_ms,
            }

        t_gen = time.perf_counter()
        answer = self.multi_hop_chain.invoke({
            "query": query,
            "sub_queries": sub_queries,
            "context": self._format_context(chunks),
        }).strip()
        synth_ms = round((time.perf_counter() - t_gen) * 1000, 1)

        citations = list(dict.fromkeys(c["source"] for c in chunks))
        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "sub_queries": sub_queries,
            "retrieval_steps": len(sub_queries),
            "crag_retry": False,
            "guardrail_triggered": False,
            "citations": citations,
            "retrieval_ms": ret_ms,
            "rerank_ms": rerank_ms,
            "generation_ms": round(dec_ms + synth_ms, 1),
        }

    def run(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        force_route: Optional[str] = None,
        retriever_mode: str = "hybrid",
        top_k: int = TOP_K_SINGLE,
    ) -> Dict:
        total_start = time.perf_counter()

        standalone_query, was_rewritten, ctx_ms = self._contextualize_query(query, history or [])
        routing = self.router.route(standalone_query, force_route=force_route)
        chosen_route = routing["route"]

        if chosen_route == "A":
            exec_res = self._run_route_a(standalone_query)
        elif chosen_route == "C":
            exec_res = self._run_route_c(standalone_query, retriever_mode=retriever_mode)
        else:
            exec_res = self._run_route_b(standalone_query, retriever_mode=retriever_mode, top_k=top_k)

        total_ms = round((time.perf_counter() - total_start) * 1000, 1)

        return {
            "query": query,
            "standalone_query": standalone_query,
            "was_rewritten": was_rewritten,
            "route": chosen_route,
            "strategy_label": STRATEGY_LABELS.get(chosen_route, STRATEGY_LABELS["B"]),
            "answer": exec_res["answer"],
            "citations": exec_res["citations"],
            "crag_retry": exec_res["crag_retry"],
            "guardrail_triggered": exec_res["guardrail_triggered"],
            "routing": routing,
            "sub_queries": exec_res["sub_queries"],
            "retrieved_chunks": exec_res["retrieved_chunks"],
            "retrieval_steps": exec_res["retrieval_steps"],
            "metrics": {
                "contextualize_ms": ctx_ms,
                "router_ms": routing["latency_ms"],
                "retrieval_ms": exec_res["retrieval_ms"],
                "rerank_ms": exec_res["rerank_ms"],
                "generation_ms": exec_res["generation_ms"],
                "total_ms": total_ms,
            },
        }
