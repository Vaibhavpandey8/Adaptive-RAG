"""
LangChain 2-Stage Hybrid Query Complexity Router.

Stage 1 (<2ms, ₹0): Fast regex heuristic + BM25 lexical overlap probe.
Stage 2 (LCEL Structured Output): ChatPromptTemplate | ChatGoogleGenerativeAI.with_structured_output(RouteDecision).
Stage 3 (Safety Fallback): Low-confidence (<0.70) or document-anchor override -> Route B.
"""

import re
import time
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from backend.config import ROUTER_CONFIDENCE_THRESHOLD, get_llm
from backend.retriever import HybridRetriever

GREETING_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|good morning|good evening|thanks|thank you|ok|bye|who are you|what can you do)[\s!?.]*$",
    re.IGNORECASE,
)

MULTI_HOP_CUES = [
    "compare", "comparison", "difference between", "versus", " vs ",
    "both", "and also", "what happens if", "which policies apply",
    "combine", "across", "multiple", "step by step", "if a ", "suppose ",
]

ENTERPRISE_ANCHORS = (
    "novatech", "company policy", "our policy", "this policy",
    "uploaded document", "uploaded doc", "this pdf", "in the document", "my resume",
)


class RouteDecision(BaseModel):
    """Structured Pydantic output schema for LangChain router chain."""
    route: Literal["A", "B", "C"] = Field(description="Selected route: A (Direct LLM), B (Single-Hop), or C (Multi-Hop)")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reason: str = Field(description="Concise 1-sentence justification for choosing this route")


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a strict Query Complexity Router for an Adaptive-RAG system.\n"
        "Given the user query, indexed documents, and a fast BM25 keyword overlap hint, classify into ONE route:\n\n"
        "- 'A' (Direct LLM / No Retrieval): General knowledge, coding/ML concepts, math, or definitions unrelated "
        "to the indexed documents (e.g., 'What is cosine similarity?', 'Explain precision vs recall'). "
        "NEVER choose 'A' if the query asks about the uploaded documents or company policies — route those to 'B' "
        "so retrieval guardrails can verify or safely abstain.\n\n"
        "- 'B' (Single-Hop RAG): A focused single-question factual lookup from the indexed documents "
        "(e.g., 'What is the annual learning budget for an L4 AI Engineer?', 'What is the SEV-1 SLA?', "
        "or checking a single fact in an uploaded PDF).\n\n"
        "- 'C' (Multi-Hop RAG): ONLY for genuinely multi-part scenarios, comparisons, or compound questions "
        "requiring combining facts across 2+ distinct sections or documents. Do NOT use 'C' for single-fact lookups."
    ),
    (
        "human",
        "Indexed Documents:\n{manifest}\n\n"
        "Fast BM25 Overlap Hint: {bm25_signal} (max_score={bm25_score}, top_sources={top_sources})\n"
        "User Query: {query}"
    ),
])


class TwoStageHybridRouter:
    """LangChain LCEL 2-stage router with document awareness and asymmetric safety fallback."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.router_chain = ROUTER_PROMPT | get_llm(temperature=0.0, max_tokens=180).with_structured_output(RouteDecision)

    def _detect_signals(self, query: str, probe: Dict) -> List[str]:
        signals: List[str] = [f"BM25 Probe: {probe['signal'].upper()} ({probe['max_score']})"]
        q_lower = query.lower()
        if probe["top_sources"]:
            signals.append(f"Matched Docs: {', '.join(probe['top_sources'][:2])}")
        if any(cue in q_lower for cue in MULTI_HOP_CUES) or query.count("?") >= 2 or len(query.split()) >= 22:
            signals.append("Multi-Hop Linguistic Cue")
        return signals

    def route(self, query: str, force_route: Optional[str] = None) -> Dict:
        start = time.perf_counter()
        probe = self.retriever.quick_bm25_probe(query)
        signals = self._detect_signals(query, probe)

        if force_route and force_route.upper() in ("A", "B", "C"):
            chosen = force_route.upper()
            return {
                "route": chosen,
                "original_route": chosen,
                "confidence": 1.0,
                "decision_source": "manual_override",
                "fallback_triggered": False,
                "reason": f"Manual override selected (Route {chosen}).",
                "signals": signals,
                "bm25_probe": probe,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }

        # Stage 1: Instant Heuristic Fast-Path
        if GREETING_PATTERNS.match(query.strip()):
            return {
                "route": "A",
                "original_route": "A",
                "confidence": 0.99,
                "decision_source": "heuristic_fast_path",
                "fallback_triggered": False,
                "reason": "Greeting/chitchat caught by Stage 1 heuristic (0 LLM router calls).",
                "signals": ["Stage 1 Fast-Path"],
                "bm25_probe": probe,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }

        # Empty knowledge base fast-path
        if not self.retriever.documents:
            doc_request = any(
                w in query.lower()
                for w in ("uploaded", "document", "this pdf", "my file", "in the doc", "my policy", "my resume")
            )
            chosen = "B" if doc_request else "A"
            return {
                "route": chosen,
                "original_route": chosen,
                "confidence": 0.95,
                "decision_source": "heuristic_fast_path",
                "fallback_triggered": False,
                "reason": (
                    "User referenced uploaded documents, but none are indexed yet."
                    if doc_request
                    else "No documents uploaded yet — answering directly with LLM (Route A)."
                ),
                "signals": ["Empty Knowledge Base"],
                "bm25_probe": probe,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }

        # Stage 2: LangChain Structured Output Router Chain
        manifest_lines = [
            f"- {d['filename']} ({d['title']}): sections [{', '.join(d.get('sections', [])[:4])}]"
            for d in self.retriever.manifest
        ]
        manifest_text = "\n".join(manifest_lines)

        predicted_route = "B"
        confidence = 0.65
        reason = "Defaulted to Single-Hop RAG."
        decision_source = "llm_router"

        try:
            decision: RouteDecision = self.router_chain.invoke({
                "manifest": manifest_text,
                "bm25_signal": probe["signal"],
                "bm25_score": probe["max_score"],
                "top_sources": probe["top_sources"],
                "query": query,
            })
            predicted_route = decision.route
            confidence = max(0.0, min(1.0, float(decision.confidence)))
            reason = decision.reason.strip()
        except Exception as exc:
            predicted_route = "B"
            confidence = 0.50
            decision_source = "error_fallback"
            reason = f"Router fallback to Route B ({type(exc).__name__})."

        # Stage 3: Asymmetric Safety Fallback
        final_route = predicted_route
        fallback_triggered = False
        has_anchor = any(a in query.lower() for a in ENTERPRISE_ANCHORS)

        if has_anchor and predicted_route == "A":
            final_route = "B"
            fallback_triggered = True
            decision_source = "enterprise_guardrail_override"
            reason = "Query references uploaded/company documents — overrode Route A to Route B for grounded verification."
        elif confidence < ROUTER_CONFIDENCE_THRESHOLD and predicted_route != "B":
            final_route = "B"
            fallback_triggered = True
            decision_source = "low_confidence_fallback"
            reason = (
                f"Low confidence ({confidence:.2f} < {ROUTER_CONFIDENCE_THRESHOLD}) on Route {predicted_route}. "
                "Fell back to Route B (Single-Hop RAG) to avoid hallucination."
            )

        return {
            "route": final_route,
            "original_route": predicted_route,
            "confidence": round(confidence, 2),
            "decision_source": decision_source,
            "fallback_triggered": fallback_triggered,
            "reason": reason,
            "signals": signals,
            "bm25_probe": probe,
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }
