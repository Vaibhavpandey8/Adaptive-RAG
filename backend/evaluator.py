"""
Self-Contained LangChain 9-Question Benchmark Evaluator for Adaptive-RAG.
Uses an isolated in-memory LangChain HybridRetriever (separate from Chat uploads)
so 'Run Live Benchmark' ALWAYS works even when the Chat sidebar has 0 uploaded docs.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import time
from typing import Dict, List, Optional
from langchain_core.documents import Document

from backend.config import STORAGE_DIR
from backend.ingestion import split_text_to_documents
from backend.pipeline import AdaptiveRAGPipeline
from backend.retriever import HybridRetriever

EVAL_CACHE_FILE = STORAGE_DIR / "last_eval.json"

BENCHMARK_DOCS: List[Dict[str, str]] = [
    {
        "source": "hr_leave_and_benefits_policy.md",
        "title": "NovaTech HR Leave, Hybrid Work & Benefits Policy",
        "content": (
            "# NovaTech HR Leave, Hybrid Work & Benefits Policy\n\n"
            "## 1. Hybrid Work & WFA\n"
            "NovaTech follows a 3+2 Hybrid Work Model (Tue-Thu in office, Mon & Fri remote). "
            "Employees who have completed their 90-day probation may work remotely from anywhere in India "
            "for up to 3 weeks per calendar year (Work-From-Anywhere weeks) with manager approval.\n\n"
            "## 2. Learning & Certification Budget\n"
            "L3 and L4 (AI / Software Engineer) employees receive up to ₹60,000 per financial year for approved "
            "courses, certifications, and conference tickets. L5 and L6 engineers receive up to ₹1,00,000 per year. "
            "Any single learning expense above ₹25,000 requires Engineering Manager plus VP of Engineering approval.\n\n"
            "## 3. Notice Period\n"
            "Standard notice period after confirmation is 60 calendar days for L3-L4 employees and 90 calendar days "
            "for L5+ roles. During the 90-day probation period, notice period is 15 calendar days from either side."
        ),
    },
    {
        "source": "engineering_security_and_ai_policy.md",
        "title": "NovaTech Engineering Security, Code Review & AI Policy",
        "content": (
            "# NovaTech Engineering Security, Code Review & AI Policy\n\n"
            "## 1. Code Review Approvals\n"
            "PR review SLA is 24 business hours. Standard services require 1 approval from an L4+ peer. "
            "Core production APIs, authentication services, and ML inference pipelines require a minimum of "
            "2 approvals including at least 1 CODEOWNER (L5+).\n\n"
            "## 2. Incident Response SLAs\n"
            "SEV-1 critical outages require acknowledgement within 15 minutes and a mandatory Blameless "
            "Post-Mortem (RCA) within 48 hours. SEV-2 requires acknowledgement within 1 hour.\n\n"
            "## 3. RAG & Model Production Gate\n"
            "Before deploying any customer-facing RAG or LLM feature to production, teams must report "
            "Faithfulness >= 0.85, p95 latency <= 2.5 seconds, and hallucination guardrail pass rate >= 95%."
        ),
    },
    {
        "source": "expense_and_travel_reimbursement_policy.md",
        "title": "NovaTech Expense, Home-Office & Travel Policy",
        "content": (
            "# NovaTech Expense, Home-Office & Travel Policy\n\n"
            "## 1. Home-Office Setup Allowance\n"
            "New hires receive a one-time ₹35,000 Home-Office Setup Allowance (monitors, chairs, keyboards), "
            "eligible after completing 30 calendar days from joining (Week 5 onwards). "
            "Clawback clause: if an employee resigns within 12 months of reimbursement, they must either "
            "return the reimbursed hardware in working condition OR repay 50% (if leaving between 6-12 months) "
            "or 100% (if leaving within 6 months).\n\n"
            "## 2. Submission & Approval Tiers\n"
            "All travel and expense claims must be submitted on FinDesk within 15 calendar days with GST invoices. "
            "Expenses up to ₹25,000 require Reporting Manager approval. Expenses from ₹25,001 to ₹1,00,000 "
            "require Reporting Manager plus VP of Engineering approval."
        ),
    },
]

GOLDEN_BENCHMARK: List[Dict] = [
    {
        "id": "Q1",
        "query": "What is the difference between precision and recall in machine learning?",
        "gold_route": "A",
        "expected_keywords": ["precision", "recall", "false positive", "false negative"],
    },
    {
        "id": "Q2",
        "query": "Explain cosine similarity in one short paragraph.",
        "gold_route": "A",
        "expected_keywords": ["angle", "vector", "dot product", "similarity"],
    },
    {
        "id": "Q3",
        "query": "Hi, what can you help me with?",
        "gold_route": "A",
        "expected_keywords": ["help", "question", "assist"],
    },
    {
        "id": "Q4",
        "query": "What is the annual learning and certification reimbursement budget for an L4 AI Engineer?",
        "gold_route": "B",
        "expected_keywords": ["60,000", "L4"],
    },
    {
        "id": "Q5",
        "query": "What is NovaTech's SEV-1 incident acknowledgement SLA and post-mortem deadline?",
        "gold_route": "B",
        "expected_keywords": ["15 minutes", "48 hours"],
    },
    {
        "id": "Q6",
        "query": "Within how many calendar days must travel expenses be submitted on FinDesk?",
        "gold_route": "B",
        "expected_keywords": ["15", "FinDesk"],
    },
    {
        "id": "Q7",
        "query": (
            "If a newly joined L4 AI Engineer buys a 30,000 INR monitor in week 6 and attends an AI "
            "conference costing 45,000 INR, who must approve the expenses and what happens if they resign after 8 months?"
        ),
        "gold_route": "C",
        "expected_keywords": ["35,000", "VP", "50%", "return"],
    },
    {
        "id": "Q8",
        "query": (
            "Compare the notice period during probation vs after probation confirmation for an L4 engineer, "
            "and also state when they become eligible for Work-From-Anywhere weeks."
        ),
        "gold_route": "C",
        "expected_keywords": ["15", "60", "90", "3 weeks"],
    },
    {
        "id": "Q9",
        "query": (
            "What code review approvals are required for a core ML inference pipeline PR, and what RAG evaluation "
            "metrics must pass before production launch?"
        ),
        "gold_route": "C",
        "expected_keywords": ["2 approvals", "CODEOWNER", "0.85", "2.5"],
    },
]

_bench_pipeline: Optional[AdaptiveRAGPipeline] = None


def _get_benchmark_pipeline() -> AdaptiveRAGPipeline:
    global _bench_pipeline
    if _bench_pipeline is not None:
        return _bench_pipeline

    all_docs: List[Document] = []
    manifest: List[Dict] = []
    hasher = hashlib.sha256()

    for doc in BENCHMARK_DOCS:
        doc_chunks = split_text_to_documents(doc["content"], source_name=doc["source"], title=doc["title"])
        sections = list(dict.fromkeys(d.metadata["section"] for d in doc_chunks))
        hasher.update(doc["source"].encode("utf-8"))
        hasher.update(doc["content"].encode("utf-8"))
        all_docs.extend(doc_chunks)
        manifest.append({
            "filename": doc["source"],
            "title": doc["title"],
            "is_sample": True,
            "chunk_count": len(doc_chunks),
            "char_count": len(doc["content"]),
            "sections": sections[:6],
        })

    bench_retriever = HybridRetriever(cache_prefix="lc_bench")
    bench_retriever.build(all_docs, manifest, hasher.hexdigest())
    _bench_pipeline = AdaptiveRAGPipeline(bench_retriever)
    return _bench_pipeline


def keyword_recall_score(answer: str, expected_keywords: List[str]) -> float:
    ans_lower = answer.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in ans_lower)
    return round((hits / max(len(expected_keywords), 1)) * 100.0, 1)


def load_cached_eval() -> Dict:
    if EVAL_CACHE_FILE.exists():
        try:
            return json.loads(EVAL_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"has_results": False}


def run_benchmark(retriever_mode: str = "hybrid") -> Dict:
    started = time.time()
    bench_pipeline = _get_benchmark_pipeline()

    configs = [
        ("No-RAG (Always A)", "A"),
        ("Naive RAG (Always B)", "B"),
        ("Adaptive RAG (Ours)", None),
    ]

    def _eval_one(task):
        label, forced, idx, item = task
        res = bench_pipeline.run(
            query=item["query"],
            history=[],
            force_route=forced,
            retriever_mode=retriever_mode,
        )
        score = keyword_recall_score(res["answer"], item["expected_keywords"])
        return label, forced, idx, item, res, score

    tasks = [
        (label, forced, idx, item)
        for label, forced in configs
        for idx, item in enumerate(GOLDEN_BENCHMARK)
    ]

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(_eval_one, tasks))

    per_question = [
        {
            "id": item["id"],
            "query": item["query"],
            "gold_route": item["gold_route"],
            "scores": {},
            "predicted_route": None,
            "route_correct": None,
        }
        for item in GOLDEN_BENCHMARK
    ]

    agg = {
        label: {"scores": [], "latencies": [], "steps": [], "route_hits": 0, "forced": forced}
        for label, forced in configs
    }

    for label, forced, idx, item, res, score in results:
        agg[label]["scores"].append(score)
        agg[label]["latencies"].append(res["metrics"]["total_ms"])
        agg[label]["steps"].append(res["retrieval_steps"])

        per_question[idx]["scores"][label] = {
            "accuracy": score,
            "latency_ms": res["metrics"]["total_ms"],
            "steps": res["retrieval_steps"],
        }

        if forced is None:
            ok = res["route"] == item["gold_route"]
            if ok:
                agg[label]["route_hits"] += 1
            per_question[idx]["predicted_route"] = res["route"]
            per_question[idx]["route_correct"] = ok

    summary = []
    for label, forced in configs:
        a = agg[label]
        n = max(len(a["scores"]), 1)
        summary.append({
            "method": label,
            "avg_accuracy": round(sum(a["scores"]) / n, 1),
            "avg_latency_ms": round(sum(a["latencies"]) / n, 1),
            "avg_retrieval_steps": round(sum(a["steps"]) / n, 2),
            "route_accuracy": round((a["route_hits"] / len(GOLDEN_BENCHMARK)) * 100.0, 1) if forced is None else None,
        })

    report = {
        "has_results": True,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retriever_mode": retriever_mode,
        "num_questions": len(GOLDEN_BENCHMARK),
        "elapsed_sec": round(time.time() - started, 1),
        "summary": summary,
        "per_question": per_question,
    }
    EVAL_CACHE_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
