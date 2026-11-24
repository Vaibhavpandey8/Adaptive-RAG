"""
LangChain Hybrid Retriever:
  - Sparse Search: BM25Okapi (rank_bm25) over LangChain Documents
  - Dense Search: LangChain InMemoryVectorStore + GoogleGenerativeAIEmbeddings (persisted to disk)
  - Fusion: Reciprocal Rank Fusion (RRF): 1/(60 + rank_bm25) + 1/(60 + rank_dense)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from backend.config import EMBEDDING_MODEL, RRF_K, STORAGE_DIR, TOP_K_SINGLE, get_embeddings

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these", "those",
    "it", "its", "what", "which", "who", "whom", "how", "when", "where", "why",
    "do", "does", "did", "can", "could", "should", "would", "will", "shall",
    "have", "has", "had", "my", "our", "your", "their", "i", "we", "you", "they",
}


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"\b[a-z0-9_]+\b", text.lower())
    filtered = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return filtered if filtered else tokens


def doc_to_dict(doc: Document) -> Dict:
    return {
        "chunk_id": doc.metadata.get("chunk_id", ""),
        "source": doc.metadata.get("source", ""),
        "title": doc.metadata.get("title", ""),
        "section": doc.metadata.get("section", "Overview"),
        "content": doc.page_content,
    }


class HybridRetriever:
    """
    Combines LangChain InMemoryVectorStore (dense semantic search) and BM25Okapi (sparse keyword search)
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, cache_prefix: str = "lc_chat"):
        self.documents: List[Document] = []
        self.chunks: List[Dict] = []
        self.manifest: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.vector_store: Optional[InMemoryVectorStore] = None
        self.doc_by_id: Dict[str, Document] = {}
        self.cache_meta: Path = STORAGE_DIR / f"{cache_prefix}_meta.json"
        self.cache_store: Path = STORAGE_DIR / f"{cache_prefix}_vectorstore.json"

    def build(self, documents: List[Document], manifest: List[Dict], corpus_hash: str):
        self.documents = documents
        self.chunks = [doc_to_dict(d) for d in documents]
        self.manifest = manifest
        self.doc_by_id = {d.metadata["chunk_id"]: d for d in documents}

        if not documents:
            self.bm25 = None
            self.vector_store = None
            return

        # 1. Build BM25Okapi sparse index
        tokenized_corpus = [
            tokenize(f"{d.metadata.get('title', '')} {d.metadata.get('section', '')} {d.page_content}")
            for d in documents
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # 2. Load cached LangChain InMemoryVectorStore or embed & dump
        embeddings = get_embeddings()
        if self.cache_meta.exists() and self.cache_store.exists():
            try:
                meta = json.loads(self.cache_meta.read_text(encoding="utf-8"))
                if (
                    meta.get("corpus_hash") == corpus_hash
                    and meta.get("embedding_model") == EMBEDDING_MODEL
                    and meta.get("num_docs") == len(documents)
                ):
                    self.vector_store = InMemoryVectorStore.load(str(self.cache_store), embedding=embeddings)
                    return
            except Exception:
                pass

        store = InMemoryVectorStore(embedding=embeddings)
        store.add_documents(documents)
        store.dump(str(self.cache_store))
        self.cache_meta.write_text(
            json.dumps(
                {
                    "corpus_hash": corpus_hash,
                    "embedding_model": EMBEDDING_MODEL,
                    "num_docs": len(documents),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self.vector_store = store

    def _bm25_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.bm25 or not self.documents:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.documents[i].metadata["chunk_id"], float(s)) for i, s in ranked]

    def _vector_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if not self.vector_store or not self.documents:
            return []
        hits = self.vector_store.similarity_search_with_score(query, k=top_k)
        return [(doc.metadata["chunk_id"], float(score)) for doc, score in hits]

    def quick_bm25_probe(self, query: str) -> Dict:
        """Ultra-fast (<2ms) BM25 lexical overlap probe for Stage 1 of the Router."""
        hits = self._bm25_search(query, top_k=3)
        if not hits:
            return {"max_score": 0.0, "signal": "low", "top_sources": []}

        max_score = hits[0][1]
        top_sources = list(
            dict.fromkeys(
                self.doc_by_id[cid].metadata["source"]
                for cid, score in hits
                if score > 0.5 and cid in self.doc_by_id
            )
        )
        signal = "high" if max_score >= 4.5 else ("medium" if max_score >= 1.8 else "low")
        return {
            "max_score": round(max_score, 2),
            "signal": signal,
            "top_sources": top_sources,
        }

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_SINGLE,
        mode: str = "hybrid",
    ) -> List[Dict]:
        if not self.documents:
            return []

        pool_k = min(max(top_k * 3, 10), len(self.documents))
        bm25_hits = self._bm25_search(query, top_k=pool_k)
        bm25_scores = {cid: score for cid, score in bm25_hits}
        bm25_ranks = {cid: rank for rank, (cid, _) in enumerate(bm25_hits, start=1)}

        if mode == "bm25":
            return [
                {
                    **doc_to_dict(self.doc_by_id[cid]),
                    "bm25_score": round(score, 3),
                    "vector_score": 0.0,
                    "rrf_score": round(score, 4),
                    "retrieval_mode": "bm25",
                }
                for cid, score in bm25_hits[:top_k]
                if cid in self.doc_by_id
            ]

        vec_hits = self._vector_search(query, top_k=pool_k)
        vec_scores = {cid: score for cid, score in vec_hits}
        vec_ranks = {cid: rank for rank, (cid, _) in enumerate(vec_hits, start=1)}

        if mode == "vector":
            return [
                {
                    **doc_to_dict(self.doc_by_id[cid]),
                    "bm25_score": round(bm25_scores.get(cid, 0.0), 3),
                    "vector_score": round(score, 4),
                    "rrf_score": round(score, 4),
                    "retrieval_mode": "vector",
                }
                for cid, score in vec_hits[:top_k]
                if cid in self.doc_by_id
            ]

        # Hybrid Reciprocal Rank Fusion (RRF)
        all_ids = set(bm25_ranks.keys()) | set(vec_ranks.keys())
        fused: List[Tuple[str, float]] = []
        for cid in all_ids:
            rrf = 0.0
            if cid in bm25_ranks and bm25_scores.get(cid, 0.0) > 0:
                rrf += 1.0 / (RRF_K + bm25_ranks[cid])
            if cid in vec_ranks:
                rrf += 1.0 / (RRF_K + vec_ranks[cid])
            fused.append((cid, rrf))

        fused.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                **doc_to_dict(self.doc_by_id[cid]),
                "bm25_score": round(bm25_scores.get(cid, 0.0), 3),
                "vector_score": round(vec_scores.get(cid, 0.0), 4),
                "rrf_score": round(rrf_score, 5),
                "retrieval_mode": "hybrid_rrf",
            }
            for cid, rrf_score in fused[:top_k]
            if cid in self.doc_by_id
        ]
