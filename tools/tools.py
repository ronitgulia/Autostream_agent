"""
tools/tools.py — RAG retrieval and lead-capture tools for AutoStream Agent.

Improvements over v1:
  1. ``retrieve_knowledge`` is now **async** — the blocking ChromaDB query is
     offloaded to a thread pool via ``asyncio.to_thread``, keeping the event
     loop free during I/O.

  2. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF)**:
     Both a dense (cosine) ChromaDB search *and* a sparse BM25 search are run
     against the same corpus.  Their ranked lists are fused with the
     parameter-free RRF formula ``score = Σ 1 / (k + rank_i)``.  This gives
     exact keyword hits ("$79", "4K", "30 minutes") the boost they deserve while
     retaining the semantic understanding of embeddings.

  3. Pre-warming: the embedding model is loaded eagerly at module import so the
     first user query is not penalised by a cold-start download.
"""

import asyncio
import csv
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


# ── Paths ──────────────────────────────────────────────────────────────────────

_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"
_CHROMA_PATH = Path(os.getenv("CHROMA_DB_PATH", str(Path(__file__).parent.parent / ".chroma_db")))
_LEADS_CSV = Path(__file__).parent.parent / "leads.csv"
_CSV_HEADERS = ["timestamp", "name", "email", "platform"]

# ── Lazy State Singletons ──────────────────────────────────────────────────────

_KB_CACHE: Optional[dict] = None
_VECTORSTORE_CACHE: Optional[tuple[chromadb.Collection, list[str], list[str]]] = None
_BM25_CACHE: Optional["_BM25Index"] = None
_RAG_LOCK = threading.Lock()


# ── Knowledge Base Loading ─────────────────────────────────────────────────────

def get_kb() -> dict:
    """Return the parsed knowledge base JSON (lazily loaded singleton)."""
    global _KB_CACHE
    if _KB_CACHE is None:
        with open(_KB_PATH, "r", encoding="utf-8") as f:
            _KB_CACHE = json.load(f)
    return _KB_CACHE


def _load_kb() -> dict:
    return get_kb()


def _build_chunks(kb: dict) -> tuple[list[str], list[str]]:
    """Convert the KB JSON into flat text chunks and stable IDs.

    One chunk per logical unit: company overview, each plan, each policy,
    each FAQ.
    """
    docs: list[str] = []
    ids: list[str] = []

    co = kb["company"]
    docs.append(f"About AutoStream: {co['description']} Tagline: {co['tagline']}")
    ids.append("company_overview")

    for plan in kb["plans"]:
        features = ", ".join(plan["features"])
        docs.append(
            f"{plan['name']} costs ${plan['price_monthly']}/month. "
            f"Best for: {plan['best_for']}. "
            f"Features include: {features}."
        )
        ids.append(f"plan_{plan['name'].lower().replace(' ', '_')}")

    for policy in kb["policies"]:
        docs.append(f"{policy['topic']}: {policy['details']}")
        ids.append(f"policy_{policy['topic'].lower().replace(' ', '_')}")

    for i, faq in enumerate(kb["faqs"]):
        docs.append(f"Question: {faq['question']} Answer: {faq['answer']}")
        ids.append(f"faq_{i}")

    return docs, ids


# ── Vector Store (ChromaDB) ────────────────────────────────────────────────────

def get_vectorstore() -> tuple[chromadb.Collection, list[str], list[str]]:
    """Create or load the persistent ChromaDB collection (lazily initialized singleton).

    On first query, chunks and embeds the knowledge base.
    On subsequent calls, reuses the cached collection instantly.
    Thread-safe under concurrent requests.
    """
    global _VECTORSTORE_CACHE
    if _VECTORSTORE_CACHE is not None:
        return _VECTORSTORE_CACHE

    with _RAG_LOCK:
        if _VECTORSTORE_CACHE is not None:
            return _VECTORSTORE_CACHE

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
        collection = client.get_or_create_collection(
            name="autostream_kb",
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

        kb = get_kb()
        docs, ids = _build_chunks(kb)

        if collection.count() == 0:
            collection.add(documents=docs, ids=ids)
            logger.info("Vector store built: %d chunks embedded → %s", len(docs), _CHROMA_PATH)
        else:
            logger.info("Vector store loaded: %d chunks from %s", collection.count(), _CHROMA_PATH)

        # Pre-warm: embed a dummy query to load the model weights into RAM now.
        collection.query(query_texts=["warm up"], n_results=1, include=["documents"])
        logger.info("Embedding model pre-warmed successfully.")

        _VECTORSTORE_CACHE = (collection, docs, ids)
        return _VECTORSTORE_CACHE


def _build_vectorstore():
    return get_vectorstore()


# ── BM25 Sparse Index ──────────────────────────────────────────────────────────

class _BM25Index:
    """Lightweight BM25 index over the knowledge base chunks.

    Built once on first query from the same documents list used by
    ChromaDB, ensuring both retrieval paths operate over an identical corpus.
    """

    def __init__(self, docs: list[str], ids: list[str]) -> None:
        tokenised = [doc.lower().split() for doc in docs]
        self._bm25 = BM25Okapi(tokenised)
        self._docs = docs
        self._ids = ids

    def query(self, text: str, top_k: int) -> list[tuple[str, str, float]]:
        """Return up to *top_k* ``(id, doc, score)`` tuples, highest score first."""
        scores = self._bm25.get_scores(text.lower().split())
        ranked = sorted(
            zip(self._ids, self._docs, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:top_k]


def get_bm25_index() -> _BM25Index:
    """Return the BM25 sparse index (lazily constructed thread-safe singleton)."""
    global _BM25_CACHE
    if _BM25_CACHE is not None:
        return _BM25_CACHE

    with _RAG_LOCK:
        if _BM25_CACHE is not None:
            return _BM25_CACHE
        _, docs, ids = get_vectorstore()
        _BM25_CACHE = _BM25Index(docs, ids)
        return _BM25_CACHE


# ── Module-level attribute access for backward compatibility ──────────────────

def __getattr__(name: str):
    if name == "_KB":
        return get_kb()
    elif name == "_COLLECTION":
        return get_vectorstore()[0]
    elif name == "_ALL_DOCS":
        return get_vectorstore()[1]
    elif name == "_ALL_IDS":
        return get_vectorstore()[2]
    elif name == "_BM25":
        return get_bm25_index()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf_fuse(
    dense_ids: list[str],
    sparse_ids: list[str],
    id_to_doc: dict[str, str],
    k: int = 60,
) -> list[str]:
    """Fuse two ranked ID lists with Reciprocal Rank Fusion.

    RRF score for a document ``d`` = Σ_i  1 / (k + rank_i(d))
    where the sum is over each retrieval system that returned ``d``.
    ``k=60`` is the standard default from the original RRF paper
    (Cormack et al., 2009).

    Args:
        dense_ids:  Doc IDs ranked by dense (cosine) retrieval, best first.
        sparse_ids: Doc IDs ranked by BM25, best first.
        id_to_doc:  Mapping from ID → document text (for the final output).
        k:          RRF smoothing constant.  Larger values reduce the impact of
                    top-ranked documents from either system.  60 is the canonical
                    default.

    Returns:
        A list of document texts ordered by descending RRF score.
    """
    scores: dict[str, float] = {}

    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    for rank, doc_id in enumerate(sparse_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    # Sort by descending RRF score; filter to IDs that have a document text.
    ranked = sorted(
        ((doc_id, s) for doc_id, s in scores.items() if doc_id in id_to_doc),
        key=lambda x: x[1],
        reverse=True,
    )
    return [id_to_doc[doc_id] for doc_id, _ in ranked]


# ── Retrieval ──────────────────────────────────────────────────────────────────

async def retrieve_knowledge(
    query: str,
    top_k: int = 5,
    min_score: float = 0.35,
) -> str:
    """Async hybrid knowledge retrieval using ChromaDB + BM25 with RRF reranking.

    Both a dense (cosine similarity) and a sparse (BM25) search are executed.
    Their ranked lists are fused using Reciprocal Rank Fusion so that both
    semantic matches *and* exact keyword hits surface to the top.

    The blocking ChromaDB ``.query()`` call is offloaded to a thread pool via
    ``asyncio.to_thread``, keeping the async event loop responsive.

    Args:
        query:      The user's natural-language question.
        top_k:      Number of candidates to fetch from each retrieval system
                    before fusion.  Default: 5 (up from 3 in v1).
        min_score:  Minimum cosine similarity (0–1) a chunk from the **dense**
                    path must reach to be eligible for fusion.  Chunks that fail
                    this threshold are still considered from the sparse path.
                    Default: 0.35.

    Returns:
        A formatted string of up to ``top_k`` fused context chunks, or a
        fallback message when no chunks meet the relevance threshold.
    """
    collection, _, _ = get_vectorstore()
    bm25 = get_bm25_index()

    if collection.count() == 0:
        return "No knowledge base context available."

    n = min(top_k, collection.count())

    # ── Dense retrieval (runs in thread pool to avoid blocking the event loop) ──
    def _dense_query() -> tuple[list[str], list[str], list[float]]:
        results = collection.query(
            query_texts=[query],
            n_results=n,
            include=["documents", "distances"],
        )
        raw_docs: list[str] = results["documents"][0]
        raw_distances: list[float] = results["distances"][0]
        raw_ids: list[str] = results["ids"][0]
        return raw_ids, raw_docs, raw_distances

    dense_ids_all, dense_docs_all, dense_distances = await asyncio.to_thread(_dense_query)

    # Filter dense results by min_score (cosine distance → similarity).
    dense_id_to_doc: dict[str, str] = {}
    dense_ranked_ids: list[str] = []
    for doc_id, doc, dist in zip(dense_ids_all, dense_docs_all, dense_distances):
        similarity = 1.0 - dist / 2.0
        if similarity >= min_score:
            dense_id_to_doc[doc_id] = doc
            dense_ranked_ids.append(doc_id)

    # ── Sparse retrieval (BM25 — pure Python, non-blocking) ───────────────────
    sparse_results = bm25.query(query, top_k=n)
    sparse_id_to_doc: dict[str, str] = {doc_id: doc for doc_id, doc, _ in sparse_results}
    sparse_ranked_ids: list[str] = [doc_id for doc_id, _, _ in sparse_results]

    # ── Merge all known IDs → doc texts ───────────────────────────────────────
    all_id_to_doc: dict[str, str] = {**sparse_id_to_doc, **dense_id_to_doc}

    if not all_id_to_doc:
        return (
            "No sufficiently relevant knowledge base context found for this query. "
            "Answer based on general knowledge about AutoStream."
        )

    # ── Fuse rankings with RRF ─────────────────────────────────────────────────
    fused_docs = _rrf_fuse(
        dense_ids=dense_ranked_ids,
        sparse_ids=sparse_ranked_ids,
        id_to_doc=all_id_to_doc,
    )

    # Return at most top_k fused docs.
    final_docs = fused_docs[:top_k]

    return "\n\n".join(
        f"[Context {i + 1}]\n{doc}" for i, doc in enumerate(final_docs)
    )


# ── Lead Capture ───────────────────────────────────────────────────────────────

def _append_lead_to_csv(name: str, email: str, platform: str) -> None:
    """Append one lead row to leads.csv, creating the file with headers if needed."""
    file_exists = _LEADS_CSV.exists()
    with open(_LEADS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_CSV_HEADERS)
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            name,
            email,
            platform,
        ])


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """Capture lead data and persist it to leads.csv.

    In production this would also POST to a real CRM API.
    """
    _append_lead_to_csv(name, email, platform)
    logger.info("Captured lead: %s <%s> on %s saved to %s", name, email, platform, _LEADS_CSV)

    return (
        f"✅ Lead captured successfully!\n"
        f"  Name: {name}\n"
        f"  Email: {email}\n"
        f"  Platform: {platform}\n\n"
        f"Our team will reach out to {email} within 24 hours to get {name} set up on AutoStream Pro. 🚀"
    )
