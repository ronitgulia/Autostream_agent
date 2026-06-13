import json
import csv
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"
_CHROMA_PATH = Path(__file__).parent.parent / ".chroma_db"


def _load_kb() -> dict:
    with open(_KB_PATH, "r") as f:
        return json.load(f)


_KB = _load_kb()


def _build_chunks(kb: dict) -> tuple[list[str], list[str]]:
    """Convert the KB JSON into flat text chunks and stable IDs.

    One chunk per logical unit: company overview, each plan, each policy, each FAQ.
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


def _build_vectorstore():
    """Create or load the persistent ChromaDB collection.

    On first run, chunks and embeds the knowledge base (~5 s to download model).
    On subsequent runs, loads the existing collection from .chroma_db/ instantly.
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=str(_CHROMA_PATH))
    collection = client.get_or_create_collection(
        name="autostream_kb",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() == 0:
        docs, ids = _build_chunks(_KB)
        collection.add(documents=docs, ids=ids)
        print(f"[RAG] Vector store built: {len(docs)} chunks embedded → .chroma_db/")
    else:
        print(f"[RAG] Vector store loaded: {collection.count()} chunks from .chroma_db/")

    return collection


_COLLECTION = _build_vectorstore()


def retrieve_knowledge(query: str, top_k: int = 3) -> str:
    """Semantic knowledge retrieval using ChromaDB cosine similarity search.

    Args:
        query:  The user's natural-language question.
        top_k:  Number of most relevant chunks to return (default: 3).

    Returns:
        A formatted string of the top-k context chunks for the LLM.
    """
    n = min(top_k, _COLLECTION.count())
    if n == 0:
        return "No knowledge base context available."

    results = _COLLECTION.query(query_texts=[query], n_results=n)
    matched_docs: list[str] = results["documents"][0]

    if not matched_docs:
        co = _KB["company"]
        return f"About AutoStream: {co['description']}\nTagline: {co['tagline']}"

    return "\n\n".join(
        f"[Context {i + 1}]\n{doc}" for i, doc in enumerate(matched_docs)
    )


_LEADS_CSV = Path(__file__).parent.parent / "leads.csv"
_CSV_HEADERS = ["timestamp", "name", "email", "platform"]


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

    print(f"\n{'='*55}")
    print(f"  ✅  LEAD CAPTURED & SAVED TO leads.csv")
    print(f"{'='*55}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Saved to : {_LEADS_CSV}")
    print(f"{'='*55}\n")

    return (
        f"✅ Lead captured successfully!\n"
        f"  Name: {name}\n"
        f"  Email: {email}\n"
        f"  Platform: {platform}\n\n"
        f"Our team will reach out to {email} within 24 hours to get {name} set up on AutoStream Pro. 🚀"
    )
