"""
tools.py — AutoStream Agent Tool Definitions
Provides:
  - mock_lead_capture(): simulates CRM lead capture
  - retrieve_knowledge(): RAG retrieval from local JSON knowledge base
"""

import json
import os
import csv
from datetime import datetime, timezone
from pathlib import Path


# ── Load knowledge base once at import time ──────────────────────────────────

_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"

def _load_kb() -> dict:
    with open(_KB_PATH, "r") as f:
        return json.load(f)

_KB = _load_kb()


# ── Tool 1: RAG Knowledge Retrieval ─────────────────────────────────────────

def retrieve_knowledge(query: str) -> str:
    """
    Simulate RAG retrieval by searching the local knowledge base.
    Returns a formatted string of relevant context for the LLM.
    """
    query_lower = query.lower()
    results = []

    # Match pricing / plan queries
    if any(kw in query_lower for kw in ["price", "pricing", "plan", "cost", "how much", "basic", "pro", "subscription"]):
        results.append("=== AutoStream Pricing Plans ===")
        for plan in _KB["plans"]:
            features_str = "\n    • ".join(plan["features"])
            results.append(
                f"\n📦 {plan['name']} — ${plan['price_monthly']}/month\n"
                f"  Best for: {plan['best_for']}\n"
                f"  Features:\n    • {features_str}"
            )

    # Match policy queries
    if any(kw in query_lower for kw in ["refund", "cancel", "support", "policy", "trial", "free"]):
        results.append("\n=== Company Policies ===")
        for policy in _KB["policies"]:
            results.append(f"\n📋 {policy['topic']}: {policy['details']}")

    # Match FAQ queries
    if any(kw in query_lower for kw in ["platform", "youtube", "instagram", "tiktok", "upgrade", "downgrade",
                                          "video length", "limit", "team", "export", "resolution"]):
        results.append("\n=== FAQs ===")
        for faq in _KB["faqs"]:
            if any(kw in faq["question"].lower() or kw in faq["answer"].lower()
                   for kw in query_lower.split()):
                results.append(f"\nQ: {faq['question']}\nA: {faq['answer']}")

    # General company info fallback
    if not results:
        co = _KB["company"]
        results.append(
            f"=== About AutoStream ===\n{co['description']}\nTagline: {co['tagline']}"
        )
        results.append("\n=== Available Plans ===")
        for plan in _KB["plans"]:
            results.append(f"  • {plan['name']}: ${plan['price_monthly']}/month")

    return "\n".join(results)


# ── Tool 2: Lead Capture (persists to leads.csv) ────────────────────────────

_LEADS_CSV = Path(__file__).parent.parent / "leads.csv"
_CSV_HEADERS = ["timestamp", "name", "email", "platform"]

def _append_lead_to_csv(name: str, email: str, platform: str) -> None:
    """Write one lead row to leads.csv, creating the file with headers if needed."""
    file_exists = _LEADS_CSV.exists()
    with open(_LEADS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_CSV_HEADERS)   # write header only on first run
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            name,
            email,
            platform,
        ])


def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Capture lead data and persist it to leads.csv.
    In production this would also POST to a real CRM API.
    """
    # ── Persist to CSV ───────────────────────────────────────────────────────
    _append_lead_to_csv(name, email, platform)

    # ── Console confirmation ─────────────────────────────────────────────────
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
