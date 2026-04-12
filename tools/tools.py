"""
tools.py — AutoStream Agent Tool Definitions
Provides:
  - mock_lead_capture(): simulates CRM lead capture
  - retrieve_knowledge(): RAG retrieval from local JSON knowledge base
"""

import json
import os
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


# ── Tool 2: Mock Lead Capture ────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> str:
    """
    Simulate sending lead data to a CRM / backend system.
    In production this would be an HTTP POST to a real API.
    """
    print(f"\n{'='*55}")
    print(f"  ✅  LEAD CAPTURED SUCCESSFULLY")
    print(f"{'='*55}")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"{'='*55}\n")

    return (
        f"✅ Lead captured successfully!\n"
        f"  Name: {name}\n"
        f"  Email: {email}\n"
        f"  Platform: {platform}\n\n"
        f"Our team will reach out to {email} within 24 hours to get {name} set up on AutoStream Pro. 🚀"
    )
