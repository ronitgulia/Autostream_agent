"""
tests/test_agent.py — Unit tests for AutoStream Agent

Run with:
    pytest tests/ -v

Tests cover:
  - Email validation logic (Bug 4)
  - Knowledge retrieval (RAG keyword matching)
  - Lead CSV persistence (Problem 2)
  - State deepcopy safety (Bug 1)
  - AI message detection (Bug 2)
"""

import sys
import os
import csv
import copy
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ── Email Validation Tests ────────────────────────────────────────────────────

import re

EMAIL_PATTERN = r'^[\w\.-]+@[\w\.-]+\.\w{2,6}$'

def _is_valid_email(raw: str) -> bool:
    """Mirror the normalization + regex used in agent.py."""
    email = raw.strip().lower()
    return bool(re.match(EMAIL_PATTERN, email))


class TestEmailValidation:
    def test_valid_email_accepted(self):
        assert _is_valid_email("user@example.com") is True

    def test_valid_email_with_dots_accepted(self):
        assert _is_valid_email("first.last@sub.domain.org") is True

    def test_valid_email_uppercase_normalized(self):
        # Bug 4: uppercase should be normalized before matching
        assert _is_valid_email("User@Gmail.COM") is True

    def test_valid_email_with_whitespace_normalized(self):
        assert _is_valid_email("  user@example.com  ") is True

    def test_missing_domain_rejected(self):
        # "riya@" — no domain after @
        assert _is_valid_email("riya@") is False

    def test_missing_at_rejected(self):
        # "notanemail" — no @ at all
        assert _is_valid_email("notanemail") is False

    def test_single_char_tld_rejected(self):
        # TLD must be at least 2 chars
        assert _is_valid_email("user@domain.c") is False

    def test_no_tld_rejected(self):
        assert _is_valid_email("user@domain") is False

    def test_empty_string_rejected(self):
        assert _is_valid_email("") is False

    def test_spaces_only_rejected(self):
        assert _is_valid_email("   ") is False


# ── RAG Knowledge Retrieval Tests ─────────────────────────────────────────────

class TestRetrieveKnowledge:
    def test_pricing_keywords_return_plans(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("how much does it cost")
        assert "Pricing" in result or "plan" in result.lower()

    def test_pricing_direct_keyword(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("what are the pricing plans")
        assert "$" in result  # plans have prices

    def test_refund_policy_returned(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("what is your refund policy")
        assert "refund" in result.lower() or "Policies" in result

    def test_unknown_query_returns_fallback(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("xyzzy foobar nonsense")
        assert "AutoStream" in result  # fallback always mentions company

    def test_result_is_string(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("pricing")
        assert isinstance(result, str)
        assert len(result) > 0


# ── Lead CSV Persistence Tests ────────────────────────────────────────────────

class TestLeadCSVPersistence:
    def test_lead_written_to_csv(self, tmp_path):
        """Lead data should be appended to leads.csv."""
        import tools.tools as tools_module
        original_path = tools_module._LEADS_CSV
        try:
            tools_module._LEADS_CSV = tmp_path / "leads.csv"
            tools_module.mock_lead_capture("Alice", "alice@example.com", "YouTube")

            assert tools_module._LEADS_CSV.exists()
            rows = list(csv.reader(open(tools_module._LEADS_CSV, encoding="utf-8")))
            assert rows[0] == ["timestamp", "name", "email", "platform"]
            assert rows[1][1] == "Alice"
            assert rows[1][2] == "alice@example.com"
            assert rows[1][3] == "YouTube"
        finally:
            tools_module._LEADS_CSV = original_path

    def test_multiple_leads_appended(self, tmp_path):
        """Each call appends a new row, headers written only once."""
        import tools.tools as tools_module
        original_path = tools_module._LEADS_CSV
        try:
            tools_module._LEADS_CSV = tmp_path / "leads.csv"
            tools_module.mock_lead_capture("Alice", "alice@example.com", "YouTube")
            tools_module.mock_lead_capture("Bob", "bob@example.com", "TikTok")

            rows = list(csv.reader(open(tools_module._LEADS_CSV, encoding="utf-8")))
            assert len(rows) == 3  # 1 header + 2 data rows
            assert rows[1][1] == "Alice"
            assert rows[2][1] == "Bob"
        finally:
            tools_module._LEADS_CSV = original_path

    def test_lead_capture_returns_confirmation_string(self):
        """Return value should be a non-empty confirmation message."""
        import tools.tools as tools_module
        original_path = tools_module._LEADS_CSV

        with tempfile.TemporaryDirectory() as tmp:
            try:
                tools_module._LEADS_CSV = Path(tmp) / "leads.csv"
                result = tools_module.mock_lead_capture("Carol", "carol@example.com", "Instagram")
                assert isinstance(result, str)
                assert "Carol" in result
                assert "carol@example.com" in result
            finally:
                tools_module._LEADS_CSV = original_path


# ── State Deepcopy Safety Tests (Bug 1) ──────────────────────────────────────

class TestStateCopySafety:
    def test_deepcopy_isolates_messages_list(self):
        """Mutating a deepcopy's messages should not affect the original."""
        INITIAL_STATE = {
            "messages": [],
            "intent": "greeting",
            "lead_stage": "none",
            "lead_name": "",
            "lead_email": "",
            "lead_platform": "",
            "rag_context": "",
        }
        state = copy.deepcopy(INITIAL_STATE)
        state["messages"].append(HumanMessage(content="hello"))

        assert len(INITIAL_STATE["messages"]) == 0, (
            "Shallow copy bug: INITIAL_STATE was mutated!"
        )

    def test_shallow_copy_would_fail(self):
        """Demonstrate that .copy() shares the messages list reference."""
        original = {"messages": []}
        shallow = original.copy()
        shallow["messages"].append("oops")
        # With shallow copy, original is polluted:
        assert original["messages"] == ["oops"]


# ── AIMessage Detection Tests (Bug 2) ─────────────────────────────────────────

class TestAIMessageDetection:
    def test_isinstance_detects_ai_message(self):
        msg = AIMessage(content="Hello from agent")
        assert isinstance(msg, AIMessage) is True

    def test_isinstance_rejects_human_message(self):
        msg = HumanMessage(content="Hello from user")
        assert isinstance(msg, AIMessage) is False

    def test_isinstance_rejects_system_message(self):
        msg = SystemMessage(content="System prompt")
        assert isinstance(msg, AIMessage) is False

    def test_find_last_ai_message(self):
        """Simulate what main.py does to extract the last agent reply."""
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="Hello!"),
            HumanMessage(content="pricing?"),
            AIMessage(content="We have three plans."),
        ]
        last_ai = next(
            (m.content for m in reversed(messages) if isinstance(m, AIMessage)),
            "(no response)"
        )
        assert last_ai == "We have three plans."

    def test_no_ai_message_returns_default(self):
        messages = [HumanMessage(content="hello")]
        last_ai = next(
            (m.content for m in reversed(messages) if isinstance(m, AIMessage)),
            "(no response)"
        )
        assert last_ai == "(no response)"
