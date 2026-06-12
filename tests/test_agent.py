"""
tests/test_agent.py — Unit tests for AutoStream Agent

Run with:
    pytest tests/ -v

Tests cover:
  - Email validation via Pydantic Lead model
  - Intent classification schema (Intent Enum)
  - Knowledge retrieval (RAG keyword matching)
  - Lead CSV persistence
  - State model_copy safety
  - AI message detection
"""

import sys
import os
import csv
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from pydantic import ValidationError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ── Email Validation via Pydantic Lead Model ─────────────────────────────────
# We test through the Lead model directly — the same code path production uses.
# No need to duplicate the regex here.

from agent.agent import Lead, Intent, AgentState, _EmailCheck


def _is_valid_email(raw: str) -> bool:
    """Try building an _EmailCheck with this email; return True if Pydantic accepts it."""
    try:
        _EmailCheck(email=raw)
        return True
    except (ValidationError, Exception):
        return False


class TestEmailValidation:
    def test_valid_email_accepted(self):
        assert _is_valid_email("user@example.com") is True

    def test_valid_email_with_dots_accepted(self):
        assert _is_valid_email("first.last@sub.domain.org") is True

    def test_valid_email_uppercase_normalized(self):
        # EmailStr auto-normalises to lowercase before validating
        assert _is_valid_email("User@Gmail.COM") is True

    def test_valid_email_with_whitespace_normalized(self):
        assert _is_valid_email("  user@example.com  ") is True

    def test_missing_domain_rejected(self):
        assert _is_valid_email("riya@") is False

    def test_missing_at_rejected(self):
        assert _is_valid_email("notanemail") is False

    def test_no_tld_rejected(self):
        assert _is_valid_email("user@domain") is False

    def test_empty_string_rejected(self):
        assert _is_valid_email("") is False

    def test_spaces_only_rejected(self):
        assert _is_valid_email("   ") is False


# ── Intent Enum Tests ─────────────────────────────────────────────────────────

class TestIntentEnum:
    def test_intent_values_are_strings(self):
        """Enum members should be usable as plain strings (str, Enum mixin)."""
        assert Intent.greeting == "greeting"
        assert Intent.product_inquiry == "product_inquiry"
        assert Intent.high_intent == "high_intent"

    def test_intent_from_string(self):
        assert Intent("greeting") is Intent.greeting
        assert Intent("high_intent") is Intent.high_intent

    def test_invalid_intent_raises(self):
        try:
            Intent("unknown")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_agent_state_default_intent(self):
        state = AgentState()
        assert state.intent == Intent.greeting


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


# ── State model_copy Safety Tests ────────────────────────────────────────────

class TestStateCopySafety:
    def test_model_copy_isolates_messages_list(self):
        """model_copy() should not mutate the original state's messages."""
        state = AgentState()
        new_state = state.model_copy(update={"messages": state.messages + [HumanMessage(content="hello")]})

        assert len(state.messages) == 0, "Original state was mutated!"
        assert len(new_state.messages) == 1

    def test_model_copy_updates_intent(self):
        state = AgentState()
        updated = state.model_copy(update={"intent": Intent.high_intent})
        assert state.intent == Intent.greeting   # original unchanged
        assert updated.intent == Intent.high_intent


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
