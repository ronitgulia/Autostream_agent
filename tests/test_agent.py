"""
tests/test_agent.py — Unit tests for AutoStream Agent

Run with:
    pytest tests/ -v

Tests cover:
  - Email validation via Pydantic _EmailCheck model
  - Intent classification schema (Intent Enum)
  - Semantic RAG knowledge retrieval (ChromaDB)
  - Lead CSV persistence
  - State model_copy safety
  - AI message detection
  - Sliding-window conversation memory (_trim_messages, now async)
  - LLM-driven lead form extraction (LeadFormSchema)
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


# ── RAG Knowledge Retrieval Tests (Semantic) ──────────────────────────────────────
# The new ChromaDB retriever uses cosine similarity, so these tests cover
# both direct keyword queries AND paraphrased / semantic queries that the
# old keyword-matching approach would have silently missed.

class TestRetrieveKnowledge:
    def test_direct_pricing_query(self):
        """Exact keyword match should return pricing context."""
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("what are the pricing plans")
        assert "$" in result or "month" in result.lower()

    def test_semantic_affordability_query(self):
        """'Is it affordable?' has no pricing keywords but should still match price chunks."""
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("Is it affordable?")
        # Semantic search should surface plan or pricing context
        assert any(kw in result.lower() for kw in ["plan", "$", "month", "cost", "price"])

    def test_semantic_video_limits_query(self):
        """'Tell me about video limits' — indirect phrasing should match FAQ chunk."""
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("Tell me about video limits")
        assert any(kw in result.lower() for kw in ["video", "minute", "hour", "limit", "basic", "pro"])

    def test_refund_policy_returned(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("what is your refund policy")
        assert "refund" in result.lower()

    def test_result_is_non_empty_string(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("pricing")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_fallback_for_nonsense_query(self):
        """Even a nonsense query returns the closest chunk (never empty)."""
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("xyzzy foobar baz")
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


# ── Conversation Memory Tests ───────────────────────────────────────────────────────────────
# _trim_messages is now async (uses llm.ainvoke internally).
# We use asyncio.run() to exercise it in sync test methods.

import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from agent.agent import _trim_messages, KEEP_LAST_N


class TestTrimMessages:
    def _make_convo(self, n_pairs: int) -> list:
        """Build n_pairs of (HumanMessage, AIMessage) for a total of 2*n_pairs msgs."""
        msgs = []
        for i in range(n_pairs):
            msgs.append(HumanMessage(content=f"User turn {i}"))
            msgs.append(AIMessage(content=f"Agent turn {i}"))
        return msgs

    def test_short_history_returned_unchanged(self):
        """When len(messages) <= KEEP_LAST_N, no summarisation should happen."""
        msgs = self._make_convo(KEEP_LAST_N // 2)
        result = asyncio.run(_trim_messages(msgs))
        assert result == msgs

    def test_exact_threshold_returned_unchanged(self):
        """Exactly KEEP_LAST_N messages should not trigger summarisation."""
        msgs = self._make_convo(KEEP_LAST_N // 2)
        while len(msgs) < KEEP_LAST_N:
            msgs.append(HumanMessage(content="extra"))
        assert len(msgs) == KEEP_LAST_N
        result = asyncio.run(_trim_messages(msgs))
        assert result == msgs

    def test_long_history_triggers_summary(self):
        """When history exceeds KEEP_LAST_N, _trim_messages should return
        a list starting with a SystemMessage summary + last KEEP_LAST_N msgs."""
        msgs = self._make_convo(KEEP_LAST_N)    # 2 * KEEP_LAST_N total > threshold

        mock_response = MagicMock()
        mock_response.content = "User asked about pricing and features."

        with patch("agent.agent.llm") as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            result = asyncio.run(_trim_messages(msgs))

        assert isinstance(result[0], SystemMessage)
        assert "Conversation Summary" in result[0].content
        assert len(result) == KEEP_LAST_N + 1
        assert result[1:] == msgs[-KEEP_LAST_N:]

    def test_summary_content_included_in_system_message(self):
        """The LLM's summary text should appear inside the SystemMessage."""
        msgs = self._make_convo(KEEP_LAST_N + 2)

        mock_response = MagicMock()
        mock_response.content = "Summary: user discussed refund policy."

        with patch("agent.agent.llm") as mock_llm:
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            result = asyncio.run(_trim_messages(msgs))

        assert "Summary: user discussed refund policy." in result[0].content


# ── Lead Form Schema Tests ────────────────────────────────────────────────────

import pytest
from agent.agent import LeadFormSchema


class TestLeadFormSchema:
    def test_all_fields_present(self):
        schema = LeadFormSchema(name="Ronit", email="ronit@example.com", platform="YouTube")
        assert schema.name == "Ronit"
        assert schema.email == "ronit@example.com"
        assert schema.platform == "YouTube"

    def test_partial_fields_allowed(self):
        """LeadFormSchema fields are all Optional — partial fills are valid."""
        schema = LeadFormSchema(name="Ronit")
        assert schema.name == "Ronit"
        assert schema.email is None
        assert schema.platform is None

    def test_empty_schema_all_none(self):
        schema = LeadFormSchema()
        assert schema.name is None
        assert schema.email is None
        assert schema.platform is None

    def test_lead_merge_logic(self):
        """Verify the merge pattern used in lead_capture_node:
        extracted non-None fields override current; None fields fall back to current."""
        current  = Lead(name="Ronit", email="", platform="")
        extracted = LeadFormSchema(name=None, email="ronit@example.com", platform="YouTube")

        name     = (extracted.name     or "").strip() or current.name
        platform = (extracted.platform or "").strip() or current.platform
        email    = (extracted.email    or "").strip() or current.email

        assert name     == "Ronit"               # kept from current
        assert email    == "ronit@example.com"   # merged from extracted
        assert platform == "YouTube"             # merged from extracted

    def test_invalid_email_detected_by_email_check(self):
        """When extraction yields a bad email, _EmailCheck raises ValidationError."""
        with pytest.raises(ValidationError):
            _EmailCheck(email="not-an-email")


# ── Simplified Lead Stage Tests ───────────────────────────────────────────────

class TestLeadStageSimplified:
    def test_default_stage_is_none(self):
        state = AgentState()
        assert state.lead_stage == "none"

    def test_collecting_stage_is_valid(self):
        state = AgentState(lead_stage="collecting")
        assert state.lead_stage == "collecting"

    def test_done_stage_is_valid(self):
        state = AgentState(lead_stage="done")
        assert state.lead_stage == "done"

    def test_old_fsm_stage_rejected(self):
        """The old FSM stages (collecting_name, etc.) are no longer valid."""
        with pytest.raises((ValidationError, ValueError)):
            AgentState(lead_stage="collecting_name")

    def test_transition_none_to_collecting(self):
        state = AgentState(lead_stage="none")
        updated = state.model_copy(update={"lead_stage": "collecting"})
        assert updated.lead_stage == "collecting"
        assert state.lead_stage == "none"   # original unchanged

    def test_transition_collecting_to_done(self):
        state = AgentState(lead_stage="collecting")
        updated = state.model_copy(update={"lead_stage": "done"})
        assert updated.lead_stage == "done"
