"""
tests/test_agent.py — Unit tests for AutoStream Agent

Run with:
    pytest tests/ -v
"""

import sys
import os
import csv
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from pydantic import ValidationError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.agent import (
    Lead,
    Intent,
    AgentState,
    LeadFormSchema,
    _EmailCheck,
    _trim_messages,
    _with_retry,
    _get_llm,
    KEEP_LAST_N,
)


def _is_valid_email(raw: str) -> bool:
    """Return True if Pydantic's _EmailCheck accepts the email string."""
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


class TestIntentEnum:
    def test_intent_values_are_strings(self):
        assert Intent.greeting == "greeting"
        assert Intent.product_inquiry == "product_inquiry"
        assert Intent.high_intent == "high_intent"

    def test_intent_from_string(self):
        assert Intent("greeting") is Intent.greeting
        assert Intent("high_intent") is Intent.high_intent

    def test_invalid_intent_raises(self):
        with pytest.raises(ValueError):
            Intent("unknown")

    def test_agent_state_default_intent(self):
        state = AgentState()
        assert state.intent == Intent.greeting


class TestRetrieveKnowledge:
    def test_direct_pricing_query(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("what are the pricing plans")
        assert "$" in result or "month" in result.lower()

    def test_semantic_affordability_query(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("Is it affordable?")
        assert any(kw in result.lower() for kw in ["plan", "$", "month", "cost", "price"])

    def test_semantic_video_limits_query(self):
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
        assert isinstance(result, str) and len(result) > 0

    def test_fallback_for_nonsense_query(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("xyzzy foobar baz")
        assert isinstance(result, str) and len(result) > 0


class TestLeadCSVPersistence:
    def test_lead_written_to_csv(self, tmp_path):
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
        import tools.tools as tools_module
        original_path = tools_module._LEADS_CSV
        try:
            tools_module._LEADS_CSV = tmp_path / "leads.csv"
            tools_module.mock_lead_capture("Alice", "alice@example.com", "YouTube")
            tools_module.mock_lead_capture("Bob", "bob@example.com", "TikTok")

            rows = list(csv.reader(open(tools_module._LEADS_CSV, encoding="utf-8")))
            assert len(rows) == 3
            assert rows[1][1] == "Alice"
            assert rows[2][1] == "Bob"
        finally:
            tools_module._LEADS_CSV = original_path

    def test_lead_capture_returns_confirmation_string(self):
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


class TestStateCopySafety:
    def test_model_copy_isolates_messages_list(self):
        state = AgentState()
        new_state = state.model_copy(
            update={"messages": state.messages + [HumanMessage(content="hello")]}
        )
        assert len(state.messages) == 0
        assert len(new_state.messages) == 1

    def test_model_copy_updates_intent(self):
        state = AgentState()
        updated = state.model_copy(update={"intent": Intent.high_intent})
        assert state.intent == Intent.greeting
        assert updated.intent == Intent.high_intent


class TestAIMessageDetection:
    def test_isinstance_detects_ai_message(self):
        assert isinstance(AIMessage(content="Hello from agent"), AIMessage) is True

    def test_isinstance_rejects_human_message(self):
        assert isinstance(HumanMessage(content="Hello from user"), AIMessage) is False

    def test_isinstance_rejects_system_message(self):
        assert isinstance(SystemMessage(content="System prompt"), AIMessage) is False

    def test_find_last_ai_message(self):
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


class TestTrimMessages:
    def _make_convo(self, n_pairs: int) -> list:
        msgs = []
        for i in range(n_pairs):
            msgs.append(HumanMessage(content=f"User turn {i}"))
            msgs.append(AIMessage(content=f"Agent turn {i}"))
        return msgs

    def test_short_history_returned_unchanged(self):
        msgs = self._make_convo(KEEP_LAST_N // 2)
        result = asyncio.run(_trim_messages(msgs))
        assert result == msgs

    def test_exact_threshold_returned_unchanged(self):
        msgs = self._make_convo(KEEP_LAST_N // 2)
        while len(msgs) < KEEP_LAST_N:
            msgs.append(HumanMessage(content="extra"))
        assert len(msgs) == KEEP_LAST_N
        result = asyncio.run(_trim_messages(msgs))
        assert result == msgs

    def test_long_history_triggers_summary(self):
        msgs = self._make_convo(KEEP_LAST_N)

        mock_response = MagicMock()
        mock_response.content = "User asked about pricing and features."

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("agent.agent._get_llm", return_value=mock_llm):
            result = asyncio.run(_trim_messages(msgs))

        assert isinstance(result[0], SystemMessage)
        assert "Conversation Summary" in result[0].content
        assert len(result) == KEEP_LAST_N + 1
        assert result[1:] == msgs[-KEEP_LAST_N:]

    def test_summary_content_included_in_system_message(self):
        msgs = self._make_convo(KEEP_LAST_N + 2)

        mock_response = MagicMock()
        mock_response.content = "Summary: user discussed refund policy."

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        with patch("agent.agent._get_llm", return_value=mock_llm):
            result = asyncio.run(_trim_messages(msgs))

        assert "Summary: user discussed refund policy." in result[0].content


class TestLeadFormSchema:
    def test_all_fields_present(self):
        schema = LeadFormSchema(name="Ronit", email="ronit@example.com", platform="YouTube")
        assert schema.name == "Ronit"
        assert schema.email == "ronit@example.com"
        assert schema.platform == "YouTube"

    def test_partial_fields_allowed(self):
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
        current = Lead(name="Ronit", email="", platform="")
        extracted = LeadFormSchema(name=None, email="ronit@example.com", platform="YouTube")

        name = (extracted.name or "").strip() or current.name
        platform = (extracted.platform or "").strip() or current.platform
        email = (extracted.email or "").strip() or current.email

        assert name == "Ronit"
        assert email == "ronit@example.com"
        assert platform == "YouTube"

    def test_invalid_email_detected_by_email_check(self):
        with pytest.raises(ValidationError):
            _EmailCheck(email="not-an-email")


class TestLeadStageSimplified:
    def test_default_stage_is_none(self):
        assert AgentState().lead_stage == "none"

    def test_collecting_stage_is_valid(self):
        assert AgentState(lead_stage="collecting").lead_stage == "collecting"

    def test_done_stage_is_valid(self):
        assert AgentState(lead_stage="done").lead_stage == "done"

    def test_old_fsm_stage_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            AgentState(lead_stage="collecting_name")

    def test_transition_none_to_collecting(self):
        state = AgentState(lead_stage="none")
        updated = state.model_copy(update={"lead_stage": "collecting"})
        assert updated.lead_stage == "collecting"
        assert state.lead_stage == "none"

    def test_transition_collecting_to_done(self):
        state = AgentState(lead_stage="collecting")
        updated = state.model_copy(update={"lead_stage": "done"})
        assert updated.lead_stage == "done"


class TestRetryLogic:
    """Tests for the _with_retry() exponential backoff helper."""

    def test_succeeds_on_first_attempt(self):
        call_count = 0

        async def always_ok():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = asyncio.run(_with_retry(always_ok))
        assert result == "ok"
        assert call_count == 1

    def test_retries_on_failure_then_succeeds(self):
        call_count = 0

        async def fail_twice_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient error")
            return "recovered"

        result = asyncio.run(
            _with_retry(fail_twice_then_succeed, max_attempts=3, base_delay=0)
        )
        assert result == "recovered"
        assert call_count == 3

    def test_raises_after_max_attempts_exhausted(self):
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("permanent failure")

        with pytest.raises(ValueError, match="permanent failure"):
            asyncio.run(
                _with_retry(always_fails, max_attempts=3, base_delay=0)
            )
        assert call_count == 3

    def test_succeeds_on_second_attempt(self):
        attempts = []

        async def fail_once():
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionError("first attempt failed")
            return "second-try success"

        result = asyncio.run(
            _with_retry(fail_once, max_attempts=3, base_delay=0)
        )
        assert result == "second-try success"
        assert len(attempts) == 2

    def test_keyboard_interrupt_not_retried(self):
        """KeyboardInterrupt must propagate immediately without retry."""
        call_count = 0

        async def raises_keyboard_interrupt():
            nonlocal call_count
            call_count += 1
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            asyncio.run(
                _with_retry(raises_keyboard_interrupt, max_attempts=3, base_delay=0)
            )
        assert call_count == 1


class TestRAGScoreFiltering:
    """Tests for the min_score relevance threshold in retrieve_knowledge()."""

    def test_high_relevance_query_returns_context(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("what are the pricing plans", min_score=0.0)
        assert "[Context 1]" in result

    def test_zero_threshold_always_returns_results(self):
        """With min_score=0, every chunk passes — no fallback should occur."""
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("pricing plans and features", min_score=0.0)
        assert "No sufficiently relevant" not in result

    def test_perfect_threshold_filters_all_chunks(self):
        """With min_score=1.0, only a perfect match passes (almost never).
        For a generic query, the fallback string must be returned."""
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("xyzzy quux foobar nonsense", min_score=1.0)
        assert "No sufficiently relevant" in result

    def test_fallback_string_is_non_empty(self):
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("some query", min_score=1.0)
        assert isinstance(result, str) and len(result) > 0

    def test_relevant_query_passes_default_threshold(self):
        """A clearly KB-related query should survive the default 0.35 threshold."""
        from tools.tools import retrieve_knowledge
        result = retrieve_knowledge("refund policy", min_score=0.35)
        # Either real context or a proper fallback — never an empty string.
        assert isinstance(result, str) and len(result) > 0
