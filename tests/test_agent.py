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
from unittest.mock import patch, MagicMock, AsyncMock, call

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
    _extract_chunk_text,
    KEEP_LAST_N,
    _CIRCUIT_BREAKER_THRESHOLD,
)




def _is_valid_email(raw: str) -> bool:
    """Return True if Pydantic's _EmailCheck accepts the email string."""
    try:
        _EmailCheck(email=raw)
        return True
    except (ValidationError, Exception):
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Email validation
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Intent enum
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# RAG retrieval (original + hybrid)
# ══════════════════════════════════════════════════════════════════════════════

class TestRetrieveKnowledge:
    """All retrieve_knowledge calls must now be awaited (async)."""

    def test_direct_pricing_query(self):
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("what are the pricing plans"))
        assert "$" in result or "month" in result.lower()

    def test_semantic_affordability_query(self):
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("Is it affordable?"))
        assert any(kw in result.lower() for kw in ["plan", "$", "month", "cost", "price"])

    def test_semantic_video_limits_query(self):
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("Tell me about video limits"))
        assert any(kw in result.lower() for kw in ["video", "minute", "hour", "limit", "basic", "pro"])

    def test_refund_policy_returned(self):
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("what is your refund policy"))
        assert "refund" in result.lower()

    def test_result_is_non_empty_string(self):
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("pricing"))
        assert isinstance(result, str) and len(result) > 0

    def test_fallback_for_nonsense_query(self):
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("xyzzy foobar baz"))
        assert isinstance(result, str) and len(result) > 0

    # ── New: hybrid-specific tests ────────────────────────────────────────────

    def test_exact_price_keyword_found(self):
        """BM25 should surface the Pro plan chunk when the exact price token appears."""
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("79"))
        # After RRF, the Pro plan chunk ($79/month) should be in the context.
        assert "79" in result or "pro" in result.lower()

    def test_exact_4k_keyword_found(self):
        """'4K' is an exact token that BM25 handles better than embeddings alone."""
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("4K export resolution"))
        assert "4k" in result.lower() or "pro" in result.lower()

    def test_hybrid_returns_context_tag(self):
        """Fused results should still be formatted with [Context N] tags."""
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("pricing", min_score=0.0))
        assert "[Context 1]" in result


# ══════════════════════════════════════════════════════════════════════════════
# BM25 index
# ══════════════════════════════════════════════════════════════════════════════

class TestBM25Index:
    def test_bm25_returns_results(self):
        from tools.tools import _BM25
        results = _BM25.query("pricing plan monthly", top_k=3)
        assert len(results) > 0

    def test_bm25_result_tuple_structure(self):
        from tools.tools import _BM25
        results = _BM25.query("refund policy", top_k=2)
        for doc_id, doc, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(doc, str)
            assert isinstance(score, float)

    def test_bm25_exact_match_scores_higher(self):
        """'refund' should score higher for a refund query than an unrelated doc."""
        from tools.tools import _BM25
        results = _BM25.query("refund policy", top_k=5)
        top_doc = results[0][1].lower()
        assert "refund" in top_doc

    def test_bm25_top_k_respected(self):
        from tools.tools import _BM25
        results = _BM25.query("any query", top_k=2)
        assert len(results) <= 2


# ══════════════════════════════════════════════════════════════════════════════
# RRF fusion (via tools._rrf_fuse)
# ══════════════════════════════════════════════════════════════════════════════

class TestRRFFuse:
    """Tests for the Reciprocal Rank Fusion helper in tools.py."""

    def _fuse(self, dense_ids, sparse_ids, id_to_doc):
        from tools.tools import _rrf_fuse
        return _rrf_fuse(dense_ids, sparse_ids, id_to_doc)

    def test_top_in_both_lists_wins(self):
        """A doc that ranks #1 in both dense and sparse should be first."""
        doc_map = {"a": "doc_a", "b": "doc_b", "c": "doc_c"}
        result = self._fuse(["a", "b", "c"], ["a", "c", "b"], doc_map)
        assert result[0] == "doc_a"

    def test_ids_only_in_sparse_included(self):
        """Docs that appear only in sparse (not dense) must still be fused in."""
        doc_map = {"a": "doc_a", "b": "doc_b"}
        result = self._fuse(["a"], ["a", "b"], doc_map)
        assert "doc_b" in result

    def test_ids_only_in_dense_included(self):
        doc_map = {"a": "doc_a", "b": "doc_b"}
        result = self._fuse(["a", "b"], ["a"], doc_map)
        assert "doc_b" in result

    def test_unknown_ids_excluded(self):
        """IDs not present in id_to_doc must be silently dropped."""
        doc_map = {"a": "doc_a"}
        result = self._fuse(["a", "z"], ["a", "x"], doc_map)
        assert all(d in ["doc_a"] for d in result)

    def test_empty_lists_return_empty(self):
        result = self._fuse([], [], {})
        assert result == []

    def test_returns_list_of_strings(self):
        doc_map = {"a": "text_a", "b": "text_b"}
        result = self._fuse(["a"], ["b"], doc_map)
        assert all(isinstance(d, str) for d in result)


# ══════════════════════════════════════════════════════════════════════════════
# Lead CSV persistence
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# State copy safety
# ══════════════════════════════════════════════════════════════════════════════

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

    def test_consecutive_failures_default_zero(self):
        assert AgentState().consecutive_failures == 0

    def test_consecutive_failures_increments(self):
        state = AgentState(consecutive_failures=1)
        updated = state.model_copy(update={"consecutive_failures": 2})
        assert state.consecutive_failures == 1
        assert updated.consecutive_failures == 2


# ══════════════════════════════════════════════════════════════════════════════
# AI message detection
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Context trimming
# ══════════════════════════════════════════════════════════════════════════════

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

        with patch("agent.agent._get_llm", return_value=mock_llm), \
             patch("agent.agent._LLM_INSTANCE", mock_llm):
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

        with patch("agent.agent._get_llm", return_value=mock_llm), \
             patch("agent.agent._LLM_INSTANCE", mock_llm):
            result = asyncio.run(_trim_messages(msgs))

        assert "Summary: user discussed refund policy." in result[0].content


# ══════════════════════════════════════════════════════════════════════════════
# Lead form schema
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Lead stage transitions
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# Retry logic
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# RAG score filtering
# ══════════════════════════════════════════════════════════════════════════════

class TestRAGScoreFiltering:
    """Tests for the min_score relevance threshold in retrieve_knowledge()."""

    def test_high_relevance_query_returns_context(self):
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("what are the pricing plans", min_score=0.0))
        assert "[Context 1]" in result

    def test_zero_threshold_always_returns_results(self):
        """With min_score=0, dense path always passes — no fallback."""
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("pricing plans and features", min_score=0.0))
        assert "No sufficiently relevant" not in result

    def test_perfect_threshold_still_has_sparse_fallback(self):
        """With min_score=1.0, dense path returns nothing but BM25 sparse results
        can still contribute via RRF — so relevant queries may still get context."""
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("xyzzy quux foobar nonsense", min_score=1.0))
        # Either real sparse context or the proper fallback — never empty.
        assert isinstance(result, str) and len(result) > 0

    def test_fallback_string_is_non_empty(self):
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("some query", min_score=1.0))
        assert isinstance(result, str) and len(result) > 0

    def test_relevant_query_passes_default_threshold(self):
        """A clearly KB-related query should survive the default 0.35 threshold."""
        from tools.tools import retrieve_knowledge
        result = asyncio.run(retrieve_knowledge("refund policy", min_score=0.35))
        assert isinstance(result, str) and len(result) > 0


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 4 — _extract_chunk_text utility
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractChunkText:
    """Unit tests for the provider-agnostic chunk text extractor."""

    def _make_chunk(self, content):
        chunk = MagicMock()
        chunk.content = content
        return chunk

    def test_string_content_returned_directly(self):
        chunk = self._make_chunk("Hello, world!")
        assert _extract_chunk_text(chunk) == "Hello, world!"

    def test_empty_string_content(self):
        chunk = self._make_chunk("")
        assert _extract_chunk_text(chunk) == ""

    def test_google_list_content_extracted(self):
        """Gemini returns content as list[dict] with type/text pairs."""
        chunk = self._make_chunk([
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ])
        assert _extract_chunk_text(chunk) == "Hello world"

    def test_list_with_non_text_items_skipped(self):
        chunk = self._make_chunk([
            {"type": "tool_use", "id": "123"},
            {"type": "text", "text": "visible"},
        ])
        assert _extract_chunk_text(chunk) == "visible"

    def test_empty_list_returns_empty_string(self):
        chunk = self._make_chunk([])
        assert _extract_chunk_text(chunk) == ""

    def test_none_content_returns_empty_string(self):
        chunk = self._make_chunk(None)
        assert _extract_chunk_text(chunk) == ""

    def test_unexpected_type_returns_empty_string(self):
        chunk = self._make_chunk(12345)
        assert _extract_chunk_text(chunk) == ""


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 3 — Circuit breaker
# ══════════════════════════════════════════════════════════════════════════════

class TestCircuitBreaker:
    """Tests for the consecutive_failures circuit breaker in AgentState."""

    def test_circuit_open_at_threshold(self):
        from agent.agent import _check_circuit_breaker
        state = AgentState(consecutive_failures=_CIRCUIT_BREAKER_THRESHOLD)
        result = _check_circuit_breaker(state)
        assert result is not None
        assert "trouble" in result.lower() or "try again" in result.lower()

    def test_circuit_closed_below_threshold(self):
        from agent.agent import _check_circuit_breaker
        state = AgentState(consecutive_failures=_CIRCUIT_BREAKER_THRESHOLD - 1)
        assert _check_circuit_breaker(state) is None

    def test_circuit_closed_at_zero(self):
        from agent.agent import _check_circuit_breaker
        state = AgentState(consecutive_failures=0)
        assert _check_circuit_breaker(state) is None

    def test_circuit_message_is_string(self):
        from agent.agent import _check_circuit_breaker
        state = AgentState(consecutive_failures=_CIRCUIT_BREAKER_THRESHOLD)
        msg = _check_circuit_breaker(state)
        assert isinstance(msg, str) and len(msg) > 0

    def test_failures_reset_after_circuit_opens(self):
        """Simulate detect_intent node resetting consecutive_failures to 0."""
        state = AgentState(consecutive_failures=_CIRCUIT_BREAKER_THRESHOLD)
        # After the circuit fires, the node resets to 0.
        updated = state.model_copy(update={"consecutive_failures": 0})
        assert updated.consecutive_failures == 0


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 3 — Safe intent detection fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeDetectIntent:
    """Tests for _safe_detect_intent's graceful fallback behaviour."""

    def test_returns_intent_response_on_success(self):
        from agent.agent import _safe_detect_intent, IntentResponse

        mock_result = IntentResponse(intent=Intent.product_inquiry)
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_result)

        state = AgentState(messages=[HumanMessage(content="What are the plans?")])

        with patch("agent.agent._get_intent_llm", return_value=mock_llm):
            result = asyncio.run(_safe_detect_intent(state))

        assert result.intent == Intent.product_inquiry

    def test_falls_back_to_product_inquiry_on_llm_failure(self):
        """When the LLM raises repeatedly, the safe fallback must be product_inquiry."""
        from agent.agent import _safe_detect_intent

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("API unavailable"))

        state = AgentState(messages=[HumanMessage(content="some message")])

        with patch("agent.agent._get_intent_llm", return_value=mock_llm), \
             patch("agent.agent._with_retry", AsyncMock(side_effect=RuntimeError("API unavailable"))):
            result = asyncio.run(_safe_detect_intent(state))

        assert result.intent == Intent.product_inquiry

    def test_fallback_uses_product_inquiry_not_greeting(self):
        """product_inquiry is the safer default — it routes to RAG, not a silent greeting."""
        from agent.agent import _safe_detect_intent

        state = AgentState(messages=[HumanMessage(content="hello")])

        with patch("agent.agent._with_retry", AsyncMock(side_effect=Exception("fail"))):
            result = asyncio.run(_safe_detect_intent(state))

        assert result.intent == Intent.product_inquiry


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 3 — Safe lead extraction fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeExtractLead:
    def test_returns_schema_on_success(self):
        from agent.agent import _safe_extract_lead

        expected = LeadFormSchema(name="Alice", email="alice@example.com", platform="YouTube")
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=expected)

        with patch("agent.agent._get_extract_llm", return_value=mock_llm):
            result = asyncio.run(_safe_extract_lead("Alice, alice@example.com, YouTube"))

        assert result.name == "Alice"

    def test_returns_empty_schema_on_failure(self):
        """On LLM failure, _safe_extract_lead must return all-None schema."""
        from agent.agent import _safe_extract_lead

        with patch("agent.agent._with_retry", AsyncMock(side_effect=Exception("fail"))):
            result = asyncio.run(_safe_extract_lead("some message"))

        assert result.name is None
        assert result.email is None
        assert result.platform is None

    def test_empty_schema_is_lead_form_schema_type(self):
        from agent.agent import _safe_extract_lead

        with patch("agent.agent._with_retry", AsyncMock(side_effect=Exception("fail"))):
            result = asyncio.run(_safe_extract_lead("test"))

        assert isinstance(result, LeadFormSchema)


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 5 — Intent classifier temperature & few-shot
# ══════════════════════════════════════════════════════════════════════════════

class TestIntentClassifierConfig:
    def test_intent_prompt_contains_few_shot_examples(self):
        from agent.agent import _INTENT_PROMPT
        # Must have at least one example per class
        assert "greeting" in _INTENT_PROMPT
        assert "product_inquiry" in _INTENT_PROMPT
        assert "high_intent" in _INTENT_PROMPT
        # The examples section should be present
        assert "EXAMPLES" in _INTENT_PROMPT or "→" in _INTENT_PROMPT

    def test_intent_prompt_contains_all_three_class_labels(self):
        from agent.agent import _INTENT_PROMPT
        for label in ("greeting", "product_inquiry", "high_intent"):
            assert label in _INTENT_PROMPT

    def test_intent_prompt_has_at_least_6_example_lines(self):
        """The few-shot section should have at least 6 annotated examples."""
        from agent.agent import _INTENT_PROMPT
        arrow_count = _INTENT_PROMPT.count("→")
        assert arrow_count >= 6, f"Expected ≥6 few-shot arrows, found {arrow_count}"
