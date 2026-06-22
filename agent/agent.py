"""
agent/agent.py — LangGraph state machine for the AutoStream AI sales agent.

Improvements over v1:
  3. Graceful Degradation & Circuit Breaker:
     - ``_safe_detect_intent`` falls back to ``product_inquiry`` on LLM failure
       instead of crashing the turn.
     - ``_safe_extract_lead`` returns an empty ``LeadFormSchema`` on failure so
       the bot simply re-asks for all fields.
     - ``AgentState.consecutive_failures`` counts back-to-back LLM errors. Once
       it hits ``_CIRCUIT_BREAKER_THRESHOLD``, the agent emits a user-facing
       "service unavailable" message and resets the counter.

  4. LLM Singleton Fix & Clean Streaming Abstraction:
     - ``@lru_cache`` replaced with an explicit module-level singleton protected
       by ``asyncio.Lock`` — safe for concurrent coroutine first-init.
     - ``_extract_chunk_text(chunk)`` is a testable utility that normalises the
       ``str | list[dict]`` content difference across providers.
     - ``stream_response(graph, state)`` is an async generator that encapsulates
       all ``astream_events`` parsing, making ``main.py`` a thin display layer.

  5. Deterministic Intent Classifier with Few-Shot Examples:
     - Intent LLM cloned at ``temperature=0`` for fully deterministic output.
     - ``_INTENT_PROMPT`` includes 6 labelled few-shot examples (2 per class)
       to anchor the classifier on borderline messages.
"""

import asyncio
import logging
import os
import random
from enum import Enum
from typing import Annotated, AsyncGenerator, Literal, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, field_validator, ConfigDict, ValidationError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools.tools import retrieve_knowledge, mock_lead_capture

load_dotenv()

logger = logging.getLogger(__name__)

# ── Circuit-breaker config ─────────────────────────────────────────────────────
_CIRCUIT_BREAKER_THRESHOLD = 3


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 4 — Async-safe LLM singleton
# ══════════════════════════════════════════════════════════════════════════════

_LLM_INSTANCE = None
_LLM_LOCK: asyncio.Lock | None = None  # Created lazily (must be inside an event loop)


def _get_or_create_lock() -> asyncio.Lock:
    """Return the module-level asyncio.Lock, creating it on first call.

    ``asyncio.Lock`` must be created inside a running event loop, so we cannot
    assign it at module level on Python < 3.10.  This helper creates it lazily
    and caches the result.
    """
    global _LLM_LOCK
    if _LLM_LOCK is None:
        _LLM_LOCK = asyncio.Lock()
    return _LLM_LOCK


async def _get_llm_async():
    """Return (or lazily construct) the shared LLM singleton.

    Using an ``asyncio.Lock`` instead of ``@lru_cache`` prevents the race
    condition where two concurrent coroutines both see ``_LLM_INSTANCE is None``
    and try to construct the client simultaneously.

    Supported providers (set via ``LLM_PROVIDER`` env var):
        anthropic (default), openai, google, groq.
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    async with _get_or_create_lock():
        # Double-checked locking: another coroutine may have initialised it
        # while we were waiting for the lock.
        if _LLM_INSTANCE is not None:
            return _LLM_INSTANCE

        provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

        if provider == "openai":
            from langchain_openai import ChatOpenAI
            _LLM_INSTANCE = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.3,
            )
        elif provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            _LLM_INSTANCE = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3,
            )
        elif provider == "groq":
            from langchain_groq import ChatGroq
            _LLM_INSTANCE = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.3,
            )
        else:
            from langchain_anthropic import ChatAnthropic
            _LLM_INSTANCE = ChatAnthropic(
                model="claude-haiku-4-5-20251001",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                temperature=0.3,
            )

    return _LLM_INSTANCE


def _get_llm():
    """Synchronous accessor for the LLM singleton (for backward compat with tests).

    If the singleton is not yet initialised, constructs it synchronously using
    the same provider logic as ``_get_llm_async``.  This path is only hit during
    test setup or on the first call from a sync context.
    """
    global _LLM_INSTANCE
    if _LLM_INSTANCE is not None:
        return _LLM_INSTANCE

    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        _LLM_INSTANCE = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _LLM_INSTANCE = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        _LLM_INSTANCE = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )
    else:
        from langchain_anthropic import ChatAnthropic
        _LLM_INSTANCE = ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.3,
        )

    return _LLM_INSTANCE


# ══════════════════════════════════════════════════════════════════════════════
# Data models
# ══════════════════════════════════════════════════════════════════════════════

class Lead(BaseModel):
    """Validated lead data collected during the sign-up flow."""

    name: str = ""
    email: str = ""
    platform: str = ""

    @field_validator("email", mode="before")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v


class _EmailCheck(BaseModel):
    """Used exclusively to validate an email string via Pydantic's EmailStr."""

    email: EmailStr

    @field_validator("email", mode="before")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v


class LeadFormSchema(BaseModel):
    """Structured output for LLM-driven lead form extraction.

    Fields left as None indicate the user has not yet provided that value.
    """

    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None


class Intent(str, Enum):
    greeting = "greeting"
    product_inquiry = "product_inquiry"
    high_intent = "high_intent"


class IntentResponse(BaseModel):
    """Structured output returned by the intent-classification LLM call."""

    intent: Intent


class AgentState(BaseModel):
    """Full runtime state of the AutoStream agent, validated by Pydantic.

    New in v2:
        ``consecutive_failures`` tracks back-to-back LLM errors for the
        circuit breaker.  It resets to 0 on any successful node execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: Annotated[List[BaseMessage], add_messages] = []
    intent: Intent = Intent.greeting
    lead_stage: Literal["none", "collecting", "done"] = "none"
    lead: Lead = Lead()
    rag_context: str = ""
    consecutive_failures: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Retry helper
# ══════════════════════════════════════════════════════════════════════════════

KEEP_LAST_N = 10


async def _with_retry(coro_fn, *, max_attempts: int = 3, base_delay: float = 1.0):
    """Retry an async callable with full-jitter exponential backoff.

    Retries up to ``max_attempts`` times when an exception is raised.  Each
    wait uses full-jitter: ``sleep(uniform(0, base_delay * 2 ** attempt))``
    so that concurrent retries do not produce a thundering herd.

    Args:
        coro_fn:      Zero-argument async callable (lambda or partial) to invoke.
        max_attempts: Total number of attempts before re-raising.  Default: 3.
        base_delay:   Base delay in seconds for the backoff formula.  Default: 1.0.

    Returns:
        The return value of ``coro_fn()`` on success.

    Raises:
        The last exception raised by ``coro_fn`` after all attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await coro_fn()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = random.uniform(0, base_delay * (2 ** attempt))
                await asyncio.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ══════════════════════════════════════════════════════════════════════════════
# Context management
# ══════════════════════════════════════════════════════════════════════════════

_SUMMARY_PROMPT = """Summarise the following conversation between a user and AutoStream's
sales assistant Alex. Be concise. Capture: the user's name (if shared), their main
questions or concerns, and any decisions or information already exchanged.
Do NOT include the most recent exchange — only prior context.

CONVERSATION:
{history}"""


async def _trim_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """Sliding-window context management.

    Returns messages unchanged when len <= KEEP_LAST_N. Otherwise condenses
    older turns into a single SystemMessage summary and appends the most recent
    KEEP_LAST_N messages, keeping every LLM call within a predictable token budget.
    """
    if len(messages) <= KEEP_LAST_N:
        return messages

    older = messages[:-KEEP_LAST_N]
    recent = messages[-KEEP_LAST_N:]

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in older
        if isinstance(m, (HumanMessage, AIMessage))
    )

    llm = await _get_llm_async()
    summary_response = await _with_retry(
        lambda: llm.ainvoke([
            SystemMessage(content=_SUMMARY_PROMPT.format(history=history_text))
        ])
    )
    summary_msg = SystemMessage(
        content=f"[Conversation Summary — earlier turns]\n{summary_response.content}"
    )
    return [summary_msg] + list(recent)


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 5 — Deterministic Intent Classifier with Few-Shot Examples
# ══════════════════════════════════════════════════════════════════════════════

_INTENT_PROMPT = """You are an intent classifier for AutoStream, a video editing SaaS.
Classify the user's latest message into exactly ONE of these intents:

- greeting        — casual hello, how are you, general chit-chat
- product_inquiry — questions about features, pricing, plans, policies, or how AutoStream works
- high_intent     — user explicitly wants to sign up, start a trial, buy a plan, or try the product

EXAMPLES:
User: "Hey, what's up?"              → greeting
User: "Good morning!"                → greeting
User: "How much does the Pro plan cost?"  → product_inquiry
User: "Do you support TikTok exports?"    → product_inquiry
User: "I want to sign up right now."      → high_intent
User: "Let's get started, I'll take the Pro plan." → high_intent

Classify the following message. Respond with the intent label only."""

# Intent LLM singleton — separate from the main LLM so we can pin temperature=0
# without affecting conversational responses.
_INTENT_LLM_INSTANCE = None


def _get_intent_llm():
    """Return the cached intent-classification LLM (lazy singleton, temperature=0).

    We clone the base LLM with ``temperature=0`` so the classifier is fully
    deterministic — the same borderline message always resolves to the same label.
    The structured output wrapper is applied here so callers get an
    ``IntentResponse`` object directly.
    """
    global _INTENT_LLM_INSTANCE
    if _INTENT_LLM_INSTANCE is None:
        base = _get_llm()
        # bind_tools / with_config temperature override — works across providers.
        try:
            cold = base.bind(temperature=0)
        except Exception:
            cold = base  # fallback: provider doesn't support bind-time temperature
        _INTENT_LLM_INSTANCE = cold.with_structured_output(IntentResponse)
    return _INTENT_LLM_INSTANCE


# Lead extraction LLM singleton
_EXTRACT_LLM_INSTANCE = None


def _get_extract_llm():
    """Return the cached lead-extraction LLM (lazy singleton)."""
    global _EXTRACT_LLM_INSTANCE
    if _EXTRACT_LLM_INSTANCE is None:
        _EXTRACT_LLM_INSTANCE = _get_llm().with_structured_output(LeadFormSchema)
    return _EXTRACT_LLM_INSTANCE


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 3 — Safe wrappers with graceful degradation
# ══════════════════════════════════════════════════════════════════════════════

async def _safe_detect_intent(state: "AgentState") -> "IntentResponse":
    """Detect intent with full-jitter retry and a safe fallback.

    On failure (after all retries), logs a warning and returns
    ``product_inquiry`` — the safest default because it will route to the RAG
    node, which will try to answer helpfully rather than silently dropping the
    user into a greeting loop.
    """
    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    )
    try:
        return await _with_retry(
            lambda: _get_intent_llm().ainvoke([
                SystemMessage(content=_INTENT_PROMPT),
                HumanMessage(content=last_human),
            ])
        )
    except Exception as exc:
        logger.warning(
            "[intent] All retries exhausted, falling back to product_inquiry. "
            "Error: %s", exc
        )
        return IntentResponse(intent=Intent.product_inquiry)


async def _safe_extract_lead(last_human: str) -> "LeadFormSchema":
    """Extract lead fields with a safe empty fallback.

    On failure, returns ``LeadFormSchema()`` (all None) so that the lead capture
    node treats no new information as available and re-asks the user for all
    missing fields — rather than crashing the turn.
    """
    try:
        return await _with_retry(
            lambda: _get_extract_llm().ainvoke([
                SystemMessage(content=_LEAD_EXTRACT_SYSTEM),
                HumanMessage(content=last_human),
            ])
        )
    except Exception as exc:
        logger.warning(
            "[lead_extract] All retries exhausted, returning empty schema. "
            "Error: %s", exc
        )
        return LeadFormSchema()


def _check_circuit_breaker(state: "AgentState") -> Optional[str]:
    """Return a user-facing message if the circuit is open, else None.

    The circuit opens once ``consecutive_failures`` reaches
    ``_CIRCUIT_BREAKER_THRESHOLD``.  The message instructs the user to retry
    in a moment and is surfaced directly by the calling node.
    """
    if state.consecutive_failures >= _CIRCUIT_BREAKER_THRESHOLD:
        return (
            "⚠️ I'm having trouble connecting to my AI backend right now. "
            "Please try again in a moment — I'll be right back! 🙏"
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Graph nodes
# ══════════════════════════════════════════════════════════════════════════════

async def detect_intent(state: AgentState) -> AgentState:
    """Classify user intent via structured LLM output (with safe fallback)."""
    circuit_msg = _check_circuit_breaker(state)
    if circuit_msg:
        # Circuit open: reset counter and surface error message.
        return state.model_copy(update={
            "consecutive_failures": 0,
            "messages": [AIMessage(content=circuit_msg)],
        })

    result = await _safe_detect_intent(state)
    return state.model_copy(update={"intent": result.intent})


def route(state: AgentState) -> Literal["greeter", "rag_answer", "lead_capture"]:
    """Route to the appropriate node based on intent and lead stage."""
    if state.lead_stage in ("collecting", "done"):
        return "lead_capture"
    if state.intent == Intent.high_intent:
        return "lead_capture"
    if state.intent == Intent.product_inquiry:
        return "rag_answer"
    return "greeter"


# ── Greeter ────────────────────────────────────────────────────────────────────

_GREETER_SYSTEM = """You are Alex, a friendly and knowledgeable sales assistant for AutoStream —
an AI-powered video editing SaaS for content creators.

Keep responses concise, warm, and helpful. If the user seems interested in the product,
mention you'd be happy to share pricing details or help them get started."""


async def greeter_node(state: AgentState) -> AgentState:
    context_msgs = await _trim_messages(state.messages)
    llm = await _get_llm_async()
    try:
        response = await _with_retry(
            lambda: llm.ainvoke([
                SystemMessage(content=_GREETER_SYSTEM),
                *context_msgs,
            ])
        )
        return state.model_copy(update={
            "consecutive_failures": 0,
            "messages": [AIMessage(content=response.content)],
        })
    except Exception as exc:
        logger.error("[greeter] Failed after retries: %s", exc)
        new_failures = state.consecutive_failures + 1
        fallback = "Hi! I'm having a small hiccup — could you repeat that? 😊"
        return state.model_copy(update={
            "consecutive_failures": new_failures,
            "messages": [AIMessage(content=fallback)],
        })


# ── RAG Answer ─────────────────────────────────────────────────────────────────

_RAG_SYSTEM = """You are Alex, a knowledgeable sales assistant for AutoStream —
an AI-powered video editing SaaS. Answer the user's question using ONLY the knowledge
base context provided below. Be concise, accurate, and helpful.

If the user seems interested in signing up after your answer, invite them to do so.

KNOWLEDGE BASE CONTEXT:
{context}
"""


async def rag_answer_node(state: AgentState) -> AgentState:
    """Answer product questions with hybrid-retrieved context.

    Improvement 1: RAG retrieval and message trimming run **concurrently** via
    ``asyncio.gather`` — saving the sequential latency of running them one after
    the other.
    """
    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    )

    # ── Parallel fetch: RAG context + trimmed message history ─────────────────
    context, context_msgs = await asyncio.gather(
        retrieve_knowledge(last_human),
        _trim_messages(state.messages),
    )

    llm = await _get_llm_async()
    try:
        response = await _with_retry(
            lambda: llm.ainvoke([
                SystemMessage(content=_RAG_SYSTEM.format(context=context)),
                *context_msgs,
            ])
        )
        return state.model_copy(update={
            "consecutive_failures": 0,
            "rag_context": context,
            "messages": [AIMessage(content=response.content)],
        })
    except Exception as exc:
        logger.error("[rag_answer] Failed after retries: %s", exc)
        new_failures = state.consecutive_failures + 1
        fallback = (
            "I couldn't look that up right now — please try again in a moment. "
            "If it's urgent, feel free to email us at support@autostream.io 🙏"
        )
        return state.model_copy(update={
            "consecutive_failures": new_failures,
            "rag_context": "",
            "messages": [AIMessage(content=fallback)],
        })


# ── Lead Capture ───────────────────────────────────────────────────────────────

_LEAD_EXTRACT_SYSTEM = """Extract sign-up information from the user's message.
Return ONLY the fields explicitly mentioned. Leave any unmentioned field as null.

Fields to extract:
- name: the user's full name
- email: the user's email address
- platform: the content platform they primarily create on (e.g. YouTube, TikTok, Instagram)"""

_LEAD_ASK_SYSTEM = """You are Alex, a warm and friendly sales assistant for AutoStream — \
an AI-powered video editing SaaS for content creators.

You are helping a user sign up for AutoStream Pro.

Already collected: {collected}
Still needed: {missing}

Ask the user for the missing information in a natural, conversational tone.
Do NOT ask for or mention fields you already have.
If multiple fields are missing, you can ask for them together in one friendly message."""


async def lead_capture_node(state: AgentState) -> AgentState:
    """LLM-driven lead form filling.

    Each turn:
      1. Extracts any name/email/platform values from the user's message
         (with safe fallback on LLM failure).
      2. Merges extracted values with existing lead data.
      3. Validates the email via _EmailCheck.
      4. Asks for any still-missing fields naturally.
      5. Fires mock_lead_capture() once all three fields are present and valid.
    """
    if state.lead_stage == "done":
        context_msgs = await _trim_messages(state.messages)
        llm = await _get_llm_async()
        try:
            response = await _with_retry(
                lambda: llm.ainvoke([
                    SystemMessage(
                        content="You are Alex from AutoStream. The user has already signed up as a lead. "
                                "Be warm, answer any remaining questions, and let them know the team will be in touch."
                    ),
                    *context_msgs,
                ])
            )
            return state.model_copy(update={
                "consecutive_failures": 0,
                "messages": [AIMessage(content=response.content)],
            })
        except Exception as exc:
            logger.error("[lead_capture/done] Failed after retries: %s", exc)
            return state.model_copy(update={
                "consecutive_failures": state.consecutive_failures + 1,
                "messages": [AIMessage(
                    content="You're all set! Our team will be in touch soon. 🚀"
                )],
            })

    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    ).strip()

    # Safe extraction — never raises; returns empty schema on failure.
    extracted: LeadFormSchema = await _safe_extract_lead(last_human)

    current = state.lead
    name = (extracted.name or "").strip() or current.name
    platform = (extracted.platform or "").strip() or current.platform

    raw_email = (extracted.email or "").strip()
    valid_email = current.email
    email_invalid = False

    if raw_email and raw_email.lower() != current.email:
        try:
            _EmailCheck(email=raw_email)
            valid_email = raw_email.lower()
        except ValidationError:
            email_invalid = True

    updated_lead = Lead(name=name, email=valid_email, platform=platform)

    missing: list[str] = []
    if not updated_lead.name:
        missing.append("your full name")
    if email_invalid:
        missing.append(f"a valid email address — '{raw_email}' didn't look right")
    elif not updated_lead.email:
        missing.append("your email address")
    if not updated_lead.platform:
        missing.append("the platform you primarily create content on (e.g. YouTube, TikTok, Instagram)")

    if not missing:
        result = mock_lead_capture(
            name=updated_lead.name,
            email=updated_lead.email,
            platform=updated_lead.platform,
        )
        return state.model_copy(update={
            "consecutive_failures": 0,
            "lead": updated_lead,
            "lead_stage": "done",
            "messages": [AIMessage(content=result)],
        })

    collected_parts: list[str] = []
    if updated_lead.name:
        collected_parts.append(f"name: {updated_lead.name}")
    if updated_lead.email:
        collected_parts.append(f"email: {updated_lead.email}")
    if updated_lead.platform:
        collected_parts.append(f"platform: {updated_lead.platform}")

    collected_str = ", ".join(collected_parts) if collected_parts else "nothing yet"
    missing_str = " and ".join(missing)

    llm = await _get_llm_async()
    try:
        response = await _with_retry(
            lambda: llm.ainvoke([
                SystemMessage(content=_LEAD_ASK_SYSTEM.format(
                    collected=collected_str,
                    missing=missing_str,
                )),
                *state.messages[-4:],
            ])
        )
        return state.model_copy(update={
            "consecutive_failures": 0,
            "lead": updated_lead,
            "lead_stage": "collecting",
            "messages": [AIMessage(content=response.content)],
        })
    except Exception as exc:
        logger.error("[lead_capture/ask] Failed after retries: %s", exc)
        return state.model_copy(update={
            "consecutive_failures": state.consecutive_failures + 1,
            "lead": updated_lead,
            "lead_stage": "collecting",
            "messages": [AIMessage(
                content="I had a moment of trouble there — could you share your details again? 🙏"
            )],
        })


# ══════════════════════════════════════════════════════════════════════════════
# Improvement 4 — Streaming abstraction utilities
# ══════════════════════════════════════════════════════════════════════════════

def _extract_chunk_text(chunk) -> str:
    """Normalise a streaming chunk's content to a plain string.

    Different LLM providers return content in different shapes:
    - Anthropic / OpenAI / Groq: ``chunk.content`` is a ``str``.
    - Google Gemini: ``chunk.content`` is a ``list[dict]`` where each dict has
      ``{"type": "text", "text": "..."}`` entries.

    This utility handles both cases and returns an empty string for anything
    unrecognised, making it safe to call unconditionally inside a stream loop.
    """
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


async def stream_response(
    graph,
    state: AgentState,
) -> AsyncGenerator[Tuple[str, Optional[AgentState]], None]:
    """Async generator that streams token-by-token text and yields the final state.

    Encapsulates all ``astream_events`` parsing so that ``main.py`` becomes a
    thin display layer with no knowledge of LangGraph internals.

    Yields:
        ``(text_chunk, None)`` for every streamed token.
        ``("", final_state)`` exactly once at the end with the completed state.

    Example (in main.py)::

        async for text, final in stream_response(graph, state):
            if final is not None:
                state = final
            else:
                print(text, end="", flush=True)
    """
    final_state: Optional[AgentState] = None

    async for event in graph.astream_events(state, version="v2"):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            text = _extract_chunk_text(chunk)
            if text:
                yield text, None

        elif kind == "on_chain_end" and event.get("name") == "LangGraph":
            raw = event["data"].get("output")
            if raw is not None:
                if isinstance(raw, dict):
                    final_state = AgentState(**raw)
                elif isinstance(raw, AgentState):
                    final_state = raw

    yield "", final_state


# ══════════════════════════════════════════════════════════════════════════════
# Graph construction
# ══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Compile and return the LangGraph state machine."""
    graph = StateGraph(AgentState)

    graph.add_node("intent_detector", detect_intent)
    graph.add_node("greeter", greeter_node)
    graph.add_node("rag_answer", rag_answer_node)
    graph.add_node("lead_capture", lead_capture_node)

    graph.set_entry_point("intent_detector")

    graph.add_conditional_edges(
        "intent_detector",
        route,
        {
            "greeter": "greeter",
            "rag_answer": "rag_answer",
            "lead_capture": "lead_capture",
        },
    )

    graph.add_edge("greeter", END)
    graph.add_edge("rag_answer", END)
    graph.add_edge("lead_capture", END)

    return graph.compile()
