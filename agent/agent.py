"""
agent.py — AutoStream Conversational AI Agent
Built with LangGraph for state management across multi-turn conversations.

Architecture:
  ┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
  │  User Input │────▶│  Intent Detector │────▶│  Router Node  │
  └─────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
                        ┌──────────────────────────────┼──────────────────────┐
                        ▼                               ▼                      ▼
               ┌────────────────┐           ┌───────────────────┐   ┌──────────────────┐
               │  Greeter Node  │           │  RAG Answer Node  │   │  Lead Capture    │
               └────────────────┘           └───────────────────┘   │  Node (LLM-form) │
                                                                     └──────────────────┘

Improvements:
  - AgentState backed by Pydantic BaseModel for runtime validation & coercion.
  - Lead model uses EmailStr for declarative email validation (no manual regex).
  - Intent detection uses .with_structured_output(IntentResponse) so the LLM
    returns a typed Enum, eliminating string-parsing fragility.
  - _trim_messages() keeps the last KEEP_LAST_N messages verbatim and condenses
    older turns into a single SystemMessage summary, bounding context cost.
  - Lead capture uses LLM-driven form filling (LeadFormSchema structured output)
    — dynamically extracts name/email/platform from any user message and asks
    only for what is still missing. No more manual FSM if/elif chains.
  - All node functions are async; main.py streams tokens via astream_events().
"""

import os
import asyncio
from enum import Enum
from typing import Annotated, Literal, List, Optional
from dotenv import load_dotenv

from pydantic import BaseModel, EmailStr, field_validator, ConfigDict, ValidationError
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools.tools import retrieve_knowledge, mock_lead_capture

load_dotenv()

# ── LLM Setup ────────────────────────────────────────────────────────────────
# Supports OpenAI (GPT-4o-mini), Google (Gemini 2.0 Flash), Groq, or Anthropic
# Set LLM_PROVIDER in .env: "openai" | "google" | "groq" | "anthropic"

def _build_llm():
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3,
        )
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.3,
        )
    else:  # default: anthropic
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.3,
        )


llm = _build_llm()


# ── Lead Model (Pydantic) ────────────────────────────────────────────────────

class Lead(BaseModel):
    """Validated lead data collected during the sign-up flow."""
    name: str = ""
    email: str = ""     # raw string; validation is done via _EmailCheck before storing
    platform: str = ""

    @field_validator("email", mode="before")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        """Lowercase + strip whitespace before storing."""
        return v.strip().lower() if isinstance(v, str) else v


class _EmailCheck(BaseModel):
    """Single-purpose model used only to validate an email string via EmailStr.
    EmailStr raises ValidationError on malformed input — we catch that upstream.
    """
    email: EmailStr

    @field_validator("email", mode="before")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v


class LeadFormSchema(BaseModel):
    """
    Structured output for LLM-driven lead form extraction.

    The LLM is asked to extract whichever fields the user has already
    mentioned in their message. Fields not mentioned remain None so they
    are NOT accidentally overwritten in the merged Lead object.
    """
    name: Optional[str] = None
    email: Optional[str] = None
    platform: Optional[str] = None


# ── Intent Schema (Structured Output) ────────────────────────────────────────

class Intent(str, Enum):
    greeting        = "greeting"
    product_inquiry = "product_inquiry"
    high_intent     = "high_intent"

class IntentResponse(BaseModel):
    """Structured output the LLM must return for intent classification."""
    intent: Intent


# ── Agent State (Pydantic) ────────────────────────────────────────────────────

class AgentState(BaseModel):
    """Full runtime state of the AutoStream agent, validated by Pydantic."""
    model_config = ConfigDict(arbitrary_types_allowed=True)  # needed for LangChain BaseMessage

    messages:    Annotated[List[BaseMessage], add_messages] = []
    intent:      Intent = Intent.greeting
    lead_stage:  Literal["none", "collecting", "done"] = "none"
    lead:        Lead = Lead()
    rag_context: str = ""


# ── Conversation Memory ───────────────────────────────────────────────────────

# Number of most-recent messages always kept verbatim in the context window.
# Older messages are condensed into a single SystemMessage summary.
KEEP_LAST_N = 10

_SUMMARY_PROMPT = """Summarise the following conversation between a user and AutoStream's
sales assistant Alex. Be concise. Capture: the user's name (if shared), their main
questions or concerns, and any decisions or information already exchanged.
Do NOT include the most recent exchange — only prior context.

CONVERSATION:
{history}"""


async def _trim_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    """
    Sliding-window context management.

    If len(messages) <= KEEP_LAST_N: returns the list unchanged.
    Otherwise:
      1. Formats the older messages as a plain-text history string.
      2. Asks the LLM to summarise them into a single SystemMessage.
      3. Returns [summary_msg] + last KEEP_LAST_N messages.

    This keeps every LLM call within a predictable token budget regardless
    of how long the conversation runs.
    """
    if len(messages) <= KEEP_LAST_N:
        return messages

    older  = messages[:-KEEP_LAST_N]
    recent = messages[-KEEP_LAST_N:]

    history_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Agent'}: {m.content}"
        for m in older
        if isinstance(m, (HumanMessage, AIMessage))
    )

    summary_response = await llm.ainvoke([
        SystemMessage(content=_SUMMARY_PROMPT.format(history=history_text))
    ])
    summary_msg = SystemMessage(
        content=f"[Conversation Summary — earlier turns]\n{summary_response.content}"
    )
    return [summary_msg] + list(recent)


# ── Intent Detection (Structured Output) ─────────────────────────────────────

_INTENT_PROMPT = """You are an intent classifier for AutoStream, a video editing SaaS.
Classify the user's latest message into exactly ONE of these intents:

- greeting        — casual hello, how are you, general chit-chat
- product_inquiry — questions about features, pricing, plans, policies, or how AutoStream works
- high_intent     — user explicitly wants to sign up, start a trial, buy a plan, or try the product"""

# The LLM is bound to return a valid IntentResponse — no string parsing needed.
_intent_llm = llm.with_structured_output(IntentResponse)

async def detect_intent(state: AgentState) -> AgentState:
    """Classify user intent via structured LLM output (no regex / string parsing)."""
    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    )

    result: IntentResponse = await _intent_llm.ainvoke([
        SystemMessage(content=_INTENT_PROMPT),
        HumanMessage(content=last_human),
    ])

    return state.model_copy(update={"intent": result.intent})


# ── Router ────────────────────────────────────────────────────────────────────

def route(state: AgentState) -> Literal["greeter", "rag_answer", "lead_capture"]:
    """Decide which node handles this turn."""
    if state.lead_stage in ("collecting", "done"):
        return "lead_capture"
    if state.intent == Intent.high_intent:
        return "lead_capture"
    if state.intent == Intent.product_inquiry:
        return "rag_answer"
    return "greeter"


# ── Node: Greeter ─────────────────────────────────────────────────────────────

_GREETER_SYSTEM = """You are Alex, a friendly and knowledgeable sales assistant for AutoStream —
an AI-powered video editing SaaS for content creators.

Keep responses concise, warm, and helpful. If the user seems interested in the product,
mention you'd be happy to share pricing details or help them get started."""

async def greeter_node(state: AgentState) -> AgentState:
    context_msgs = await _trim_messages(state.messages)
    response = await llm.ainvoke([
        SystemMessage(content=_GREETER_SYSTEM),
        *context_msgs,
    ])
    return state.model_copy(update={"messages": [AIMessage(content=response.content)]})


# ── Node: RAG Answer ──────────────────────────────────────────────────────────

_RAG_SYSTEM = """You are Alex, a knowledgeable sales assistant for AutoStream —
an AI-powered video editing SaaS. Answer the user's question using ONLY the knowledge
base context provided below. Be concise, accurate, and helpful.

If the user seems interested in signing up after your answer, invite them to do so.

KNOWLEDGE BASE CONTEXT:
{context}
"""

async def rag_answer_node(state: AgentState) -> AgentState:
    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    )
    context = retrieve_knowledge(last_human)
    context_msgs = await _trim_messages(state.messages)

    response = await llm.ainvoke([
        SystemMessage(content=_RAG_SYSTEM.format(context=context)),
        *context_msgs,
    ])
    return state.model_copy(update={
        "rag_context": context,
        "messages": [AIMessage(content=response.content)],
    })


# ── Node: Lead Capture (LLM-driven form extraction) ──────────────────────────
#
# Instead of a rigid FSM (collecting_name → collecting_email → …), the LLM
# dynamically extracts whichever fields the user has provided in their message
# and asks only for what is still missing. This handles natural multi-field
# inputs ("I'm Ronit, ronit@example.com, I post on YouTube") in a single turn.

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

# Bind the structured-output extractor once at module load.
_extract_llm = llm.with_structured_output(LeadFormSchema)


async def lead_capture_node(state: AgentState) -> AgentState:
    """
    LLM-driven lead form filling.

    Each turn:
      1. The LLM extracts any name/email/platform values from the user's message.
      2. Extracted values are merged with existing lead data.
      3. Email is validated via _EmailCheck (strict EmailStr).
      4. If any field is still missing → LLM asks for it naturally.
      5. When all three fields are present and valid → fire mock_lead_capture().
    """
    # ── Stage: done — post-signup follow-up ─────────────────────────────────
    if state.lead_stage == "done":
        context_msgs = await _trim_messages(state.messages)
        response = await llm.ainvoke([
            SystemMessage(
                content="You are Alex from AutoStream. The user has already signed up as a lead. "
                        "Be warm, answer any remaining questions, and let them know the team will be in touch."
            ),
            *context_msgs,
        ])
        return state.model_copy(update={"messages": [AIMessage(content=response.content)]})

    # ── Extract fields from the latest user message ──────────────────────────
    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    ).strip()

    extracted: LeadFormSchema = await _extract_llm.ainvoke([
        SystemMessage(content=_LEAD_EXTRACT_SYSTEM),
        HumanMessage(content=last_human),
    ])

    # ── Merge extracted fields with existing lead (non-empty wins) ───────────
    current = state.lead
    name     = (extracted.name     or "").strip() or current.name
    platform = (extracted.platform or "").strip() or current.platform

    # Email needs validation before being stored
    raw_email = (extracted.email or "").strip()
    valid_email = current.email   # start with whatever was already accepted
    email_invalid = False

    if raw_email and raw_email.lower() != current.email:
        try:
            _EmailCheck(email=raw_email)
            valid_email = raw_email.lower()
        except ValidationError:
            email_invalid = True   # tell the user below

    updated_lead = Lead(name=name, email=valid_email, platform=platform)

    # ── Determine which fields are still missing ─────────────────────────────
    missing: list[str] = []
    if not updated_lead.name:
        missing.append("your full name")
    if email_invalid:
        missing.append(f"a valid email address — '{raw_email}' didn't look right")
    elif not updated_lead.email:
        missing.append("your email address")
    if not updated_lead.platform:
        missing.append("the platform you primarily create content on (e.g. YouTube, TikTok, Instagram)")

    # ── All fields collected → fire the tool ────────────────────────────────
    if not missing:
        result = mock_lead_capture(
            name=updated_lead.name,
            email=updated_lead.email,
            platform=updated_lead.platform,
        )
        return state.model_copy(update={
            "lead":       updated_lead,
            "lead_stage": "done",
            "messages":   [AIMessage(content=result)],
        })

    # ── Ask for missing fields naturally ─────────────────────────────────────
    collected_parts: list[str] = []
    if updated_lead.name:
        collected_parts.append(f"name: {updated_lead.name}")
    if updated_lead.email:
        collected_parts.append(f"email: {updated_lead.email}")
    if updated_lead.platform:
        collected_parts.append(f"platform: {updated_lead.platform}")

    collected_str = ", ".join(collected_parts) if collected_parts else "nothing yet"
    missing_str   = " and ".join(missing)

    response = await llm.ainvoke([
        SystemMessage(content=_LEAD_ASK_SYSTEM.format(
            collected=collected_str,
            missing=missing_str,
        )),
        *state.messages[-4:],   # small recent window — keep prompt tight
    ])

    return state.model_copy(update={
        "lead":       updated_lead,
        "lead_stage": "collecting",
        "messages":   [AIMessage(content=response.content)],
    })


# ── Build LangGraph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("intent_detector", detect_intent)
    graph.add_node("greeter",         greeter_node)
    graph.add_node("rag_answer",      rag_answer_node)
    graph.add_node("lead_capture",    lead_capture_node)

    graph.set_entry_point("intent_detector")

    graph.add_conditional_edges(
        "intent_detector",
        route,
        {
            "greeter":       "greeter",
            "rag_answer":    "rag_answer",
            "lead_capture":  "lead_capture",
        },
    )

    graph.add_edge("greeter",       END)
    graph.add_edge("rag_answer",    END)
    graph.add_edge("lead_capture",  END)

    return graph.compile()
