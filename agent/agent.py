"""
agent.py — AutoStream Conversational AI Agent
Built with LangGraph for state management across multi-turn conversations.

Architecture:
  ┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
  │  User Input │────▶│  Intent Detector │────▶│  Router Node  │
  └─────────────┘     └──────────────────┘     └───────┬───────┘
                                                         │
                        ┌───────────────────────────────┼─────────────────────┐
                        ▼                               ▼                     ▼
               ┌────────────────┐           ┌───────────────────┐   ┌──────────────────┐
               │  Greeter Node  │           │  RAG Answer Node  │   │  Lead Capture    │
               └────────────────┘           └───────────────────┘   │  Node (3-step)   │
                                                                     └──────────────────┘

Improvements:
  - AgentState backed by Pydantic BaseModel for runtime validation & coercion.
  - Lead model uses EmailStr for declarative email validation (no manual regex).
  - Intent detection uses .with_structured_output(IntentResponse) so the LLM
    returns a typed Enum, eliminating string-parsing fragility.
"""

import os
from enum import Enum
from typing import Annotated, Literal, List
from dotenv import load_dotenv

from pydantic import BaseModel, EmailStr, field_validator, ConfigDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from tools.tools import retrieve_knowledge, mock_lead_capture

load_dotenv()

# ── LLM Setup ────────────────────────────────────────────────────────────────
# Supports OpenAI (GPT-4o-mini), Google (Gemini 1.5 Flash), or Anthropic (Claude 3 Haiku)
# Set LLM_PROVIDER in .env: "openai" | "google" | "anthropic"

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

    messages: Annotated[List[BaseMessage], add_messages] = []  # full conversation history
    intent: Intent = Intent.greeting                            # current classified intent
    lead_stage: Literal[
        "none", "collecting_name", "collecting_email",
        "collecting_platform", "done"
    ] = "none"                                                  # lead capture FSM state
    lead: Lead = Lead()                                         # validated lead data
    rag_context: str = ""                                       # retrieved knowledge for current turn


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


def _trim_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
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

    summary_response = llm.invoke([
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
- high_intent     — user explicitly wants to sign up, start a trial, buy a plan, or try the product"""

# The LLM is bound to return a valid IntentResponse — no string parsing needed.
_intent_llm = llm.with_structured_output(IntentResponse)

def detect_intent(state: AgentState) -> AgentState:
    """Classify user intent via structured LLM output (no regex / string parsing)."""
    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    )

    result: IntentResponse = _intent_llm.invoke([
        SystemMessage(content=_INTENT_PROMPT),
        HumanMessage(content=last_human),
    ])

    return state.model_copy(update={"intent": result.intent})


# ── Router ────────────────────────────────────────────────────────────────────

def route(state: AgentState) -> Literal["greeter", "rag_answer", "lead_capture"]:
    """Decide which node handles this turn."""
    if state.lead_stage not in ("none", "done"):
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

def greeter_node(state: AgentState) -> AgentState:
    context_msgs = _trim_messages(state.messages)
    response = llm.invoke([
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

def rag_answer_node(state: AgentState) -> AgentState:
    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    )
    context = retrieve_knowledge(last_human)
    context_msgs = _trim_messages(state.messages)

    prompt = _RAG_SYSTEM.format(context=context)
    response = llm.invoke([
        SystemMessage(content=prompt),
        *context_msgs,
    ])
    return state.model_copy(update={
        "rag_context": context,
        "messages": [AIMessage(content=response.content)],
    })


# ── Node: Lead Capture ────────────────────────────────────────────────────────

def lead_capture_node(state: AgentState) -> AgentState:
    stage = state.lead_stage

    # ── Stage: none → start collection ──────────────────────────────────────
    if stage == "none":
        msg = (
            "Great to hear you're interested in AutoStream Pro! 🎉\n\n"
            "I'd love to get you set up. Could I start with your **full name**?"
        )
        return state.model_copy(update={
            "lead_stage": "collecting_name",
            "messages": [AIMessage(content=msg)],
        })

    last_human = next(
        (m.content for m in reversed(state.messages) if isinstance(m, HumanMessage)),
        ""
    ).strip()

    # ── Stage: collecting_name ───────────────────────────────────────────────
    if stage == "collecting_name":
        updated_lead = state.lead.model_copy(update={"name": last_human})
        return state.model_copy(update={
            "lead": updated_lead,
            "lead_stage": "collecting_email",
            "messages": [AIMessage(content=f"Nice to meet you, **{last_human}**! 👋\n\nWhat's your **email address**?")],
        })

    # ── Stage: collecting_email ──────────────────────────────────────────────
    if stage == "collecting_email":
        # Use the dedicated _EmailCheck model (strict EmailStr) for validation.
        # Lead.email is a plain str field, so we validate here before storing.
        from pydantic import ValidationError
        try:
            _EmailCheck(email=last_human)   # raises ValidationError if malformed
        except ValidationError:
            return state.model_copy(update={
                "messages": [AIMessage(content="Hmm, that doesn't look like a valid email. Could you double-check and re-enter it?")],
            })
        updated_lead = state.lead.model_copy(update={"email": last_human})
        return state.model_copy(update={
            "lead": updated_lead,
            "lead_stage": "collecting_platform",
            "messages": [AIMessage(content="Got it! Last question — which platform do you primarily create content on?\n(e.g., YouTube, Instagram, TikTok, LinkedIn…)")],
        })

    # ── Stage: collecting_platform → fire tool ───────────────────────────────
    if stage == "collecting_platform":
        updated_lead = state.lead.model_copy(update={"platform": last_human})
        result = mock_lead_capture(
            name=updated_lead.name,
            email=str(updated_lead.email),
            platform=updated_lead.platform,
        )
        return state.model_copy(update={
            "lead": updated_lead,
            "lead_stage": "done",
            "messages": [AIMessage(content=result)],
        })

    # ── Stage: done — follow-up conversation ─────────────────────────────────
    context_msgs = _trim_messages(state.messages)
    response = llm.invoke([
        SystemMessage(content="You are Alex from AutoStream. The user has already signed up as a lead. "
                               "Be warm, answer any remaining questions, and let them know the team will be in touch."),
        *context_msgs,
    ])
    return state.model_copy(update={"messages": [AIMessage(content=response.content)]})


# ── Build LangGraph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
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
