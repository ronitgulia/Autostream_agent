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
"""

import os
import re
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
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


# ── Agent State ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]          # full conversation history
    intent: str                                       # "greeting" | "product_inquiry" | "high_intent"
    lead_stage: str                                   # "none" | "collecting_name" | "collecting_email" | "collecting_platform" | "done"
    lead_name: str
    lead_email: str
    lead_platform: str
    rag_context: str                                  # retrieved knowledge for current turn


# ── Intent Detection ─────────────────────────────────────────────────────────

_INTENT_PROMPT = """You are an intent classifier for AutoStream, a video editing SaaS.
Classify the user's latest message into exactly ONE of these intents:

1. greeting        — casual hello, how are you, general chit-chat
2. product_inquiry — questions about features, pricing, plans, policies, or how AutoStream works
3. high_intent     — user explicitly wants to sign up, start a trial, buy a plan, or try the product

Respond with ONLY the intent label (no explanation, no punctuation):
greeting | product_inquiry | high_intent
"""

def detect_intent(state: AgentState) -> AgentState:
    """Classify user intent from latest message."""
    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )

    response = llm.invoke([
        SystemMessage(content=_INTENT_PROMPT),
        HumanMessage(content=last_human),
    ])
    raw = response.content.strip().lower()

    # Normalise
    if "high" in raw or "sign" in raw or "buy" in raw:
        intent = "high_intent"
    elif "product" in raw or "inquiry" in raw:
        intent = "product_inquiry"
    else:
        intent = "greeting"

    return {**state, "intent": intent}


# ── Router ────────────────────────────────────────────────────────────────────

def route(state: AgentState) -> Literal["greeter", "rag_answer", "lead_capture"]:
    """Decide which node handles this turn."""
    if state["lead_stage"] not in ("none", "done"):
        return "lead_capture"
    if state["intent"] == "high_intent":
        return "lead_capture"
    if state["intent"] == "product_inquiry":
        return "rag_answer"
    return "greeter"


# ── Node: Greeter ─────────────────────────────────────────────────────────────

_GREETER_SYSTEM = """You are Alex, a friendly and knowledgeable sales assistant for AutoStream — 
an AI-powered video editing SaaS for content creators. 

Keep responses concise, warm, and helpful. If the user seems interested in the product, 
mention you'd be happy to share pricing details or help them get started."""

def greeter_node(state: AgentState) -> AgentState:
    response = llm.invoke([
        SystemMessage(content=_GREETER_SYSTEM),
        *state["messages"],
    ])
    return {**state, "messages": [AIMessage(content=response.content)]}


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
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    )
    context = retrieve_knowledge(last_human)

    prompt = _RAG_SYSTEM.format(context=context)
    response = llm.invoke([
        SystemMessage(content=prompt),
        *state["messages"],
    ])
    return {
        **state,
        "rag_context": context,
        "messages": [AIMessage(content=response.content)],
    }


# ── Node: Lead Capture ────────────────────────────────────────────────────────

def lead_capture_node(state: AgentState) -> AgentState:
    stage = state["lead_stage"]

    # ── Stage: none → start collection ──────────────────────────────────────
    if stage == "none":
        msg = (
            "Great to hear you're interested in AutoStream Pro! 🎉\n\n"
            "I'd love to get you set up. Could I start with your **full name**?"
        )
        return {
            **state,
            "lead_stage": "collecting_name",
            "messages": [AIMessage(content=msg)],
        }

    last_human = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        ""
    ).strip()

    # ── Stage: collecting_name ───────────────────────────────────────────────
    if stage == "collecting_name":
        return {
            **state,
            "lead_name": last_human,
            "lead_stage": "collecting_email",
            "messages": [AIMessage(content=f"Nice to meet you, **{last_human}**! 👋\n\nWhat's your **email address**?")],
        }

    # ── Stage: collecting_email ──────────────────────────────────────────────
    if stage == "collecting_email":
        # Basic email validation
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w{2,}$'
        if not re.match(email_pattern, last_human):
            return {
                **state,
                "messages": [AIMessage(content="Hmm, that doesn't look like a valid email. Could you double-check and re-enter it?")],
            }
        return {
            **state,
            "lead_email": last_human,
            "lead_stage": "collecting_platform",
            "messages": [AIMessage(content="Got it! Last question — which platform do you primarily create content on?\n(e.g., YouTube, Instagram, TikTok, LinkedIn…)")],
        }

    # ── Stage: collecting_platform → fire tool ───────────────────────────────
    if stage == "collecting_platform":
        platform = last_human
        name = state["lead_name"]
        email = state["lead_email"]

        result = mock_lead_capture(name=name, email=email, platform=platform)

        return {
            **state,
            "lead_platform": platform,
            "lead_stage": "done",
            "messages": [AIMessage(content=result)],
        }

    # ── Stage: done — follow-up conversation ─────────────────────────────────
    response = llm.invoke([
        SystemMessage(content="You are Alex from AutoStream. The user has already signed up as a lead. "
                               "Be warm, answer any remaining questions, and let them know the team will be in touch."),
        *state["messages"],
    ])
    return {**state, "messages": [AIMessage(content=response.content)]}


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
