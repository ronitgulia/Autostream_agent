"""
main.py — AutoStream Agent CLI
Run this file to start an interactive conversation with the AutoStream AI agent.

Usage:
    python main.py

Environment variables (set in .env):
    LLM_PROVIDER    = anthropic | openai | google   (default: anthropic)
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
"""

import sys
import os
import copy
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage, AIMessage
from agent.agent import build_graph, AgentState


BANNER = """
╔══════════════════════════════════════════════════════════╗
║          AutoStream AI Assistant                         ║
║  Powered by LangGraph + RAG | Type 'quit' to exit        ║
╚══════════════════════════════════════════════════════════╝
"""

INITIAL_STATE: AgentState = {
    "messages": [],
    "intent": "greeting",
    "lead_stage": "none",
    "lead_name": "",
    "lead_email": "",
    "lead_platform": "",
    "rag_context": "",
}


def run_cli():
    print(BANNER)
    graph = build_graph()
    state = copy.deepcopy(INITIAL_STATE)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! ")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Agent: Thanks for chatting with AutoStream! Have a great day 🚀")
            break

        # Append user message to state
        state = {**state, "messages": state["messages"] + [HumanMessage(content=user_input)]}

        # Run one step of the graph
        result = graph.invoke(state)
        state = result  # persist updated state (memory across turns)

        # Print latest AI message
        last_ai = next(
            (m.content for m in reversed(state["messages"])
             if isinstance(m, AIMessage)),
            "(no response)"
        )
        print(f"\nAgent: {last_ai}\n")


if __name__ == "__main__":
    run_cli()
