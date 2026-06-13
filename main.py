"""
AutoStream Agent CLI — async token streaming.

Usage:
    python main.py

Environment variables (set in .env):
    LLM_PROVIDER  = anthropic | openai | google | groq  (default: anthropic)
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / GROQ_API_KEY

    # LangSmith observability (optional)
    LANGCHAIN_TRACING_V2 = true
    LANGCHAIN_API_KEY    = <your LangSmith API key>
    LANGCHAIN_PROJECT    = autostream-agent
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage, AIMessage
from agent.agent import build_graph, AgentState


BANNER = """
╔══════════════════════════════════════════════════════════╗
║          AutoStream AI Assistant                         ║
║  Powered by LangGraph + RAG | Type 'quit' to exit        ║
╚══════════════════════════════════════════════════════════╝
"""


async def run_cli() -> None:
    print(BANNER)
    graph = build_graph()
    state = AgentState()

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

        state = state.model_copy(
            update={"messages": state.messages + [HumanMessage(content=user_input)]}
        )

        print("\nAgent: ", end="", flush=True)

        try:
            final_state = None

            async for event in graph.astream_events(state, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    content = chunk.content

                    if isinstance(content, str) and content:
                        print(content, end="", flush=True)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text = part.get("text", "")
                                if text:
                                    print(text, end="", flush=True)

                elif kind == "on_chain_end" and event.get("name") == "LangGraph":
                    final_state = event["data"].get("output")

            print("\n")

            if final_state is not None:
                if isinstance(final_state, dict):
                    state = AgentState(**final_state)
                elif isinstance(final_state, AgentState):
                    state = final_state

        except Exception as e:
            print(f"\n\nAgent: Oops, something went wrong — {e}\n")
            state = state.model_copy(update={"messages": state.messages[:-1]})


if __name__ == "__main__":
    asyncio.run(run_cli())
