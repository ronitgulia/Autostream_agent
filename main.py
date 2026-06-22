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

from langchain_core.messages import HumanMessage
from agent.agent import build_graph, AgentState, stream_response


BANNER = """
╔══════════════════════════════════════════════════════════╗
║          AutoStream AI Assistant                         ║
║  Powered by LangGraph + Hybrid RAG | Type 'quit' to exit ║
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
            async for text, final in stream_response(graph, state):
                if final is not None:
                    state = final
                elif text:
                    print(text, end="", flush=True)

            print("\n")

        except Exception as e:
            print(f"\n\nAgent: Oops, something went wrong — {e}\n")
            # Roll back the last human message so the state stays consistent.
            state = state.model_copy(update={"messages": state.messages[:-1]})


if __name__ == "__main__":
    asyncio.run(run_cli())
