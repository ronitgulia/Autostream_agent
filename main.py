"""
AutoStream Agent CLI — async token streaming.

Usage:
    python main.py
    python main.py --provider openai --model gpt-4o-mini
    python main.py --debug

Environment variables (set in .env):
    LLM_PROVIDER  = anthropic | openai | google | groq  (default: anthropic)
    LLM_MODEL     = optional model override
    ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / GROQ_API_KEY

    # LangSmith observability (optional)
    LANGCHAIN_TRACING_V2 = true
    LANGCHAIN_API_KEY    = <your LangSmith API key>
    LANGCHAIN_PROJECT    = autostream-agent
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.messages import HumanMessage
from agent.agent import build_graph, AgentState, stream_response

logger = logging.getLogger(__name__)

BANNER = """
╔══════════════════════════════════════════════════════════╗
║          AutoStream AI Assistant                         ║
║  Powered by LangGraph + Hybrid RAG | Type 'quit' to exit ║
╚══════════════════════════════════════════════════════════╝
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AutoStream Agent CLI")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google", "groq"],
        help="Override LLM_PROVIDER from .env",
    )
    parser.add_argument(
        "--model",
        help="Override LLM_MODEL from .env",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG logging output",
    )
    return parser.parse_args()


async def run_cli() -> None:
    args = _parse_args()

    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    active_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    active_model = os.getenv("LLM_MODEL", "provider default")

    print(BANNER)
    print(f"  ⚡ Provider : {active_provider}")
    print(f"  🤖 Model    : {active_model}\n")

    graph = build_graph()
    state = AgentState()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
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

        except KeyboardInterrupt:
            print("\n\n[Response cancelled by user]\n")
        except Exception as e:
            logger.exception("Error during graph execution")
            print(f"\n\nAgent: Oops, something went wrong — {e}\n")
            # Roll back the last human message so the state stays consistent.
            state = state.model_copy(update={"messages": state.messages[:-1]})


if __name__ == "__main__":
    asyncio.run(run_cli())
