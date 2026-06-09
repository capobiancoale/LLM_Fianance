# =============================================================================
# COMMAND-LINE INTERFACE — interactive chat with the multi-agent system
# =============================================================================
# Usage:
#   python -m src.app_cli                 # interactive chat (verbose routing)
#   python -m src.app_cli "your question" # single-shot query, print the answer
# =============================================================================

import sys

BANNER = """
============================================================
  MULTI-AGENT COMPANY & FINANCE RESEARCH ASSISTANT
============================================================
  A supervisor routes your question to the right specialist:
    - document_qa  : background facts & definitions (RAG)
    - data_analyst : numbers from the bundled dataset
    - web_research : current / real-time information
  Type 'exit' or 'quit' to leave.
------------------------------------------------------------
"""


def _single_shot(question: str) -> None:
    from .agents import ask

    print(ask(question, verbose=True))


def _interactive() -> None:
    # Import inside the function so `--help`-style misuse is fast and so the
    # vector store / LLM are only initialised when actually chatting.
    from .agents import ask

    print(BANNER)
    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        print("\n[routing]")
        try:
            answer = ask(user_input, verbose=True)
            print(f"\nassistant> {answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"\n[error] {e}\n")


def main() -> None:
    if len(sys.argv) > 1:
        _single_shot(" ".join(sys.argv[1:]))
    else:
        _interactive()


if __name__ == "__main__":
    main()
