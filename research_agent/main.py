"""
main.py
───────
CLI entry point for the PhD Research Agent.

Usage
─────
  # Minimal
  python main.py --topic "Transformer attention mechanisms"

  # With scope and focus
  python main.py \\
      --topic "CRISPR-Cas9 off-target effects in gene therapy" \\
      --scope "Peer-reviewed biomedical research 2018–2025" \\
      --focus "Emphasise safety, clinical trials, and regulatory landscape"

  # Higher recursion limit for broad topics
  python main.py --topic "Climate change and machine learning" --recursion-limit 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the package root is importable when running from this directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from phd_research_agent import run_research_agent
from phd_research_agent.logger import get_logger

log = get_logger("main")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="PhD Research Agent",
        description="Exhaustive, PhD-level literature review powered by LangGraph + GPT-4.1.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--topic", "-t",
        required=True,
        help="Research topic (required).\nExample: \"Large Language Models in Scientific Discovery\"",
    )
    parser.add_argument(
        "--scope", "-s",
        default=None,
        help=(
            "Optional scope constraint.\n"
            "Example: \"Peer-reviewed work 2018–2025, biomedical focus\""
        ),
    )
    parser.add_argument(
        "--focus", "-f",
        default=None,
        help=(
            "Optional synthesis focus angle.\n"
            "Example: \"Emphasise reproducibility issues and real-world deployment gaps\""
        ),
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=250,
        help="LangGraph recursion limit (default: 250). Increase for broad topics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    log.info("Starting PhD Research Agent via CLI")
    log.info(f"  --topic            : {args.topic}")
    log.info(f"  --scope            : {args.scope or '(not set)'}")
    log.info(f"  --focus            : {args.focus or '(not set)'}")
    log.info(f"  --recursion-limit  : {args.recursion_limit}")

    final_state = run_research_agent(
        topic=args.topic,
        scope=args.scope,
        focus=args.focus,
        recursion_limit=args.recursion_limit,
    )

    print(f"\n✅  Done.  Report → {final_state['output_path']}\n")


if __name__ == "__main__":
    main()
