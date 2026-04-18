from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parent.parent))
from research_agent import run_research_agent
from .logger import get_logger

log = get_logger('main')

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog = "PhD Research Agent" ,
        description = "Exhaustive , PhD-level literature review powered by Langgraph + GPT-4.1",
        formatter_class = argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--topic","-t",
        required = True,
        help="Research topic (required).\nExample: \"Large Language Models in Scientific Discovery\"",
    )
    parser.add_argument(
        "--scope","-s",
        default=None,
        help = (
            "Optional scope constraint.\n"
            "Example: \"Peer-reviewed work 2018-2025,biomedical focus\""
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
        help="LangGraph recursion limit (default:250).Increase for broad topics.",
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



