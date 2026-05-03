from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from research_writer import run_writer
from research_writer.schemas import WriterInput,SourceItem
from research_writer.logger import get_logger

log= get_logger("writer_main")

################# Helpers

def _load_urls_from_file(path: str)-> List[SourceItem]:
    """
    Read a plain-text file of URLs (one per line).
    Lines starting with # or empty lines are ignored.
    """
    file_path = Path(path)
    if not file_path.exists():
        log.error(f"URL file not found : {path}")
        sys.exit(1)

    sources :List[SourceItem] =[]
    with file_path.open(encoding="utf-8") as fh:
        for line in fh:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            sources.append(SourceItem(
                title=url,
                url=url,
                domain="user_provided",
            ))
    log.info(f"Loaded {len(sources)} URL from {path}")
    return sources


def _generate_sources_from_topic(topic :str , n:int =20) -> List[SourceItem]:
    """
    If no URL file is provided , use Tavily to get an initial source list.
    This makes the writer fully self-contained even without phd_research_agent.
    """
    log.info(f"No Url file provided -- Generating {n} sources via Tavily for : {topic}")
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        tool = TavilySearchResults(max_results=n, search_depth="advanced")
        results=tool.invoke(topic)
        sources = [
            SourceItem(
                title= r.get("title","Unititled"),
                url=r.get("url",""),
                snippet=r.get("content","")[:500],
                domain = "tavily_search",
            )
            for r in results 
            if isinstance(r,dict) and r.get("url")
        ]
        log.info(f"Generated {len(sources)} sources via Tavily")
        return sources
    except Exception as exc:
        log.error(f"Tavily source generation failed : {exc}")
        log.error("Either provide a --urls file or ensure TAVILY_API_KEY is set")
        sys.exit(1)


################ Argument parser

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog = "Research Writer ",
        description =(
            "Standalone Mirofish debate writer.\n"
            "Fetches URLS,cluster them by theme , runs Writer<-> Critic debate loops,\n"
            "and produces a long-form research document . \n\n"
            "If no --urls file is given Tavily is used to find sources automatically."
        ),
        formatter_class =argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--topic", "-t", required=True,
        help='Research topic.\nExample: "Large Language Models in scientific discovery"',
    )
    p.add_argument(
        "--urls", "-u", default=None,
        help="Path to a .txt file with one URL per line.\nIf omitted, Tavily finds sources.",
    )
    p.add_argument(
        "--scope", "-s", default=None,
        help='Scope constraint.\nExample: "Peer-reviewed 2020–2025, biomedical emphasis"',
    )
    p.add_argument(
        "--focus", "-f", default=None,
        help='Writing angle.\nExample: "Emphasise reproducibility and deployment gaps"',
    )
    p.add_argument(
        "--rounds", "-r", type=int, default=3,
        help="Writer↔Critic debate rounds per section (default: 3, range: 1–6)",
    )
    p.add_argument(
        "--clusters", "-c", type=int, default=8,
        help="Max thematic clusters / document sections (default: 8)",
    )
    p.add_argument(
        "--words", "-w", type=int, default=600,
        help="Target words per written section (default: 600)",
    )
    p.add_argument(
        "--recursion-limit", type=int, default=300,
        help="LangGraph recursion limit (default: 300)",
    )
    return p.parse_args()

##################################### Main

def main() -> None:
    args = _parse_args()

    log.info("Starting Research Writer (standalone mode)")
    log.info(f"  --topic    : {args.topic}")
    log.info(f"  --urls     : {args.urls or '(none — Tavily auto-search)'}")
    log.info(f"  --scope    : {args.scope  or '(not set)'}")
    log.info(f"  --focus    : {args.focus  or '(not set)'}")
    log.info(f"  --rounds   : {args.rounds}")
    log.info(f"  --clusters : {args.clusters}")
    log.info(f"  --words    : {args.words}")

    # Build source list
    if args.urls:
        sources = _load_urls_from_file(args.urls)
    else:
        sources = _generate_sources_from_topic(args.topic, n=args.clusters * 8)

    if not sources:
        log.error("No sources available — aborting.")
        sys.exit(1)

    writer_input = WriterInput(
        topic=args.topic,
        scope=args.scope,
        focus=args.focus,
        sources=sources,
        max_debate_rounds=args.rounds,
        max_clusters=args.clusters,
        target_section_words=args.words,
    )

    writer_output = run_writer(
        writer_input=writer_input,
        recursion_limit=args.recursion_limit,
    )

    print(f"\n✅  Writing complete!")
    print(f"   📝 Document  : {writer_output.output_path}")
    print(f"   📋 Transcript: saved alongside document")
    print(f"   📊 Sections  : {len(writer_output.sections)}")
    print(f"   📖 Words     : {writer_output.total_words:,}")
    print(f"   🔄 Rounds    : {writer_output.total_rounds}")
    print()


if __name__ == "__main__":
    main()


