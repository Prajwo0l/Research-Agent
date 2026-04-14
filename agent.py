from __future__ import annotations

import sys
import time
from datetime import date
from typing import Optional 
from dotenv import load_dotenv

from .graph import build_graph
from .logger import get_logger,phase_banner,step,success,warn
from .schemas import ResearchState

load_dotenv()

log=get_logger(__name__)

_BANNER ="""
                PhD RESEARCH AGENT (Langgraph)
        Exhaustive Systematic   PhD-Level Literature Review

        """



def run_research_agent(
        topic:str,
        scope:Optional[str] = None,
        focus : Optional[str]=None,
        recursion_limit:int=250,
)-> ResearchState:
    """
    Run the full PhD Research Agent pipeline.
    Parameters
    ----------
    topic : The research topic(required)
    scope : Optional scope constratint (time period ,geography , sub-domain)
    focus : Optional angle to emphasise in synthesis
    recursion_limit : LangGraph recursion cap(increase for 20-query topics)

    Returns
    -------
    Final ResearchState containing:
        - decomposition : TopicDecompostion
        - evidence_items : List[EvidenceItem]
        - evaluation  : EvidenceEvaluation
        - synthesis_outputs : List[str]
        - final_report : str
        - output_path : str
    """
    print(_BANNER)
    log.info(f"Topic :{topic}")
    log.info(f"Scope :{scope or '(none - global coverage)'}")
    log.info(f"Focus : {focus or '(none-all angles)'}")
    log.info(f'Date: {date.today().strftime('%B %d ,%Y')}')
    log.info("="*62)

    # build graph
    graph,rec_limit=build_graph(recursion_limit=recursion_limit)

    initial_state:ResearchState={
        'topic': topic,
        'scope':scope,
        'focus':focus,
        'evidence_items':[],
        'synthesis_output':[],
    }
    t0 = time.perf_counter()
    try:
        final_state=graph.invoke(
            initial_state,
            config={'recursion_limit':rec_limit},
        )
    except Exception as exc:
        log.error(f"Agent failed with exception : {exc}",exc_info=True)
        raise

    elapsed=time.perf_counter()-t0
    #SUmmary
    log.info("")
    log.info("=" * 62)
    success(log, "RESEARCH COMPLETE")
    step(log, f"Total time     : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    step(log, f"Evidence sources : {len(final_state.get('evidence_items', []))}")
    step(log, f"Report length  : {len(final_state.get('final_report','').split()):,} words")
    step(log, f"Saved to       : {final_state.get('output_path','(unknown)')}")
    log.info("=" * 62)

    return final_state