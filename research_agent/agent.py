from __future__ import annotations

import hashlib
import sys
import time
from datetime import date
from typing import Optional 
from dotenv import load_dotenv
from pathlib import Path

from .graph import build_graph
from .logger import get_logger,phase_banner,step,success,warn
from .schemas import ResearchState
from shared.hitl_cli import collect_hitl_response,display_hitl_prompt,build_hitl1_payload
from shared.exceptions import PipelineAbortedError


load_dotenv(Path(__file__).parent / '.env')

log=get_logger(__name__)

_BANNER ="""
                PhD RESEARCH AGENT (Langgraph)
        Exhaustive Systematic   PhD-Level Literature Review

        """

def _make_thread_id(topic:str,scope:Optional[str],focus:Optional[str])-> str:
    fingerprint=f"{topic}| {scope or ''}|{focus or ''}"
    return "phd-" + hashlib.sha1(fingerprint.encode()).hexdigest()[:16]


def _collect_hitl_1_response(interrupt_payload:dict,persisted_state:dict) -> dict:
    """Collect human response for HITL-1 (query plan review )from stdin"""
    response = collect_hitl_response(
        options=["a", "e", "x"],
        extra_prompts={
            "e": 'Enter JSON patch: {"add":[...], "remove":[0,3], "modify":{"2":"new text"}}',
            "x": "Enter optional abort reason:",
        },
    )
    if response.choice == "x":
        raise PipelineAbortedError("HITL-1: Query Plan Review",str(response.payload or ""))
    resume_value={'choice':response.choice}
    if response.payload is not None:
        resume_value['payload'] = response.payload
    return resume_value


def run_research_agent(
        topic:str,
        scope:Optional[str] = None,
        focus : Optional[str]=None,
        recursion_limit:int=250,
        thread_id : Optional[str]=None,
        hitl_mode : str = 'full',
        guardrails_mode:str='strict',
)-> ResearchState:
    """
    Run the PhD Research Agent pipeline.

    Parameters
    ----------
    topic           : Research topic
    scope           : Optional scope constraint
    focus           : Optional synthesis angle
    recursion_limit : LangGraph recursion cap
    thread_id       : For checkpointing — auto-generated if None
    hitl_mode       : "full" | "report-only" | "none"
    guardrails_mode : "strict" | "warn" | "off"
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
        'synthesis_outputs':[],
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