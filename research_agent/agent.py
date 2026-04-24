from __future__ import annotations

import hashlib
import sys
import time
from datetime import date
from typing import Optional 
from dotenv import load_dotenv
from pathlib import Path
from langgraph.types import Command

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
    topic:           str,
    scope:           Optional[str] = None,
    focus:           Optional[str] = None,
    recursion_limit: int = 250,
    thread_id:       Optional[str] = None,
    hitl_mode:       str = "full",
    guardrails_mode: str = "strict",
) -> ResearchState:
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

    tid = thread_id or _make_thread_id(topic, scope, focus)

    log.info(f"Topic           : {topic}")
    log.info(f"Scope           : {scope or '(none)'}")
    log.info(f"Focus           : {focus or '(none)'}")
    log.info(f"Date            : {date.today().strftime('%B %d, %Y')}")
    log.info(f"Thread ID       : {tid}")
    log.info(f"HITL mode       : {hitl_mode}")
    log.info(f"Guardrails mode : {guardrails_mode}")
    log.info("=" * 62)

    graph, rec_limit = build_graph(
        recursion_limit=recursion_limit,
        hitl_mode=hitl_mode,
        guardrails_mode=guardrails_mode,
    )

    run_config = {
        "configurable":   {"thread_id": tid},
        "recursion_limit": rec_limit,
    }

    initial_state: ResearchState = {
        "topic":               topic,
        "scope":               scope,
        "focus":               focus,
        "hitl_mode":           hitl_mode,
        "guardrails_mode":     guardrails_mode,
        "thread_id":           tid,
        "evidence_items":      [],
        "synthesis_outputs":   [],
        "guardrail_1_results": [],
        "guardrail_2_results": [],
    }

    t0 = time.perf_counter()

    # ── HITL Resume Loop ─────────────────────────────────────────────────
    # In LangGraph 0.6.4, interrupt() does NOT raise an exception.
    # graph.invoke() returns a dict with "__interrupt__" key when suspended.
    # We check for that key, collect input, then resume with Command(resume=...).
    # ─────────────────────────────────────────────────────────────────────
    current_input = initial_state

    try:
        while True:
            result = graph.invoke(current_input, config=run_config)

            # Check if pipeline suspended at a HITL checkpoint
            if "__interrupt__" not in result:
                # Clean finish — no more interrupts
                final_state = result
                break

            # ── Pipeline is suspended ────────────────────────────────────
            interrupts = result["__interrupt__"]
            interrupt_payload = interrupts[0].value if interrupts else {}
            checkpoint_name   = (
                interrupt_payload.get("checkpoint", "HITL checkpoint")
                if isinstance(interrupt_payload, dict) else "HITL checkpoint"
            )
            log.info(f"  ⏸ Pipeline suspended at: {checkpoint_name}")

            # Get persisted state for context
            snapshot = graph.get_state(run_config)
            persisted = snapshot.values if snapshot else {}
            next_nodes = list(snapshot.next) if snapshot and snapshot.next else []
            log.info(f"  Pending nodes: {next_nodes}")

            # ── Route to correct HITL handler ────────────────────────────
            if "hitl_1_query_review" in next_nodes or "HITL-1" in checkpoint_name:
                resume_value = _collect_hitl_1_response(interrupt_payload, persisted)
            else:
                warn(log, f"Unknown HITL '{checkpoint_name}' — auto-approving")
                resume_value = {"choice": "a"}

            # Resume from checkpoint with human response
            current_input = Command(resume=resume_value)

    except PipelineAbortedError:
        raise
    except Exception as exc:
        log.error(f"Agent failed: {exc}", exc_info=True)
        log.error(f"To resume: use thread_id='{tid}'")
        raise

    elapsed = time.perf_counter() - t0

    log.info("=" * 62)
    success(log, "RESEARCH COMPLETE")
    step(log, f"Time            : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    step(log, f"Evidence sources: {len(final_state.get('evidence_items', []))}")
    step(log, f"Report length   : {len(final_state.get('final_report','').split()):,} words")
    step(log, f"Thread ID       : {tid}")
    log.info("=" * 62)

    return final_state
