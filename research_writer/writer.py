from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv(Path(__file__).parent.parent / ".env")

from .graph import build_writer_graph
from .logger import get_logger,step,success,warn
from .schemas import WriterInput, WriterOutput, WriterState

log=get_logger(__name__)

_BANNER = """
------------------------------------------------------------------
                RESEARCH WRITER  (Mirofish)
    Writer <-> Critic Debate  |  Checkpointing
------------------------------------------------------------------
"""

def _make_thread_id(writer_input : WriterInput) -> str:
    url_sample = "|".join(s.url for s in writer_input.sources[:5])
    fingerprint = f"{writer_input.topic}|{url_sample}"
    return "writer-" + hashlib.sha1(fingerprint.encode()).hexdigest()[:16]


def run_writer(
        writer_input : WriterInput,
        recursion_limit : int=300,
        thread_id : Optional[str]=None,
        
)-> WriterOutput:
    """ Run the Research writer pipeline with correct Langgraph resume loop."""

    print(_BANNER)

    tid = thread_id or _make_thread_id(writer_input)
    log.info(f"Topic           : {writer_input.topic}")
    log.info(f"Sources         : {len(writer_input.sources)}")
    log.info(f"Debate rounds   : {writer_input.max_debate_rounds}")
    log.info(f"Max clusters    : {writer_input.max_clusters}")
    log.info(f"Target words/s  : {writer_input.target_section_words}")
    log.info(f"Lit context     : {'yes' if writer_input.literature_summary else 'no'}")
    log.info(f"Thread ID       : {tid}")
    log.info("=" * 62)

    graph,rec_limit=build_writer_graph(
        recursion_limit=recursion_limit
    )
    initial_state :WriterState={
        "writer_input": writer_input,
        "thread_id": tid,
        "fetched_sources": [],
        "debate_results":[],
        "revision_cycle":0,
    }
    run_config={
        "configurable": {"thread_id":tid},
        "recursion_limit":rec_limit,
    }
    t0 = time.perf_counter()
    current_input=initial_state

    try:
        while True:
            result=graph.invoke(current_input,config=run_config)

            if "__interrupt__" not in result:
                final_state=result
                break 
            # Pipeline suspended
            interrupts =result["__interrupt__"]
            interrupt_payload = interrupts[0].value if interrupts else{}
            checkpoint_name =(
                interrupt_payload.get("checkpoint","")
                if isinstance(interrupt_payload,dict) else ""
            )
            log.info ( f" Pipeline Suspended at :{checkpoint_name or 'HITL checkpoint'}")

            snapshot = graph.get_state(run_config)
            persisted=snapshot.values if snapshot else{}
            next_nodes = list(snapshot.next) if snapshot and snapshot.next else []
            log.info (f" Pending nodes :{next_nodes}")

    except Exception as exc:
        log.error(f"Research Writer failed :{exc}" , exc_info=True)
        log.error(f"To resume : thread_id='{tid}'")
        raise
    elapsed = time.perf_counter()-t0
    writer_output :WriterOutput = final_state['writer_output']

    log.info("=" * 62)
    success(log, "WRITING COMPLETE")
    step(log, f"Time        : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    step(log, f"Sections    : {len(writer_output.sections)}")
    step(log, f"Total words : {writer_output.total_words:,}")
    step(log, f"Rounds      : {writer_output.total_rounds}")
    step(log, f"Saved to    : {writer_output.output_path}")
    step(log, f"Thread ID   : {tid}")
    log.info("=" * 62)

    return writer_output
