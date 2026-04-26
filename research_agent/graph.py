"""
graph.py
────────
PhD Research Agent LangGraph — v3 with HITL + Guardrails

Topology
────────
  START
    └─► topic_decomposer
          └─► guardrail_1_query_validator   (G1)
                └─► hitl_1_query_review      (HITL-1 — calls interrupt() internally)
                      └─► [N × search_worker]
                            └─► evidence_aggregator
                                  └─► guardrail_2_source_quality  (G2)
                                        └─► critical_evaluator
                                              └─► [3 × synthesis_worker]
                                                    └─► final_assembler
                                                          └─► save_outputs
                                                                └─► END

How HITL works here
────────────────────
  hitl_1_query_review calls langgraph.types.interrupt() internally.
  This raises GraphInterrupt inside graph.invoke(), which agent.py catches.
  agent.py then calls graph.invoke(Command(resume=human_response), same config)
  to resume from the checkpoint.

  Do NOT use interrupt_before= for nodes that call interrupt() themselves —
  that would cause a double-interrupt. We compile WITHOUT interrupt_before.
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from .logger import get_logger, step
from .nodes import (
    topic_decomposer,
    guardrail_1_query_validator,
    hitl_1_query_review,
    search_worker,
    evidence_aggregator,
    guardrail_2_source_quality,
    critical_evaluator,
    synthesis_worker,
    final_assembler,
    save_outputs,
)
from .schemas import ResearchState, SearchWorkerState, SynthesisWorkerState

log = get_logger(__name__)


def _make_llms():
    llm_strong = ChatOpenAI(model="gpt-4o-mini",      temperature=0.2)
    llm_fast   = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    return llm_strong, llm_fast


def _route_search_tasks(state: ResearchState) -> List[Send]:
    decomposition = state["decomposition"]
    log.info(f"  Routing {len(decomposition.search_queries)} parallel search tasks...")
    return [
        Send("search_worker", SearchWorkerState(
            topic=state["topic"],
            query=sq.query, domain=sq.domain, rationale=sq.rationale,
            evidence_items=[],
        ))
        for sq in decomposition.search_queries
    ]


def _route_synthesis_tasks(state: ResearchState) -> List[Send]:
    log.info("  Routing 3 parallel synthesis tasks...")
    topic              = state["topic"]
    decomposition_json = state["decomposition"].model_dump_json()
    evidence_json      = json.dumps([e.model_dump() for e in state.get("evidence_items", [])])
    evaluation_json    = state["evaluation"].model_dump_json()
    return [
        Send("synthesis_worker", SynthesisWorkerState(
            output_type=ot, topic=topic,
            decomposition_json=decomposition_json,
            evidence_json=evidence_json,
            evaluation_json=evaluation_json,
            synthesis_outputs=[],
        ))
        for ot in ["literature_summary", "knowledge_map", "annotated_bibliography"]
    ]


def build_graph(
    recursion_limit: int = 250,
    hitl_mode:       str = "full",
    guardrails_mode: str = "strict",
):
    """
    Build and compile the PhD Research Agent graph.

    IMPORTANT: We do NOT use interrupt_before= here.
    hitl_1_query_review calls interrupt() internally when hitl_mode != "none".
    The GraphInterrupt is caught by the resume loop in agent.py.
    """
    llm_strong, llm_fast = _make_llms()

    _topic_decomposer   = partial(topic_decomposer,   llm_fast=llm_fast)
    _critical_evaluator = partial(critical_evaluator, llm_strong=llm_strong)
    _synthesis_worker   = partial(synthesis_worker,   llm_strong=llm_strong)

    builder = StateGraph(ResearchState)

    builder.add_node("topic_decomposer",            _topic_decomposer)
    builder.add_node("guardrail_1_query_validator", guardrail_1_query_validator)
    builder.add_node("hitl_1_query_review",         hitl_1_query_review)
    builder.add_node("search_worker",               search_worker)
    builder.add_node("evidence_aggregator",         evidence_aggregator)
    builder.add_node("guardrail_2_source_quality",  guardrail_2_source_quality)
    builder.add_node("critical_evaluator",          _critical_evaluator)
    builder.add_node("synthesis_worker",            _synthesis_worker)
    builder.add_node("final_assembler",             final_assembler)
    builder.add_node("save_outputs",                save_outputs)

    builder.add_edge(START, "topic_decomposer")
    builder.add_edge("topic_decomposer",            "guardrail_1_query_validator")
    builder.add_edge("guardrail_1_query_validator", "hitl_1_query_review")

    builder.add_conditional_edges(
        "hitl_1_query_review",
        _route_search_tasks,
        ["search_worker"],
    )

    builder.add_edge("search_worker",              "evidence_aggregator")
    builder.add_edge("evidence_aggregator",        "guardrail_2_source_quality")
    builder.add_edge("guardrail_2_source_quality", "critical_evaluator")

    builder.add_conditional_edges(
        "critical_evaluator",
        _route_synthesis_tasks,
        ["synthesis_worker"],
    )

    builder.add_edge("synthesis_worker", "final_assembler")
    builder.add_edge("final_assembler",  "save_outputs")
    builder.add_edge("save_outputs",     END)

    # ── Compile with MemorySaver — NO interrupt_before ──────────────────
    # The HITL node calls interrupt() directly. interrupt_before would
    # suspend BEFORE the node runs (never showing the query table to human).
    # We want the node to run (build + display the table), THEN suspend.
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    step(log, f"PhD Agent compiled  (HITL={hitl_mode}  Guardrails={guardrails_mode}  ✓)")
    return graph, recursion_limit
