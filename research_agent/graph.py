from __future__ import annotations
import json 
from functools import partial
from typing import List

from langgraph.graph import END,START,StateGraph
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from .logger import get_logger,step,success
from .nodes import(
    critical_evaluator,
    evidence_aggregator,
    final_assembler,
    save_outputs,
    search_worker,
    synthesis_worker,
    topic_decomposer,
)
from dotenv import load_dotenv
load_dotenv()

from .schemas import ResearchState, SearchWorkerState,SynthesisWorkerState

log=get_logger(__name__)

def llm():
    llm_strong=ChatOpenAI(model='gpt-4o-mini',temperature=0.2)#adjust according to your need use better model for better result
    llm_fast=ChatOpenAI(model='gpt-4o-mini',temperature=0.1)
    return llm_strong,llm_fast

def _route_search_tasks(state:ResearchState)-> List[Send]:
    '''Fan-out : one Send per SearchQuery → search_worker(parallel)'''
    decomposition=state['decomposition']
    log.info(f"Routing {len(decomposition.search_queries)} parallel search tasks...")
    return [
        Send(
            'search_worker',
            SearchWorkerState(
                topic=state['topic'],
                query=sq.query,
                domain=sq.domain,
                rationale=sq.rationale,
                evidence_items=[],
            ),
        )
        for sq in decomposition.search_queries
    ]

def _route_synthesis_tasks(state:ResearchState)-> List[Send]:
    '''Fan-out: three Sends → sysnthesis_worker (parallel).'''
    log.info(" Routing 3 parallel synthesis tasks (Literature / knowledge Map/Bibliography)...")
    topic= state['topic']
    decomposition_json=state['decomposition'].model_dump_json()
    evidence_json=json.dumps([e.model_dump()for e in state.get('evidence_items',[])])
    evaluation_json=state['evaluation'].model_dump_json()
    output_types=['literature_summary','knowledge_map','annotated_bibliography']
    return [
        Send('synthesis_worker',
             SynthesisWorkerState(
                 output_type=ot,
                 topic=topic,
                 decomposition_json=decomposition_json,
                 evidence_json=evidence_json,
                 evaluation_json=evaluation_json,
                 synthesis_outputs=[],
             ),
             )
        for ot in output_types
    ]


def build_graph(recursion_limit : int =250)->tuple:
    """
    Construct and compile the full research agent graph.

    Parameters
    ----------
    recursion_limit : int
        Passed to graph.invoke() config. Increase for topics with 20 queries.

    Returns
    -------
    Compiled LangGraph StateGraph
    """
    llm_strong,llm_fast=llm()

    #bind llms to nodes that need them via functools.partial
    _topic_decomposer = partial(topic_decomposer,llm_fast=llm_fast)
    _critical_evaluator=partial(critical_evaluator,llm_strong=llm_strong)
    _synthesis_worker=partial(synthesis_worker,llm_strong=llm_strong)

    builder = StateGraph(ResearchState)

    builder.add_node("topic_decomposer", _topic_decomposer)
    builder.add_node('search_worker',search_worker)
    builder.add_node('evidence_aggregator',evidence_aggregator)
    builder.add_node("critical_evaluator",  _critical_evaluator)
    builder.add_node('synthesis_worker',_synthesis_worker)
    builder.add_node('final_assembler',final_assembler)
    builder.add_node('save_outputs',save_outputs)


    builder.add_edge(START,'topic_decomposer')
    builder.add_conditional_edges(
        'topic_decomposer',
        _route_search_tasks,
        ['search_worker'],
    )

    builder.add_edge('search_worker','evidence_aggregator')
    builder.add_edge('evidence_aggregator','critical_evaluator')

    builder.add_conditional_edges(
        'critical_evaluator',
        _route_synthesis_tasks,
        ['synthesis_worker'],
    )

    builder.add_edge('synthesis_worker','final_assembler')
    builder.add_edge('final_assembler','save_outputs')
    builder.add_edge('save_outputs',END)

    graph = builder.compile()

    step(log,"Langgraph compiled successfully")
    log.debug(
        "Topology: START → topic_decomposer → [N×search_worker] → "
        "evidence_aggregator → critical_evaluator → [3×synthesis_worker] → "
        "final_assembler → save_outputs → END"
    )

    return graph, recursion_limit
