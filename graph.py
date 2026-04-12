from __future__ import annotations
import json 
from functools import partila
from typing import List

from langgraph.graph import END,START,STATEGraph
from langgraph.types import Send
from langchain_openai import ChatOpenAI


from .schemas import ResearchState, SearchWorkerState,SynthesisWorkerState


def llm():
    llm_strong=ChatOpenAI(model='gpt-4o-mini',temperature=0.2)#adjust according to your need use better model for better result
    llm_fast=ChatOpenAI(model='gpt-4o-mini',tempreature=0.1)
    return llm_strong,llm_fast

def _route_search_tasks(state:ResearchState)-> List[Send]:
    '''Fan-out : one Send per SearchQuery → search_worker(parallel)'''
    decomposition=state['decomposition']
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
                 decompostion_json=decomposition_json,
                 evidence_json=evidence_json,
                 evaluation_json=evaluation_json,
                 synthesis_outputs=[],
             ),
             )
        for ot in output_types
    ]