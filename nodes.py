from __future__ import annotations
import json 
import re
from datetime import date
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage,SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from .schemas import (
    EvidenceEvaluation,
    EvidenceItem,
    ResearchState,
    SearchWorkerState,
    SynthesisWorkerState,
    TopicDecomposition
)

#node1-Topic Decomposer

def topic_decomposer(state:ResearchState,llm_fast)->dict:
    '''
    Phase1 -Decompose the topic into subtopics and generate targeted search queries.
    Returns
    --------
    dict with key 'decompostion' (Topic Decomposition)
    '''
    topic=state['topic']
    scope=state.get('scope') or 'Exhaustive global coverage, all the periods'
    focus=state.get('focus') or 'No specific angle -cover all major prespectives'

    decomposition : TopicDecomposition =(
        llm_fast
        .with_structured_output(TopicDecomposition)
        .invoke([
            SystemMessage(),
            HumanMessage()

        ])
    )
    return {'decompostion':decomposition}

# Node 2 - Search Worker (runs in parallel via send)

def search_worker(state:SearchWorkerState)-> dict:
    '''
    Phase 2 -Execute a single search query and return EvidenceItems.
    Each instance runs in parallel(one per SearchQuery from the decompostion).
    Results are fan-in aggregated via operator.add on 'evidence_items'.

    Returns
    dict with key 'evidence_items'(List[EvidenceItem])
    '''
    query = state['query']
    domain = state['domain']

    display_q=query[:65]+ '...' if len(query)> 65 else query
    template='dummy'#get(domain,'{q}')
    augmented=template.replace('{q}',query)

    try:
        search = TavilySearchResults(
            max_results=8,
            search_depths='advanced',
            include_answer=True,
            include_raw_content=False,

        )
        raw_results=search.invoke(augmented)
    except Exception as exc:
        return {'evidence_items':[]}
    items: List[EvidenceItem]=[]
    for r in raw_results:
        if isinstance(r,dict) and r.get('url'):
            snippet_raw=r.get('content')or r.get('snippet') or ""
            items.append(EvidenceItem(
                title=r.get('title','Untitled'),
                url=r.get('url',""),
                snippet=snippet_raw[:500],
                source=str(r.get('score',"")),
                domain=domain,
                query_used=query,
            ))
    return {'evidence_items':items}

