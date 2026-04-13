from __future__ import annotations
import json 
import re
from datetime import date
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage,SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from .prompts import (
    TOPIC_DECOMPOSER_SYSTEM,
    TOPIC_DECOMPOSER_USER,
    CRITICAL_EVALUATOR_SYSTEM,
    CRITICAL_EVALUATOR_USER,
    LITERATURE_SUMMARY_SYSTEM,
    KNOWLEDGE_MAP_SYSTEM,
    ANNOTATED_BIBLIOGRAPHY_SYSTEM,
    SYNTHESIS_USER_TEMPLATE,
)
from .schemas import (
    EvidenceEvaluation,
    EvidenceItem,
    ResearchState,
    SearchWorkerState,
    SynthesisWorkerState,
    TopicDecomposition
)
#Domain aware query augmentation
_DOMAIN_PREFIXES: dict[str, str] = {
    "academic_papers": (
        "site:scholar.google.com OR site:pubmed.ncbi.nlm.nih.gov "
        "OR site:semanticscholar.org {q}"
    ),
    "preprints": "site:arxiv.org OR site:biorxiv.org OR site:ssrn.com {q}",
    "patents_applied": "patent OR USPTO OR EPO {q} application",
    "grey_literature": "technical report OR white paper OR dissertation OR policy {q}",
    "news_commentary": "research news OR expert analysis {q} 2023 OR 2024 OR 2025",
    "conference_proceedings": "conference proceedings OR symposium OR workshop paper {q}",
}

_OUTPUT_LABELS={
    'literature_summary': 'OUTPUT 1: Literature Summary',
    'knowledge_map': 'OUTPUT 2 : Knowledge Map',
    'annotated_bibliography':'OUTPUT 3 : Annotated Bibliography',
}

_SYNTHESIS_SYSTEM_PROMPTS={
    'literature_summary': LITERATURE_SUMMARY_SYSTEM,
    'knowledge_map': KNOWLEDGE_MAP_SYSTEM,
    'annotated_bibliography': ANNOTATED_BIBLIOGRAPHY_SYSTEM,
}


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
            SystemMessage(content=TOPIC_DECOMPOSER_SYSTEM),
            HumanMessage(content=TOPIC_DECOMPOSER_USER.format(
                topic=topic,
                scope=scope,
                focus=focus
            )),

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
    template=_DOMAIN_PREFIXES.get(domain,"{q}")
    augmented=template.replace('{q}',query)

    try:
        search = TavilySearchResults(
            max_results=8,
            search_depths='advanced',
            include_answer=True,
            include_raw_content=False,
            include_images=False,

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

# Node 3 -Evidence Aggregator

def evidence_aggregator(state:ResearchState)-> dict:
    """
    Deduplicate and summarise the collected evidence corpus.
    NOTE: this node does not modify state - it only logs.
    The fan-in via operator.add has already merged all items.

    Returns
    -------
    Empty dict(state already populated by operator.add)    
    """
    items=state.get('evidence_items',[])

    #Deduplicate by URL
    seen:set[str]=set()
    unique:List[EvidenceItem]=[]
    for item in items:
        if items.url and item.url not in seen:
            seen.add(item.url)
            unique.append(item)

        domain_counts: dict[str,int]={}    
        for item in unique:
            domain_counts[item.domain]=domain_counts.get(item.domain,0)+1
        return {'evidence_items':unique}
    
# Node 4 -critical Evaluator
def critical_evaluator(state :ResearchState,llm_strong)->dict:
    '''Phase 3 -Critically evaluate the evidence corpus.
    
    Identifies consensus areas, contested claims,key debates,
    replication concerns, and the methodological landscape.
    Returns
    -------
    dict with key 'evaluation'(EvidenceEvaluation)
    '''
    topic= state['topic']
    items=state.get('evidence_items',[])
    decompostion=state['decomposition']

    capped = items[:60]
    evidence_digest="\n\n".join([
        f"[{i+1}] DOMAIN: {item.domain}\n"
        f"TITLE:{item.title}\n"
        f"URL:{item.url}\n"
        f"SNIPPET:{(item.snippet or 'N/A')[:300]}"
        for i , item in enumerate(capped)
    ])
    subtopics_str="\n".join(f" {s}"for s in decompostion.core_subtopics)
    evaluation: EvidenceEvaluation=(
        llm_strong
        .with_structured_output(EvidenceEvaluation)
        .invoke([
            SystemMessage(content=CRITICAL_EVALUATOR_SYSTEM),
            HumanMessage(content=CRITICAL_EVALUATOR_USER.format(
                topic=topic,
                subtopics=subtopics_str,
                n_sources=len(items),
                evidence_digest=evidence_digest,
            )),
        ])
    )
    return {'evulation':evaluation}





