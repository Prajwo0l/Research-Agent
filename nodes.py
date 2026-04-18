from __future__ import annotations
import json 
import re
from datetime import date
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage,SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from .logger import get_logger,phase_banner,step,substep,warn,success,section_title

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

log=get_logger(__name__)
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

    phase_banner(log,1,'Topic Decomposition')
    log.info(f"Topic : {topic}")
    log.info(f'Scope : {scope}')
    log.info(f'Focus : {focus}')


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
    return {'decomposition':decomposition}

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
    log.info(f' search [{domain:<24}] {display_q}')

    template=_DOMAIN_PREFIXES.get(domain,"{q}")
    augmented=template.replace('{q}',query)

    try:
        search = TavilySearchResults(
            max_results=8,
            search_depth='advanced',
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
    substep(log,f"Found {len(items)} results")
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
        if item.url and item.url not in seen:
            seen.add(item.url)
            unique.append(item)

    phase_banner(log,2,"Evidence Aggregation Complte")
    step(log,f"Total raw results: {len(items)}")
    step(log,f"Unique sources : {len(unique)}")

    domain_counts: dict[str,int]={}    
    for item in unique:
        domain_counts[item.domain]=domain_counts.get(item.domain,0)+1
    for domain,count in sorted(domain_counts.items()):
        substep(log,f'{domain:<30} {count} sources')
    # return deduplicated list back into state
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

    phase_banner(log,3,'Critical Evaluation')
    log.info(f"Evaluating {len(items)} sources ...")

    # build a concise evidence diegst
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
    step(log,f"Consenseus areas : {len(evaluation.consensus_areas)}")
    step(log,f"Contested claims : {len(evaluation.contested_claims)} ")
    step(log,f"Key debates : {len(evaluation.key_debates)}")
    step(log,f"Replication concerns : {len(evaluation.replication_concerns)}")

    log.debug("Evaluation: \n%s",evaluation.model_dump_json(indent=2))
    return {'evaluation':evaluation}


# Node 5 - Synthesis Worker (runs parllael send*3)

def synthesis_worker(state:SynthesisWorkerState,llm_strong)-> dict:
    """
    Phase 4 - Write one of the three Phd research outputs.

    Three instances run in parallel :
     1. literature_summary
     2. knowledge_map
     3. annotated_bibliography

    Returns
    -------
    dict with key 'synthesis_outputs' (List[str]) - single-element list
    """
    output_type = state['output_type']
    topic = state['topic']
    label = _OUTPUT_LABELS[output_type]

    section_title(log, f"Writing {label}....")

    # desearialize context 
    evidence_list = json.loads(state['evidence_json'])
    evaluation = json.loads(state['evaluation_json'])
    decomp=json.loads(state['decomposition_json'])

    #build evidence digest 
    evidence_digest = "\n\n".join([
        f"[{i+1}] {e.get('domain','?').upper()} | {e.get('title','Untitled')}\n"
        f"URL: {e.get('url','')}\n"
        f"{(e.get('snippet') or '')[:300]}"
        for i, e in enumerate(evidence_list[:80])
    ])


    #build user message 
    def _bullet_list(items: list) -> str:
        return "\n".join(f"  • {x}" for x in items) if items else "  (none identified)"

    user_msg = SYNTHESIS_USER_TEMPLATE.format(
        topic               = topic,
        scope_statement     = decomp.get("scope_statement", ""),
        subtopics           = ", ".join(decomp.get("core_subtopics", [])),
        disciplines         = ", ".join(decomp.get("disciplines", [])),
        time_periods        = ", ".join(decomp.get("time_periods", [])),
        consensus_areas     = _bullet_list(evaluation.get("consensus_areas", [])),
        contested_claims    = _bullet_list(evaluation.get("contested_claims", [])),
        key_debates         = _bullet_list(evaluation.get("key_debates", [])),
        replication_concerns= _bullet_list(evaluation.get("replication_concerns", [])),
        methodological_landscape = evaluation.get("methodological_landscape", ""),
        quality_notes       = evaluation.get("quality_notes", ""),
        n_sources           = len(evidence_list),
        evidence_digest     = evidence_digest,
        output_label        = label,
    )

    system_prompt = _SYNTHESIS_SYSTEM_PROMPTS[output_type]

    result = llm_strong.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_msg),
    ])

    output_text = result.content
    word_count  = len(output_text.split())

    step(log, f"{label} → {word_count:,} words written")
    log.debug("First 500 chars of %s:\n%s", output_type, output_text[:500])

    return {"synthesis_outputs": [output_text]}


#node 6 -final assembler

def final_assembler(state: ResearchState)-> dict:
    """
    Merge all three synthesis outputs into a single cohesive Markdwon report.

    Returns
    -------
    dict with key 'final_report' (str)
    """
    topic = state['topic']
    outputs = state.get('synthesis_outputs',[])
    today= date.today().strftime('%B %d, %Y')
    n_sources=len(state.get('evidence_items',[]))

    phase_banner(log,3,'Final Assembly')
    log.info(f"Merging { len(outputs)} synthesis outputs ....")

    #identify which output is which 
    lit_summary =""
    knowledge_map = "" 
    bibliography = "" 

    for output in outputs:
        if "OUTPUT 1" in output or "Literature Summary" in output:
            lit_summary = output
        elif "OUTPUT 2" in output or "Knowledge Map" in output:
            knowledge_map = output
        elif "OUTPUT 3" in output or "Annotated Bibliography" in output:
            bibliography=output
    if not lit_summary:
        warn(log,"Literature Summary output not identified - using raw first output")
        lit_summary=outputs[0] if outputs else "(missing)"
    if not knowledge_map:
        warn(log,"Knowledge Map output not identified - using raw second output")
        knowledge_map = outputs[1] if len(outputs) > 1 else "(missing)"
    if not bibliography:
        warn(log,"Annotated Bibliography output not identified- using raw third output")
        bibliography = outputs[2] if len(outputs) > 2 else'(missing)'

    header = f"""\
# 🎓 PhD Research Report

## Topic: {topic}

| Field          | Value                                        |
|----------------|----------------------------------------------|
| **Generated**  | {today}                                      |
| **Agent**      | PhD Research Agent (LangGraph + GPT-4.1)     |
| **Sources**    | {n_sources} unique sources                   |
| **Subtopics**  | {len(state["decomposition"].core_subtopics)} |

---

> ⚠️ *This report was produced by an automated PhD-level research agent.
> All citations marked `[VERIFY]` should be independently confirmed
> before use in academic work.*

---

## Table of Contents
1. [Literature Summary](#output-1--literature-summary)
2. [Knowledge Map](#output-2--knowledge-map)
3. [Annotated Bibliography](#output-3--annotated-bibliography)

---
"""
    final_report="\n\n".join([
        header,
        lit_summary,
        "\n---\n",
        knowledge_map,
        "\n---\n",
        bibliography,
    ])
    total_words = len(final_report.split())
    step(log,f"Final Report Assembled : {total_words:,} words")

    return {'final_report':final_report}


#Node 7 -saving output

def save_outputs(state:ResearchState)-> dict:
    """
    Persist the final report as a Markdown file in the outputs/directory.

    Returns
    -------
    dict with key 'output_path'(str)
    """
    topic= state['topic']
    report=state['final_report']

    #Safe filename
    safe_topic = re.sub(r"[^\w\s-]", "", topic.lower())
    safe_topic = re.sub(r"\s+", "_", safe_topic)[:60]
    today_str  = date.today().strftime("%Y%m%d")
    filename   = f"{today_str}_{safe_topic}_phd_research_report.md"

    #save next to this package in an outpiut
    output_dir = Path(__file__).parent/"outputs"
    output_dir.mkdir(parents=True,exist_ok=True)
    output_path=output_dir /filename

    output_path.write_text(report,encoding='utf-8')
    size_kb = output_path.stat().st_size/1024
    success(log,f"Report saved -> {output_path.resolve()}")
    step(log,f"File size : {size_kb:.1f} KB")

    return {'output_path':str(output_path.resolve())}

