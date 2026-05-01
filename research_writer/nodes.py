from __future__ import annotations
import json 
import re
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import List

import httpx
from langchain_core.messages import HumanMessage,SystemMessage
from functools import partial

sys.path.insert(0,str(Path(__file__).parent.parent))

from .logger import (
    get_logger,phase_banner,debate_banner,
    step,substep ,warn,success,writer_says,critic_says,
)
from .prompts import(
    CLUSTER_PLANNER_SYSTEM ,CLUSTER_PLANNER_USER,
    WRITER_INITIAL_SYSTEM,WRITER_INITIAL_USER,
    CRITIC_SYSTEM, CRITIC_USER,
    WRITER_REVISION_SYSTEM,WRITER_REVISION_USER,
    POLISH_SYSTEM,POLISH_USER,
    ASSEMBLER_SYSTEM,ASSEMBLER_USER,
    )
from .schemas import(
    Cluster,ClusterPlan,
    DebateResult,DebateTurn,
    FetchedSource,SourceFetchWorkerState,DebateWorkerState,
    WriterInput,WriterOutput,WriterState,
)

log = get_logger(__name__)

_FETCH_TIMEOUT=12
_FETCH_MAX_CHARS=8_000
_FETCH_HEADERS={
    'User-Agent':"Mozilla/5.0(compatible; ResearchWriterBot/1.0)"
}

#####Internal helper

def _extract_text_from_html(html:str)->str:
    html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", " ", html,
            flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+"," ",text).trip()
    return text[:_FETCH_MAX_CHARS]

def _safe_fetch(url:str)-> tuple[str,str]:
    try:
        resp = httpx.get(url,timeout=_FETCH_TIMEOUT,headers=_FETCH_HEADERS,
                         follow_redirects=True)
        resp.raise_for_status()
        text=_extract_text_from_html(resp.text)
        return (text,"success")if len(text) > 100 else(text,"fallback")
    except Exception as exc:
        log.debug(f"Fetch error[{url}]: {exc}")
        return "","failed"
    

def _build_literature_context(writer_input :WriterInput) -> str:
    parts : list[str]=[]
    if writer_input.literature_summary:
        parts.append("== LITERATURE SUMMARY ===")
        parts.append(writer_input.literature_summary[:3000])
    if writer_input.knowledge_map:
        parts.append("==KNOWLEDGE MAP ==")
        parts.append(writer_input.knowledge_map[:2000])
    return "\n\n".join(parts) if parts else "(no prior research context provided)"


def  _format_source_content(sources:List[FetchedSource], max_chars_each:int = 1_500)-> str:
    separator="-" * 60 + "\n"
    blocks :list[str]=[]
    for i , s in enumerate(sources,1):
        content=(s.content or s.fetch_status)[:max_chars_each]
        blocks.append(
            f"[SOURCE {i}]{s.title}\n"
            f"URL : {s.url}\n"
            f"Domain : {s.domain} | Status :{s.fetch_status}\n"
            f"Content: \n {content}"
        ) 
    return "\n"+ separator.join(blocks)


#########################Stage 1 - Fetch Worker (parallel,one per url)#####################################
def fetch_worker(state:SourceFetchWorkerState)-> dict:
    """
    Fetch full page text for one URL.
    Priority: pre_fetched_content -> httpx -> snippet fallback.
    """
    source = state["source"]
    pre_fetched_content = state.get("pre_fetched_content",{})

    # Priority 1 : Agent A alreadt has it
    if source.url in pre_fetched_content:
        cached=pre_fetched_content[source.url]
        if cached and len(cached) > 50:
            fetched = FetchedSource(
                title = source.title, url = source.url , domain= source.domain,
                query_used = source.query_used,content = cached[:8_000],
                fetch_status = "pre_fetched", word_count=len(cached.split()),
            )
            substep(log, f" PRE_FETCHED {fetched.word_count: >5} words")
            return {"fetched_sources":[fetched]}
        
    # Priority 2 : live HTTP 
    log.info(f" WEB Fetching[{source.domain}] {source.url[:75]}")
    content, status = _safe_fetch(source.url)

    # Priority 3 :snippet
    if not content or status =="failed":
        content= source.snippet or ""
        status= "fallback" if content else "failed"

    fetched = FetchedSource(
        title = source.title, url = source.url ,domain = source.domain,
        query_used = source.query_used,content=content,
        fetch_status=status,word_count=len(content.split()),
    )
    icon = {"success": "✓", "pre_fetched": "⚡", "fallback": "~", "failed": "✗"}.get(status, "?")
    substep(log,f"{icon}{status.upper():<12} {fetched.word_count:>5}words {source.title[:55]}")
    return {"fetched_sources":[fetched]}


################################## Stage 1 b - Content Aggregator

def content_aggregator(state :WriterState)-> dict:
    fetched = state.get("fetched_sources", [])
    counts : dict[str,int] = {"success":0 , "pre_fetched":0,"fallback":0,"failed":0}
    for f in fetched:
        counts[f.fetch_status] = counts.get(f.fetch_status,0)+1

    total_words = sum(f.word_count for f in fetched)
    phase_banner(log,1,"Content Fetching Complete")
    step(log, f"Total fetched : {len(fetched)}")
    substep(log,f"pre_fetched {counts['pre_fetched']}   (Agent A cache -no HTTP)")
    substep(log,f"success {counts['success']}")
    substep(log,f"fallback {counts['fallback']} (snippet used)")
    substep(log, f"failed {counts['failed']} (dropped)")
    step(log, f"Total words : {total_words:,}")
    step(log, f"Usable sources : {len([f for f in fetched if f.content])}")
    return {}

############################## Cluster Planner

def cluster_planner(state:WriterState,llm_strong)->dict:
    writer_input=state['writer_input']
    fetched = state.get("fetched_sources",[])
    usable = [f for f in fetched if f.content]

    phase_banner(log,2,"Thematic Clustering & Document Planning")
    log.info(f"Clustering{len(usable)} usable sources into {writer_input.max_clusters} themes....")

    source_digest = "\n\n".join([
        f"[{i+1}] TITLE : {s.title}\n"
        f" URL :{s.url}\n"
        f" DOMAIN : {s.domain}\n"
        f" TEXT :{s.content[:400]}"
        for i, s in enumerate(usable[:60]) 
    ])
    lit_context = _build_literature_context(writer_input)
    cluster_plan : ClusterPlan =(
        llm_strong.with_structured_output(ClusterPlan).invoke([
            SystemMessage(content=CLUSTER_PLANNER_SYSTEM.format(
                max_clusters=writer_input.max_clusters,
            )),
            HumanMessage(content=CLUSTER_PLANNER_USER.format(
                topic=writer_input.topic,
                scope=writer_input.scope or "global coverage",
                focus=writer_input.focus or "all major angles",
                max_clusters=writer_input.max_clusters,
                literature_context=lit_context[:3000],
                n_sources=len(usable),
                source_digest=source_digest,

            )),
        ])
    )
    step(log,f"Document title : {cluster_plan.document_title}")
    step(log, f"Cluster planned :{len(cluster_plan.clusters)}")
    for cl in cluster_plan.clusters:
        substep(log,f"[{cl.cluster_id:02d}]{cl.section_title:<45}{len(cl.source_urls)}src")

    return {"cluster_plan":cluster_plan}



####################### Stage 3 Debate Worker (mirofish , parallel per cluster)

def debate_worker(state:DebateWorkerState,llm_writer,llm_critic)-> dict:
    cluster= state['cluster']
    sources = state['fetched_sources']
    topic = state["topic"]
    lit_context = state['literature_context']
    max_rounds=state['max_rounds']
    target_words = state['target_words']

    extra_instruction = state.get("extra_instruction","")
    phase_banner(log,3,f"Debate --{cluster.section_title[:45]}")
    log.info(f"Cluster [{cluster.cluster_id:02d}] | {len(sources)} sources | {max_rounds} rounds")

    if extra_instruction:
        log.info(f"Extra Instruction from human reviewer : {extra_instruction[:80]}")
    source_content = _format_source_content(sources)
    turns : List[DebateTurn]=[]

    #Round 0 : Initial Draft 
    debate_banner (log,0,"Initial Draft")
    system_content = WRITER_INITIAL_SYSTEM.format(
        theme = cluster.theme,section_title=cluster.section_title,
        writing_goal = cluster.writing_goal, target_words=target_words,
        topic = topic,n_sources=len(sources),
    )
    if extra_instruction:
        system_content += f"\n\nADDITIONAL INSTRUCTION FROM HUMAN REVIEWER :\n{extra_instruction}"

    init_result=llm_writer.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=WRITER_INITIAL_USER.format(
            theme=cluster.theme,section_title=cluster.section_title,
            writing_goal=cluster.writing_goal,target_words=target_words,
            literature_context=lit_context[:1_500],
            n_sources=len(sources), source_content=source_content,
        )),
    ])
    current_draft=init_result.content
    turns.append(DebateTurn(role="writer",content=current_draft,round=0))
    writer_says(log,current_draft)

    for round_num in range(1,max_rounds+1):
        debate_banner(log,round_num,cluster.theme)

        crit_result=llm_critic.invoke([
            SystemMessage(content=CRITIC_SYSTEM.format(target_words=target_words)),
            HumanMessage(content=CRITIC_USER.format(
                theme=cluster.theme,writing_goal=cluster.writing_goal,
                target_words=target_words,round_num=round_num,
                draft=current_draft,source_content=source_content[:4_000],
            )),
        ])
        critique=crit_result.content
        turns.append(DebateTurn(role="critic",content=critique,round=round_num))
        critic_says(log,critique)

        rev_result=llm_writer.invoke([
            SystemMessage(content=WRITER_REVISION_SYSTEM.format(target_words=target_words)),
            HumanMessage(content=WRITER_REVISION_USER.format(
                theme=cluster.theme,target_words=target_words,
                draft=current_draft,critique=critique,
                source_content=source_content[:3_000],
            )),
        ])
        current_draft=rev_result.content
        turns.append(DebateTurn(role="writer",content=current_draft,round=round_num))
        writer_says(log,current_draft)

        # Final Polish 
    substep(log,"Final polish pass...")
    polish_result=llm_writer.invoke([
            SystemMessage(content=POLISH_SYSTEM.format(target_words=target_words)),
            HumanMessage(content=POLISH_USER.format(
                theme=cluster.theme, section_title=cluster.section_title,
                target_words=target_words,draft=current_draft,
            )),
        ])
    final_section = polish_result.content
    word_count = len(final_section.split())

    step(log, f"Section done -> {word_count}words | {max_rounds} rounds")
    result = DebateResult(
            cluster_id=cluster.cluster_id, cluster_theme = cluster.theme,
            section_title=cluster.section_title, debate_turns=turns,
            final_section = final_section, word_count=word_count,rounds_taken=max_rounds,
        )
    return {"debate_results":[result]}

#################stage 3 b - debate aggregator

def debate_aggregator(state:WriterState)->dict:
    results= state.get("debate_results",[])
    real_sections=[r for r in results if r.section_title != "__intro_conclusion__"]
    sorted_results=sorted(real_sections,key=lambda r: r.cluster_id)

    total_words = sum(r.word_count for r in sorted_results)
    total_rounds=sum(r.rounds_taken for r in sorted_results)

    phase_banner(log,4,"All Debates Complete")
    step(log, f"Section written : {len(sorted_results)}")
    step(log, f"Total words : {total_words:,}")
    step(log, f"Total rounds : {total_rounds}")
    for r in sorted_results:
        substep(log,f"[{r.cluster_id:02d}] {r.section_title[:50]:<50} {r.word_count:>5} words")

    return {"debate_results": sorted_results}


###########################stage 4 a Introduction + Conclusion Writer

def intro_conclusion_writer(state: WriterState,llm_strong)-> dict:
    writer_input = state['writer_input']
    cluster_plan = state['cluster_plan']
    debate_results= state.get("debate_results",[])

    phase_banner(log,5,"Writing Introduction and conclusion")
    section_summaries= "\n\n".join([
        f"Section {r.cluster_id}:{r.section_title}\n"
        f"Theme : {r.cluster_theme}\n"
        f"Opening: {r.final_section[:350]}..."
        for r in debate_results
        if r.section_title != "__intro_conclusion__"
        ])

    result=llm_strong.invoke([
        SystemMessage(content=ASSEMBLER_SYSTEM),
        HumanMessage(content=ASSEMBLER_USER.format(
            document_title=cluster_plan.document_title,
            topic=writer_input.topic,
            abstract=cluster_plan.document_abstract,
            section_summaries=section_summaries,
        )),
    ])
    text=result.content
    step(log, f"Intro + Conclusion -> {len(text.split())}words")

    placeholder=DebateResult(
        cluster_id=-1,cluster_theme="__framing__",
        section_title="__intro_conclusion__",
        debate_turns=[],final_section=text,
        word_count=len(text.split()), rounds_taken=0,
    )
    return {"debate_results":[placeholder]}


############################Stage 4b Document Assembler

def document_assembler(state : WriterState)-> dict:
    writer_input=state["writer_input"]
    cluster_plan=state["cluster_plan"]
    debate_results=state.get("debate_results",[])
    today = date.today().strftime("%B %d , %Y ")

    phase_banner(log,6,"Document Assembly")
    intro_conclusion_text=""
    body_sections:List[DebateResult]=[]

    for r in debate_results:
        if r.section_title=="__intro_conclusion__":
            intro_conclusion_text=r.final_section
        else:
            body_sections.append(r)
    body_sections.sort(key=lambda r: r.cluster_id)

    intro_text = conclusion_text="" 
    if intro_conclusion_text:
        if '## Conckusion' in intro_conclusion_text:
            split=intro_conclusion_text.split("## Conclusion",1)
            intro_text = split[0].replace('## Introduction', "").strip()
            conclusion_text=split[1].strip()
        else:
            intro_text = intro_conclusion_text.replace("## Introduction", "").strip()
    n_fetched_usable=len([f for f in state.get("fetched_sources",[])if f.content])
    total_rounds=sum(r.rounds_taken for r in body_sections)

    doc_parts:list[str]=[]
    doc_parts.append(
        f"# {cluster_plan.document_title}\n\n"
        f"> **Topic:** {writer_input.topic} \n"
        f"> **Generated:** {today}\n"
        f"> **Method:** Mirofish Writer<-> Critic debate ({total_rounds}total_rounds) \n"
        f"> ** Sources fetched: **{n_fetched_usable} \n"
        f"> **Sections:** {len(body_sections)} \n\n--\n"
    )
    doc_parts.append(f"## Abstract\n\n {cluster_plan.document_abstract}\n\n --\n")
    toc = ["## Table of Contents\n", "1. [Introduction](#introduction)"]
    for r in body_sections:
        slug = re.sub(r"[^\w\s-]", "", r.section_title.lower())
        slug = re.sub(r"\s+", "-", slug)
        toc.append(f"{r.cluster_id + 1}. [{r.section_title}](#{slug})")
    toc.append(f"{len(body_sections) + 2}. [Conclusion](#conclusion)")
    doc_parts.append("\n".join(toc) + "\n\n---\n")

    if intro_text:
        doc_parts.append(f"## Introduction\n\n{intro_text}\n\n---\n")
    for r in body_sections:
        doc_parts.append(f"## {r.section_title}\n\n{r.final_section}\n\n---\n")
    if conclusion_text:
        doc_parts.append(f"## Conclusion\n\n{conclusion_text}\n\n---\n")

    total_words = sum(len(p.split()) for p in doc_parts)
    doc_parts.append(
        f"*Document by Research Writer (Mirofish). "
        f"~{total_words:,} words | {len(body_sections)} sections | {total_rounds} debate rounds*\n"
    )

    full_document = "\n".join(doc_parts)
    step(log, f"Document assembled → ~{total_words:,} words | {len(body_sections)} sections")

    writer_output = WriterOutput(
        topic=writer_input.topic, document_title=cluster_plan.document_title,
        abstract=cluster_plan.document_abstract, sections=body_sections,
        full_document=full_document, output_path="",
        total_words=total_words, total_rounds=total_rounds,
        sources_used=n_fetched_usable,
    )
    return {"writer_output": writer_output}

################## Stage 4c Save Document

def save_document(state:WriterState)-> dict:
    """
    Write the final document and debate transcript to disk
    """
    writer_output=state['writer_output']
    topic = state['writer_input'].topic
    debate_results=state.get('debate_results',[])

    safe_topic = re.sub(r"[^\w\s-]","",topic.lower())
    safe_topic=re.sub(r"\s+","_",safe_topic)[:60]
    today_str=date.today().strftime("%Y%m%d")

    output_dir = Path(__file__).parent/"outputs"
    output_dir.mkdir(parents=True,exist_ok=True)


    #########Save Document#######################
    doc_path=output_dir / f'{today_str}_{safe_topic}_research_document.md'
    doc_path.write_text(writer_output.full_document,encoding="utf-8")
    doc_kb=doc_path.stat().st_size /1024
    success(log, f"Document saved -> {doc_path.name} ({doc_kb:.1f} KB)")

    ##################Save transcript
    transcript_lines:list[str] =[
        f' # Debate Transcript \n\n'
        f"**Topic:** {topic} \n"
        f"**Generated:** {date.today().strftime('%B %d, %Y')} \n\n---\n"
        
    ]
    for r in debate_results:
        if r.section_title== "__intro_conclusion__":
            continue
        transcript_lines.append(
            f"\n## Cluster [{r.cluster_id}]: {r.section_title}\n\n"
            f"*Theme : {r.cluster_theme} | Rounds : {r.rounds_taken}*\n"
        )
        for turn in r.debate_turns:
            icon = "✍" if turn.role == "writer" else "🔍"
            transcript_lines.append(
                f"\n### {icon} {turn.role.upper()} -- Round {turn.round}\n\n"
                f"{turn.content}\n\n ---\n"
            )
        transcript_lines.append(
            f"\n### FINAL \N\N{r.final_section}\n\n{'='*60}\n"            
        )
    transcript_path=output_dir /f"{today_str}_{safe_topic}_debate_transcript.md"
    transcript_path.write_text("\n".join(transcript_lines), encoding="utf-8")
    step(log,f"Transcript saved -> {transcript_path.name}")

    writer_output.output_path=str(doc_path.resolve())
    return {'writer_output': writer_output}