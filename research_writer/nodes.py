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


