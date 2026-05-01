from __future__ import annotations

import sys
from functools import partial
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph.graph import END,START ,StateGraph
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from .logger import get_logger,step
from .nodes import (
    fetch_worker,
    content_aggregator,
    cluster_planner,
    debate_worker,
    debate_aggregator,
    intro_conclusion_writer,
    document_assembler,
    save_document,
    _build_literature_context,
)
from .schemas import (
    WriterState,
    SourceFetchWorkerState,
    DebateWorkerState,
)
log= get_logger(__name__)

def _make_llms():
    llm_writer=ChatOpenAI(model = "gpt-4o-mini",temperature=0.5)
    llm_critic=ChatOpenAI(model = "gpt-4o-mini",temperature=0.2)
    llm_strong=ChatOpenAI(model = "gpt-4o-mini",temperature=0.2)
    return llm_writer,llm_critic,llm_strong

def _route_fetch_tasks(state:WriterState) -> List[Send]:
    writer_input = state['writer_input']
    sources=writer_input.sources
    pre_fetched_content=writer_input.pre_fetched_content

    cached_count = sum(1 for s in sources if s.url in pre_fetched_content)
    log.info(
        f" Routing{len(sources)} fetch tasks"
        f"({cached_count} pre-fetched, {len(sources) - cached_count} need HTTP)"
    )
    return [
        Send("fetch_worker", SourceFetchWorkerState(
            source=source,
            pre_fetched_content=pre_fetched_content,
            fetched_sources=[],
        ))
        for source in sources
    ]


def _route_debate_tasks(state:WriterState) -> List[Send]:
    writer_input = state['writer_input']
    cluster_plan = state['cluster_plan']
    fetched = state.get("fetched_sources",[])
    url_to_fetched = {f.url: f for f in fetched}
    lit_context = _build_literature_context(writer_input)
    fallback_sources = [f for f in fetched if f.content][:5]
    log.info(f" Routing {len(cluster_plan.clusters)} parallel debate tasks...")

    sends : List[Send]=[]
    for cluster in cluster_plan.clusters:
        cluster_sources=[
            url_to_fetched[url]
            for url in cluster.sources_urls
            if url in url_to_fetched
        ]
        if not cluster_sources:
            log.warning(f"Cluster [{cluster.cluster_id}] no matched sources - using fallback")
            cluster_sources = fallback_sources
        sends.append(Send("debate_worker",DebateWorkerState(
            cluster=cluster,
            fetched_sources=cluster_sources,
            topic = writer_input.topic,
            literature_context = lit_context[:2000],
            max_rounds=writer_input.max_debate_rounds,
            target_words=writer_input.target_section_words,
            debate_results=[],
        )))
    return sends

def build_writer_graph(
        recursion_limit:int =300
):
    """Build and compile the research writer graph"""
    llm_writer,llm_critic,llm_strong=_make_llms()

    _cluster_planner_node = partial(cluster_planner, llm_strong = llm_strong)
    _debate_worker_node = partial(debate_worker, llm_writer=llm_writer,llm_critic=llm_critic)
    _intro_conclusion_writer_node = partial(intro_conclusion_writer, llm_strong=llm_strong)


    builder = StateGraph(WriterState)

    builder.add_node("fetch_worker", fetch_worker)
    builder.add_node("content_aggregator", content_aggregator)
    builder.add_node("cluster_planner", _cluster_planner_node)
    builder.add_node('debate_worker', _debate_worker_node)
    builder.add_node('debate_aggregator', debate_aggregator)
    builder.add_node("intro_conclusion_writer", _intro_conclusion_writer_node)
    builder.add_node("document_assembler", document_assembler)
    builder.add_node("save_document", save_document)

    builder.add_contional_edges(START,_route_fetch_tasks,['fetch_worker'])
    builder.add_edge('fetch_worker', 'content_aggregator')
    builder.add_edge('content_aggregator', cluster_planner)
    builder.add_conditional_edges(
        'cluster_planner',
        _route_debate_tasks,
        ['debate_worker'],
    )
    builder.add_edge('debate_worker' , 'debate_aggregator')
    builder.add_edge('debate_aggregator', 'intro_conclusion_writer')
    builder.add_edge('intro_conclusion_writer', 'document_assembler')
    builder.add_edge('document_assembler', 'save_document')

    builder.add_edge('save_document', END)

    return builder.compile(recursion_limit=recursion_limit)



