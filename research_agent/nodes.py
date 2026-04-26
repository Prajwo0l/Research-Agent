"""
nodes.py
────────
All LangGraph node functions for the PhD Research Agent.

Node execution order
────────────────────
  topic_decomposer          (Phase 1)
    ↓  [fan-out via Send]
  search_worker × N         (Phase 2, parallel)
    ↓  [fan-in via operator.add]
  evidence_aggregator       (Phase 2b)
    ↓
  guardrail_1_query_validator  (G1 — runs after topic_decomposer, before HITL-1)
    ↓
  hitl_1_query_review          (HITL-1 — human approves/edits/aborts)
    ↓  [fan-out via Send — now using approved queries]
  search_worker × N
    ↓
  evidence_aggregator
    ↓
  guardrail_2_source_quality   (G2 — cleans + validates evidence corpus)
    ↓
  critical_evaluator        (Phase 3)
    ↓  [fan-out via Send]
  synthesis_worker × 3      (Phase 4, parallel)
    ↓  [fan-in via operator.add]
  final_assembler
    ↓
  save_outputs
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults

sys.path.insert(0, str(Path(__file__).parent.parent))

from .logger import get_logger, phase_banner, step, substep, warn, success, section_title
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
    TopicDecomposition,
)
from .source_apis import enrich_with_apis

from shared.guardrails import (
    run_guardrail_1_query_validator,
    run_guardrail_2_source_quality,
)
from shared.exceptions import GuardrailError, PipelineAbortedError
from shared.hitl_cli import display_hitl_prompt, build_hitl1_payload

log = get_logger(__name__)

_DOMAIN_PREFIXES: dict[str, str] = {
    "academic_papers": (
        "site:scholar.google.com OR site:pubmed.ncbi.nlm.nih.gov "
        "OR site:semanticscholar.org {q}"
    ),
    "preprints":              "site:arxiv.org OR site:biorxiv.org OR site:ssrn.com {q}",
    "patents_applied":        "patent OR USPTO OR EPO {q} application",
    "grey_literature":        "technical report OR white paper OR dissertation OR policy {q}",
    "news_commentary":        "research news OR expert analysis {q} 2023 OR 2024 OR 2025",
    "conference_proceedings": "conference proceedings OR symposium OR workshop paper {q}",
}

_API_ENRICHED_DOMAINS = {"preprints", "academic_papers", "conference_proceedings"}

_OUTPUT_LABELS = {
    "literature_summary":     "OUTPUT 1: Literature Summary",
    "knowledge_map":          "OUTPUT 2: Knowledge Map",
    "annotated_bibliography": "OUTPUT 3: Annotated Bibliography",
}

_SYNTHESIS_SYSTEM_PROMPTS = {
    "literature_summary":     LITERATURE_SUMMARY_SYSTEM,
    "knowledge_map":          KNOWLEDGE_MAP_SYSTEM,
    "annotated_bibliography": ANNOTATED_BIBLIOGRAPHY_SYSTEM,
}


# ═══════════════════════════════════════════════════════════════════════════
# NODE 1 — Topic Decomposer
# ═══════════════════════════════════════════════════════════════════════════

def topic_decomposer(state: ResearchState, llm_fast) -> dict:
    topic = state["topic"]
    scope = state.get("scope") or "Exhaustive global coverage, all time periods"
    focus = state.get("focus") or "No specific angle — cover all major perspectives"

    phase_banner(log, 1, "Topic Decomposition")
    log.info(f"  Topic  : {topic}")
    log.info(f"  Scope  : {scope}")
    log.info(f"  Focus  : {focus}")

    decomposition: TopicDecomposition = (
        llm_fast
        .with_structured_output(TopicDecomposition)
        .invoke([
            SystemMessage(content=TOPIC_DECOMPOSER_SYSTEM),
            HumanMessage(content=TOPIC_DECOMPOSER_USER.format(
                topic=topic, scope=scope, focus=focus,
            )),
        ])
    )

    step(log, f"Subtopics identified : {len(decomposition.core_subtopics)}")
    step(log, f"Search queries built : {len(decomposition.search_queries)}")
    step(log, f"Disciplines          : {', '.join(decomposition.disciplines[:5])}")
    log.debug("Full decomposition:\n%s", decomposition.model_dump_json(indent=2))

    return {"decomposition": decomposition}


# ═══════════════════════════════════════════════════════════════════════════
# NODE G1 — Guardrail 1: Query Validator
# ═══════════════════════════════════════════════════════════════════════════

def guardrail_1_query_validator(state: ResearchState) -> dict:
    """
    Validate search queries. Runs BEFORE HITL-1 so human sees guardrail
    results in the approval screen.
    BLOCK rules raise GuardrailError when mode=strict.
    """
    mode   = state.get("guardrails_mode", "strict")
    decomp = state["decomposition"]

    step(log, f"Guardrail-1: Query Validator (mode={mode})")

    try:
        results = run_guardrail_1_query_validator(decomp, mode=mode)
    except GuardrailError:
        raise

    warns = sum(1 for r in results if r.is_warning())
    step(log, f"  → {len(results)} checks | {warns} warnings")

    return {"guardrail_1_results": results}


# ═══════════════════════════════════════════════════════════════════════════
# NODE HITL-1 — Query Plan Review
# ═══════════════════════════════════════════════════════════════════════════

def hitl_1_query_review(state: ResearchState) -> dict:
    """
    HITL checkpoint: pause and show human the query plan.
    Uses LangGraph interrupt() — state is checkpointed, resumes on response.

    Choices:
      a → approve (continue unchanged)
      e → approve with edits (JSON patch applied to search_queries)
      x → abort (raises PipelineAbortedError)
    """
    from langgraph.types import interrupt as _interrupt

    hitl_mode = state.get("hitl_mode", "full")
    if hitl_mode == "none":
        step(log, "HITL-1 skipped (hitl_mode=none)")
        return {}

    decomp     = state["decomposition"]
    thread_id  = state.get("thread_id", "unknown")
    gr_results = state.get("guardrail_1_results", [])

    log.info(f"  ⏸ HITL-1: Query Plan Review | thread={thread_id}")

    payload = build_hitl1_payload(decomp, gr_results, thread_id)
    display_hitl_prompt(payload)

    # Suspend — LangGraph checkpoints state here
    response = _interrupt(payload)

    choice       = response.get("choice", "a") if isinstance(response, dict) else "a"
    payload_data = response.get("payload")     if isinstance(response, dict) else None

    log.info(f"  HITL-1 response: choice='{choice}'")

    if choice == "x":
        reason = str(payload_data or "")
        warn(log, f"HITL-1: Aborted by human. Reason: {reason}")
        raise PipelineAbortedError("HITL-1: Query Plan Review", reason)

    if choice == "e" and isinstance(payload_data, dict):
        queries = list(decomp.search_queries)
        from .schemas import SearchQuery as _SQ

        for idx in sorted(payload_data.get("remove", []), reverse=True):
            if 0 <= idx < len(queries):
                queries.pop(idx)

        for idx_str, new_text in payload_data.get("modify", {}).items():
            idx = int(idx_str)
            if 0 <= idx < len(queries):
                old = queries[idx]
                queries[idx] = _SQ(query=new_text, domain=old.domain, rationale=old.rationale)

        for new_q in payload_data.get("add", []):
            if isinstance(new_q, dict):
                queries.append(_SQ(**new_q))
            elif isinstance(new_q, str):
                queries.append(_SQ(query=new_q, domain="academic_papers",
                                   rationale="Added by human reviewer"))

        decomp.search_queries = queries
        step(log, f"HITL-1: Applied edits → {len(queries)} queries")
        return {"decomposition": decomp}

    step(log, "HITL-1: Human approved query plan")
    return {}


# ═══════════════════════════════════════════════════════════════════════════
# NODE 2 — Search Worker
# ═══════════════════════════════════════════════════════════════════════════

def search_worker(state: SearchWorkerState) -> dict:
    """Execute a single search query via Tavily + academic API."""
    query  = state["query"]
    domain = state["domain"]

    display_q = query[:65] + "..." if len(query) > 65 else query
    log.info(f"  🔍 [{domain:<24}] {display_q}")

    template  = _DOMAIN_PREFIXES.get(domain, "{q}")
    augmented = template.replace("{q}", query)

    tavily_items: List[EvidenceItem] = []
    try:
        search = TavilySearchResults(
            max_results=8, search_depth="advanced",
            include_answer=True, include_raw_content=False, include_images=False,
        )
        raw_results = search.invoke(augmented)
        for r in raw_results:
            if isinstance(r, dict) and r.get("url"):
                snippet_raw = r.get("content") or r.get("snippet") or ""
                tavily_items.append(EvidenceItem(
                    title=r.get("title", "Untitled"), url=r.get("url", ""),
                    snippet=snippet_raw[:500],
                    source=f"tavily  score:{r.get('score','')}",
                    domain=domain, query_used=query,
                ))
    except Exception as exc:
        warn(log, f"Tavily failed for '{query}': {exc}")

    substep(log, f"Tavily      → {len(tavily_items)} results")

    api_items: List[EvidenceItem] = []
    if domain in _API_ENRICHED_DOMAINS:
        try:
            api_items = enrich_with_apis(query=query, domain=domain, max_results=8)
            substep(log, f"Academic API → {len(api_items)} results  [{domain}]")
        except Exception as exc:
            warn(log, f"Academic API enrichment failed for '{query}': {exc}")

    combined: List[EvidenceItem] = []
    seen_urls: set[str] = set()
    for item in tavily_items + api_items:
        if item.url and item.url not in seen_urls:
            seen_urls.add(item.url)
            combined.append(item)

    substep(log, f"Combined    → {len(combined)} unique items")
    return {"evidence_items": combined}


# ═══════════════════════════════════════════════════════════════════════════
# NODE 3 — Evidence Aggregator
# ═══════════════════════════════════════════════════════════════════════════

def evidence_aggregator(state: ResearchState) -> dict:
    """Global deduplication across all parallel search workers."""
    items = state.get("evidence_items", [])

    seen: set[str] = set()
    unique: List[EvidenceItem] = []
    for item in items:
        if item.url and item.url not in seen:
            seen.add(item.url)
            unique.append(item)

    source_counts: dict[str, int] = {"tavily": 0, "arxiv": 0, "semantic_scholar": 0, "other": 0}
    for item in unique:
        src = item.source or ""
        if "tavily" in src:           source_counts["tavily"] += 1
        elif "arxiv:" in src:         source_counts["arxiv"] += 1
        elif "ss:" in src:            source_counts["semantic_scholar"] += 1
        else:                         source_counts["other"] += 1

    domain_counts: dict[str, int] = {}
    for item in unique:
        domain_counts[item.domain] = domain_counts.get(item.domain, 0) + 1

    phase_banner(log, 2, "Evidence Aggregation Complete")
    step(log, f"Total raw results   : {len(items)}")
    step(log, f"Unique sources      : {len(unique)}")
    step(log, f"  via Tavily            : {source_counts['tavily']}")
    step(log, f"  via arXiv API         : {source_counts['arxiv']}")
    step(log, f"  via Semantic Scholar  : {source_counts['semantic_scholar']}")
    for domain, count in sorted(domain_counts.items()):
        substep(log, f"{domain:<30} {count} sources")

    return {"evidence_items": unique}


# ═══════════════════════════════════════════════════════════════════════════
# NODE G2 — Guardrail 2: Source Quality Filter
# ═══════════════════════════════════════════════════════════════════════════

def guardrail_2_source_quality(state: ResearchState) -> dict:
    """
    Validate and CLEAN the evidence corpus.
    Strips bad URLs, duplicates, and short snippets.
    BLOCK rules raise GuardrailError when mode=strict.
    """
    mode  = state.get("guardrails_mode", "strict")
    items = state.get("evidence_items", [])

    step(log, f"Guardrail-2: Source Quality Filter (mode={mode}, {len(items)} sources)")

    try:
        cleaned, results = run_guardrail_2_source_quality(items, mode=mode)
    except GuardrailError:
        raise

    warns = sum(1 for r in results if r.is_warning())
    step(log, f"  → {len(items)} → {len(cleaned)} sources | {warns} warnings")

    return {
        "evidence_items":      cleaned,
        "guardrail_2_results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# NODE 4 — Critical Evaluator
# ═══════════════════════════════════════════════════════════════════════════

def critical_evaluator(state: ResearchState, llm_strong) -> dict:
    topic         = state["topic"]
    items         = state.get("evidence_items", [])
    decomposition = state["decomposition"]

    phase_banner(log, 3, "Critical Evaluation")
    log.info(f"  Evaluating {len(items)} sources...")

    capped = items[:60]
    evidence_digest = "\n\n".join([
        f"[{i+1}] DOMAIN: {item.domain}\n"
        f"TITLE: {item.title}\n"
        f"URL: {item.url}\n"
        f"META: {(item.source or 'N/A')[:150]}\n"
        f"SNIPPET: {(item.snippet or 'N/A')[:300]}"
        for i, item in enumerate(capped)
    ])

    subtopics_str = "\n".join(f"  • {s}" for s in decomposition.core_subtopics)

    evaluation: EvidenceEvaluation = (
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

    step(log, f"Consensus areas        : {len(evaluation.consensus_areas)}")
    step(log, f"Contested claims       : {len(evaluation.contested_claims)}")
    step(log, f"Key debates            : {len(evaluation.key_debates)}")
    step(log, f"Replication concerns   : {len(evaluation.replication_concerns)}")

    return {"evaluation": evaluation}


# ═══════════════════════════════════════════════════════════════════════════
# NODE 5 — Synthesis Worker
# ═══════════════════════════════════════════════════════════════════════════

def synthesis_worker(state: SynthesisWorkerState, llm_strong) -> dict:
    output_type = state["output_type"]
    topic       = state["topic"]
    label       = _OUTPUT_LABELS[output_type]

    section_title(log, f"Writing {label}...")

    evidence_list = json.loads(state["evidence_json"])
    evaluation    = json.loads(state["evaluation_json"])
    decomp        = json.loads(state["decomposition_json"])

    evidence_digest = "\n\n".join([
        f"[{i+1}] {e.get('domain','?').upper()} | {e.get('title','Untitled')}\n"
        f"URL: {e.get('url','')}\n"
        f"META: {(e.get('source') or '')[:120]}\n"
        f"{(e.get('snippet') or '')[:300]}"
        for i, e in enumerate(evidence_list[:80])
    ])

    def _bullet_list(items: list) -> str:
        return "\n".join(f"  • {x}" for x in items) if items else "  (none identified)"

    user_msg = SYNTHESIS_USER_TEMPLATE.format(
        topic                    = topic,
        scope_statement          = decomp.get("scope_statement", ""),
        subtopics                = ", ".join(decomp.get("core_subtopics", [])),
        disciplines              = ", ".join(decomp.get("disciplines", [])),
        time_periods             = ", ".join(decomp.get("time_periods", [])),
        consensus_areas          = _bullet_list(evaluation.get("consensus_areas", [])),
        contested_claims         = _bullet_list(evaluation.get("contested_claims", [])),
        key_debates              = _bullet_list(evaluation.get("key_debates", [])),
        replication_concerns     = _bullet_list(evaluation.get("replication_concerns", [])),
        methodological_landscape = evaluation.get("methodological_landscape", ""),
        quality_notes            = evaluation.get("quality_notes", ""),
        n_sources                = len(evidence_list),
        evidence_digest          = evidence_digest,
        output_label             = label,
    )

    result = llm_strong.invoke([
        SystemMessage(content=_SYNTHESIS_SYSTEM_PROMPTS[output_type]),
        HumanMessage(content=user_msg),
    ])

    output_text = result.content
    step(log, f"{label} → {len(output_text.split()):,} words written")

    return {"synthesis_outputs": [output_text]}


# ═══════════════════════════════════════════════════════════════════════════
# NODE 6 — Final Assembler
# ═══════════════════════════════════════════════════════════════════════════

def final_assembler(state: ResearchState) -> dict:
    topic     = state["topic"]
    outputs   = state.get("synthesis_outputs", [])
    today     = date.today().strftime("%B %d, %Y")
    n_sources = len(state.get("evidence_items", []))

    items       = state.get("evidence_items", [])
    arxiv_count = sum(1 for e in items if "arxiv:" in (e.source or ""))
    ss_count    = sum(1 for e in items if "ss:"    in (e.source or ""))

    phase_banner(log, 4, "Final Assembly")

    lit_summary = knowledge_map = bibliography = ""
    for output in outputs:
        if "OUTPUT 1" in output or "Literature Summary" in output:
            lit_summary = output
        elif "OUTPUT 2" in output or "Knowledge Map" in output:
            knowledge_map = output
        elif "OUTPUT 3" in output or "Annotated Bibliography" in output:
            bibliography = output

    if not lit_summary:
        lit_summary   = outputs[0] if outputs else "(missing)"
    if not knowledge_map:
        knowledge_map = outputs[1] if len(outputs) > 1 else "(missing)"
    if not bibliography:
        bibliography  = outputs[2] if len(outputs) > 2 else "(missing)"

    header = (
        f"# 🎓 PhD Research Report\n\n"
        f"## Topic: {topic}\n\n"
        f"| Field | Value |\n|---|---|\n"
        f"| **Generated** | {today} |\n"
        f"| **Sources** | {n_sources} unique sources |\n"
        f"| **arXiv** | {arxiv_count} papers |\n"
        f"| **Semantic Scholar** | {ss_count} papers |\n"
        f"| **Subtopics** | {len(state['decomposition'].core_subtopics)} |\n\n"
        f"---\n\n"
        f"> ⚠️ All citations marked `[VERIFY]` should be independently confirmed.\n\n"
        f"---\n\n"
        f"## Table of Contents\n"
        f"1. [Literature Summary](#output-1)\n"
        f"2. [Knowledge Map](#output-2)\n"
        f"3. [Annotated Bibliography](#output-3)\n\n---\n"
    )

    final_report = "\n\n".join([header, lit_summary, "\n---\n", knowledge_map, "\n---\n", bibliography])
    step(log, f"Final report assembled : {len(final_report.split()):,} words")

    return {"final_report": final_report}


# ═══════════════════════════════════════════════════════════════════════════
# NODE 7 — Save Outputs
# ═══════════════════════════════════════════════════════════════════════════

def save_outputs(state: ResearchState) -> dict:
    topic  = state["topic"]
    report = state["final_report"]

    safe_topic = re.sub(r"[^\w\s-]", "", topic.lower())
    safe_topic = re.sub(r"\s+", "_", safe_topic)[:60]
    today_str  = date.today().strftime("%Y%m%d")
    filename   = f"{today_str}_{safe_topic}_phd_research_report.md"

    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    output_path.write_text(report, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    success(log, f"Report saved → {output_path.resolve()}")
    step(log, f"File size : {size_kb:.1f} KB")

    return {"output_path": str(output_path.resolve())}
