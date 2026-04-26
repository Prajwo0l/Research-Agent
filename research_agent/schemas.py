"""
schemas.py
──────────
All Pydantic schemas used across the PhD Research Agent pipeline.

v3 additions
────────────
  ResearchState gains HITL/guardrail fields:
    hitl_mode        — "full" | "report-only" | "none"
    guardrails_mode  — "strict" | "warn" | "off"
    guardrail_1_results  — results from query validator
    guardrail_2_results  — results from source quality filter
    thread_id            — for HITL resume display
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class SearchQuery(BaseModel):
    query: str = Field(..., description="A precise search query string (5–12 words).")
    domain: Literal[
        "academic_papers", "preprints", "patents_applied",
        "grey_literature", "news_commentary", "conference_proceedings",
    ] = Field(...)
    rationale: str = Field(...)


class TopicDecomposition(BaseModel):
    topic: str
    scope_statement: str = Field(...)
    core_subtopics: List[str] = Field(..., min_length=4, max_length=10)
    disciplines: List[str] = Field(...)
    time_periods: List[str] = Field(...)
    search_queries: List[SearchQuery] = Field(..., min_length=10, max_length=20)


class EvidenceItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    source: Optional[str] = None
    domain: str = "unknown"
    query_used: str = ""


class EvidenceEvaluation(BaseModel):
    consensus_areas: List[str] = Field(...)
    contested_claims: List[str] = Field(...)
    key_debates: List[str] = Field(...)
    replication_concerns: List[str] = Field(default_factory=list)
    methodological_landscape: str = Field(...)
    quality_notes: str = Field(...)


class SynthesisWorkerState(TypedDict):
    output_type: str
    topic: str
    decomposition_json: str
    evidence_json: str
    evaluation_json: str
    synthesis_outputs: Annotated[List[str], operator.add]


class SearchWorkerState(TypedDict):
    topic: str
    query: str
    domain: str
    rationale: str
    evidence_items: Annotated[List[EvidenceItem], operator.add]


class ResearchState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────
    topic: str
    scope: Optional[str]
    focus: Optional[str]

    # ── HITL / Guardrail config (injected at graph build time) ──────────
    hitl_mode:       str   # "full" | "report-only" | "none"
    guardrails_mode: str   # "strict" | "warn" | "off"
    thread_id:       str   # for display in HITL prompts

    # ── Phase 1 ──────────────────────────────────────────────────────────
    decomposition: TopicDecomposition

    # ── Guardrail results ─────────────────────────────────────────────────
    guardrail_1_results: List[Any]   # List[GuardrailResult]
    guardrail_2_results: List[Any]

    # ── Phase 2 ──────────────────────────────────────────────────────────
    evidence_items: Annotated[List[EvidenceItem], operator.add]

    # ── Phase 3 ──────────────────────────────────────────────────────────
    evaluation: EvidenceEvaluation

    # ── Phase 4 ──────────────────────────────────────────────────────────
    synthesis_outputs: Annotated[List[str], operator.add]

    # ── Final ────────────────────────────────────────────────────────────
    final_report: str
    output_path:  str
