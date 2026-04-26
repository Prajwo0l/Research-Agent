from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List,Optional 
from urllib.parse import urlparse

from .exceptions import GuardrailError

log=logging.getLogger("guardrails")

_COLLOQUIALISMS = [
    "basically", "stuff", "things", "a lot", "pretty much",
    "kind of", "sort of", "you know", "like", "awesome",
    "amazing", "incredibly", "super", "really", "very",
]

# ── Passive voice pattern ────────────────────────────────────────────────────
_PASSIVE_PATTERN = re.compile(
    r'\b(is|are|was|were|be|been|being)\s+\w+ed\b',
    re.IGNORECASE,
)

_CITATION_PATTERN = re.compile(r'\(([A-Za-z][^,)]+),\s*(1[89]\d{2}|2[01]\d{2})\)')

_DOI_PATTERN = re.compile(r'10\.\d{4,}/\S+')

_URL_IN_TEXT_PATTERN = re.compile(r'https?://[^\s\)\]\>\"]+')


@dataclass 
class GuardrailResult:
    """Result of a single guardrail check."""
    guardrail :str
    status : str
    message:str
    details:dict=field(default_factory=dict)

    def is_blocking(self)-> bool:
        return self.status == "block"
    def is_warning(self) -> bool:
        return self.status =="warn"
    
def _apply_or_raise(
    result : GuardrailResult,
    mode:str,
)-> GuardrailResult:
    """Apply the guardrail mode:
    strict Block raises GuardrailError
    warn downgrade BLOCK to WARN
    off return pass regardless
    
    Always logs at the appropriate level.
    """

    if mode == "warn" and result.status == "block":
        result = GuardrailResult(
            guardrail=result.guardrail,
            status="warn",
            message=f"[downgraded to warn] {result.message}",
            details=result.details,
        )

    # Log at appropriate level
    if result.status == "block":
        log.error(
            "[%s] BLOCK: %s | details=%s",
            result.guardrail, result.message, result.details,
        )
    elif result.status == "warn":
        log.warning(
            "[%s] WARN: %s | details=%s",
            result.guardrail, result.message, result.details,
        )
    else:
        log.debug("[%s] PASS: %s", result.guardrail, result.message)

    return result

# Guardrail 1 -Query Validator

def run_guardrail_1_query_validator(
        decomposition,
        mode:str="strict",
)-> List[GuardrailResult]:
    """
    Validate the search query plan produced by topic_decomposer.

    BLOCK rules (halt if mode=strict):
      - Fewer than 5 queries
      - Any query fewer than 4 words
      - More than 25 queries
      - Duplicate queries (case-insensitive)

    WARN rules (log and continue):
      - Fewer than 10 queries
      - All queries in same domain
    """
    NAME    = "Guardrail-1: Query Validator"
    queries = decomposition.search_queries
    n       = len(queries)
    results: List[GuardrailResult] = []
    
    #block too few
    if n < 5:
        r = _apply_or_raise(GuardrailResult(
            guardrail=NAME, status="block",
            message=f"Too few queries: {n}. Minimum is 5 for adequate coverage.",
            details={"query_count": n},
        ), mode)
        results.append(r)
        if r.is_blocking():
            raise GuardrailError(NAME,r.message,r.details)
        
    if n > 25:
        r = _apply_or_raise(GuardrailResult(
            guardrail=NAME,status="block",
            message=f"Too many queries:{n}. Maximum is 25 to avoid runway API costs",
            details={"query_count":n},
        ),mode)
        results.append(r)
        if r.is_blocking():
            raise GuardrailError(NAME,r.message,r.details)
    
    #Block :any qyery too vague
    vague =[sq.query for sq in queries if len(sq.query.split()) < 4]
    for q in vague:
        r = _apply_or_raise(GuardrailResult(
            guardrail=NAME,status="block",
            message=f"Query too vague(< 4 words):'{q}'",
            details={'offending_query':q},
        ),mode)
        results.append(r)
        if r.is_blocking():
            raise GuardrailError(NAME,r.message,r.details)
        
    
    # Block Duplicates
    seen: set[str] = set()
    for sq in queries:
        normalised = sq.query.lower().strip()
        if normalised in seen:
            r = _apply_or_raise(GuardrailResult(
                guardrail=NAME,status="block",
                message=f"Duplicate query detected: '{sq.query}'",
                details={'duplicate':sq.query},
            ),mode)
            results.append(r)
            if r.is_blocking():
                raise GuardrailError(NAME,r.message,r.details)
        seen.add(normalised)

    
    # Warn fewer than 10
    if 5 <=n< 10:
        results.append(_apply_or_raise(GuardrailResult(
            guardrail=NAME,status='warn',
            message=f"Only {n} queries - consider broadening topic for richer coverage.",
            details={'query_Count':n},
        ),mode))

    # wRN all queries in same domain
    domains = {sq.domain for sq in queries}
    if len(domains)==1:
        d = next(iter(domains))
        results.append(_apply_or_raise(GuardrailResult(
            guardrail =NAME,status="warn",
            message=f"All queries are domain '{d}' - source diversity may be low",
            details={'single_domain':d},
        ),mode))
    
    if not results:
        results.append(_apply_or_raise(GuardrailResult(
            guardrail=NAME, status="pass",
            message=f"All {n} queries passed validation.",
            details={'query_count': n},
        ), mode))
    return results


def run_guardrail_2_source_quality(
    items: list,
    mode: str = "strict",
):
    """
    Validate and clean the evidence corpus.
    Removes items with missing/bad URLs, duplicate URLs, and very short snippets.

    BLOCK rules (halt if mode=strict):
      - Fewer than 3 valid sources remain after cleaning

    WARN rules (log and continue):
      - More than 30% of sources were stripped
    """
    NAME = "Guardrail-2: Source Quality Filter"
    results: List[GuardrailResult] = []

    original_count = len(items)
    cleaned: list = []
    seen_urls: set = set()

    for item in items:
        url = getattr(item, "url", "") or ""
        snippet = getattr(item, "snippet", "") or ""

        # Skip items with no URL or obviously bad URLs
        if not url or not url.startswith("http"):
            continue

        # Skip duplicates
        if url in seen_urls:
            continue

        # Skip items with very short snippets (likely empty/junk)
        if len(snippet.strip()) < 20:
            continue

        seen_urls.add(url)
        cleaned.append(item)

    removed = original_count - len(cleaned)
    removal_pct = (removed / original_count * 100) if original_count > 0 else 0

    # WARN: high removal rate
    if removal_pct > 30:
        results.append(_apply_or_raise(GuardrailResult(
            guardrail=NAME, status="warn",
            message=f"{removal_pct:.0f}% of sources removed during quality filtering ({removed}/{original_count}).",
            details={"removed": removed, "original": original_count},
        ), mode))

    # BLOCK: too few sources remain
    if len(cleaned) < 3:
        r = _apply_or_raise(GuardrailResult(
            guardrail=NAME, status="block",
            message=f"Only {len(cleaned)} valid sources remain after filtering. Minimum is 3.",
            details={"remaining": len(cleaned)},
        ), mode)
        results.append(r)
        if r.is_blocking():
            raise GuardrailError(NAME, r.message, r.details)

    if not results:
        results.append(_apply_or_raise(GuardrailResult(
            guardrail=NAME, status="pass",
            message=f"All {len(cleaned)} sources passed quality filter.",
            details={"remaining": len(cleaned)},
        ), mode))

    return cleaned, results


