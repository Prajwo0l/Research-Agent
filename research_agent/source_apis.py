"""
source_apis.py
──────────────
Real academic API integrations for the PhD Research Agent.

Provides direct, structured access to:
  1. arXiv API      — preprints, PDFs, exact dates, version history
  2. Semantic Scholar API — citation counts, open-access full text, author h-index

Both clients are designed to be called from search_worker as drop-in enrichment
on top of Tavily results, and are also callable standalone.

Public functions
────────────────
  fetch_arxiv(query, max_results)       → List[EvidenceItem]
  fetch_semantic_scholar(query, max_results) → List[EvidenceItem]
  enrich_with_apis(query, domain)       → List[EvidenceItem]
    (called by search_worker — selects the right API based on domain)
"""

from __future__ import annotations

import time
from typing import List, Optional
from urllib.parse import quote_plus

import httpx

from .logger import get_logger, substep, warn
from .schemas import EvidenceItem

log = get_logger(__name__)

# ── HTTP client config ────────────────────────────────────────────────────────
_TIMEOUT = 15          # seconds
_HEADERS = {
    "User-Agent": "PhDResearchAgent/1.0 (academic-research; contact: research@example.com)",
    "Accept":     "application/json",
}

# ── Rate limiting (be a good API citizen) ────────────────────────────────────
_ARXIV_DELAY_SECONDS          = 3.0   # arXiv asks for ≥3s between requests
_SEMANTIC_SCHOLAR_DELAY_SECS  = 1.0   # Semantic Scholar: generous but rate-limited


# ═══════════════════════════════════════════════════════════════════════════
# 1 — arXiv API
#
# Docs: https://arxiv.org/help/api/user-manual
# Endpoint: http://export.arxiv.org/api/query
# Free, no API key required.
# Returns Atom XML — we parse it with stdlib xml.etree.
# ═══════════════════════════════════════════════════════════════════════════

def fetch_arxiv(
    query: str,
    max_results: int = 10,
    sort_by: str = "relevance",        # "relevance" | "lastUpdatedDate" | "submittedDate"
    sort_order: str = "descending",
) -> List[EvidenceItem]:
    """
    Query the arXiv API and return structured EvidenceItems.

    Parameters
    ----------
    query       : Search query string (natural language or arXiv search syntax)
    max_results : Max papers to return (default 10, max 100)
    sort_by     : Sort field — relevance | lastUpdatedDate | submittedDate
    sort_order  : ascending | descending

    Returns
    -------
    List[EvidenceItem] — each item has:
        title    : paper title
        url      : arXiv abstract page URL (https://arxiv.org/abs/<id>)
        snippet  : abstract (truncated to 500 chars)
        source   : "arxiv:<id>  v<version>  <date>  cited-by:<N if known>"
        domain   : "preprints"
    """
    import xml.etree.ElementTree as ET

    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{quote_plus(query)}",
        "start":        0,
        "max_results":  min(max_results, 100),
        "sortBy":       sort_by,
        "sortOrder":    sort_order,
    }

    log.debug(f"arXiv query: {query!r} (max={max_results})")

    try:
        # arXiv asks for polite rate limiting
        time.sleep(_ARXIV_DELAY_SECONDS)

        resp = httpx.get(
            base_url,
            params=params,
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )
        resp.raise_for_status()
    except Exception as exc:
        warn(log, f"arXiv fetch failed: {exc}")
        return []

    # ── Parse Atom XML ────────────────────────────────────────────────────
    ns = {
        "atom":   "http://www.w3.org/2005/Atom",
        "arxiv":  "http://arxiv.org/schemas/atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    }

    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError as exc:
        warn(log, f"arXiv XML parse error: {exc}")
        return []

    items: List[EvidenceItem] = []
    for entry in root.findall("atom:entry", ns):
        # Paper ID (last segment of <id> URL, strip version)
        id_url  = (entry.findtext("atom:id", "", ns) or "").strip()
        arxiv_id = id_url.split("/abs/")[-1] if "/abs/" in id_url else id_url

        title   = (entry.findtext("atom:title",   "", ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")
        updated = (entry.findtext("atom:updated", "", ns) or "").strip()[:10]   # YYYY-MM-DD
        published = (entry.findtext("atom:published", "", ns) or "").strip()[:10]

        # Authors (first 3)
        author_names = [
            (a.findtext("atom:name", "", ns) or "").strip()
            for a in entry.findall("atom:author", ns)
        ]
        authors_str = ", ".join(author_names[:3])
        if len(author_names) > 3:
            authors_str += " et al."

        # Categories / primary category
        primary_cat = ""
        cat_el = entry.find("arxiv:primary_category", ns)
        if cat_el is not None:
            primary_cat = cat_el.get("term", "")

        # Version info from <link title="pdf"> href
        version = "v1"
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "pdf":
                href = link.get("href", "")
                if "v" in href.split("/")[-1]:
                    version = "v" + href.split("/")[-1].split("v")[-1]
                break

        # Compose source metadata string
        source_meta = f"arxiv:{arxiv_id}  {version}  published:{published}  updated:{updated}  authors:{authors_str}  category:{primary_cat}"

        items.append(EvidenceItem(
            title=title,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            snippet=summary[:500],
            source=source_meta,
            domain="preprints",
            query_used=query,
        ))

    substep(log, f"arXiv → {len(items)} papers for: {query[:50]}")
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 2 — Semantic Scholar API
#
# Docs: https://api.semanticscholar.org/graph/v1
# Free for non-commercial use, no API key required (rate-limited).
# Returns JSON with citation counts, open-access PDFs, author h-indices.
# ═══════════════════════════════════════════════════════════════════════════

_SS_PAPER_FIELDS = ",".join([
    "paperId",
    "title",
    "abstract",
    "year",
    "citationCount",
    "influentialCitationCount",
    "openAccessPdf",
    "authors",
    "venue",
    "publicationDate",
    "externalIds",
    "fieldsOfStudy",
])

_SS_BASE = "https://api.semanticscholar.org/graph/v1"


def fetch_semantic_scholar(
    query: str,
    max_results: int = 10,
    min_citation_count: int = 0,
    fields_of_study: Optional[List[str]] = None,
) -> List[EvidenceItem]:
    """
    Query the Semantic Scholar paper search API.

    Parameters
    ----------
    query              : Natural language search query
    max_results        : Papers to return (default 10, max 100)
    min_citation_count : Filter out papers with fewer citations (default 0)
    fields_of_study    : Optional filter e.g. ["Computer Science", "Biology"]

    Returns
    -------
    List[EvidenceItem] — each item has:
        title   : paper title
        url     : Semantic Scholar paper page, or open-access PDF URL if available
        snippet : abstract (truncated to 500 chars)
        source  : "ss:<id>  cited:{N}  influential:{N}  year:{Y}  venue:{V}"
        domain  : "academic_papers"
    """
    params: dict = {
        "query":  query,
        "limit":  min(max_results, 100),
        "fields": _SS_PAPER_FIELDS,
    }
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    log.debug(f"Semantic Scholar query: {query!r} (max={max_results})")

    try:
        time.sleep(_SEMANTIC_SCHOLAR_DELAY_SECS)
        resp = httpx.get(
            f"{_SS_BASE}/paper/search",
            params=params,
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 429:
            warn(log, "Semantic Scholar rate-limited — skipping (add S2_API_KEY for higher limits)")
        else:
            warn(log, f"Semantic Scholar HTTP error {exc.response.status_code}: {exc}")
        return []
    except Exception as exc:
        warn(log, f"Semantic Scholar fetch failed: {exc}")
        return []

    papers = data.get("data", [])
    items: List[EvidenceItem] = []

    for p in papers:
        citation_count    = p.get("citationCount") or 0
        influential_count = p.get("influentialCitationCount") or 0

        # Apply citation filter
        if citation_count < min_citation_count:
            continue

        title    = (p.get("title") or "Untitled").strip()
        abstract = (p.get("abstract") or "").strip()
        year     = p.get("year") or ""
        venue    = (p.get("venue") or "").strip()
        paper_id = p.get("paperId") or ""
        pub_date = (p.get("publicationDate") or str(year))[:10]

        # Authors (first 3)
        author_names = [
            a.get("name", "") for a in (p.get("authors") or [])
        ]
        authors_str = ", ".join(author_names[:3])
        if len(author_names) > 3:
            authors_str += " et al."

        # Prefer open-access PDF URL, fall back to S2 page
        oa_pdf  = (p.get("openAccessPdf") or {}).get("url", "")
        page_url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else ""
        url = oa_pdf or page_url

        if not url:
            continue

        # DOI for cross-referencing
        doi = (p.get("externalIds") or {}).get("DOI", "")

        source_meta = (
            f"ss:{paper_id}  cited:{citation_count}  influential:{influential_count}"
            f"  year:{year}  venue:{venue}  authors:{authors_str}"
            + (f"  doi:{doi}" if doi else "")
            + (f"  oa_pdf:yes" if oa_pdf else "")
        )

        items.append(EvidenceItem(
            title=title,
            url=url,
            snippet=abstract[:500],
            source=source_meta,
            domain="academic_papers",
            query_used=query,
        ))

    # Sort by citation count descending (highest-impact first)
    items.sort(
        key=lambda e: int((e.source or "").split("cited:")[-1].split()[0]) if "cited:" in (e.source or "") else 0,
        reverse=True,
    )

    substep(log, f"Semantic Scholar → {len(items)} papers for: {query[:50]}")
    return items


# ═══════════════════════════════════════════════════════════════════════════
# 3 — Domain Router
#
# Called by search_worker to select the best API(s) for a given domain.
# Returns additional EvidenceItems to MERGE with Tavily results.
# ═══════════════════════════════════════════════════════════════════════════

def enrich_with_apis(
    query: str,
    domain: str,
    max_results: int = 8,
) -> List[EvidenceItem]:
    """
    Select and call the appropriate academic API based on the query domain.
    Returns items to be merged with the Tavily results in search_worker.

    Domain routing
    ──────────────
    preprints              → arXiv API
    academic_papers        → Semantic Scholar API
    conference_proceedings → Semantic Scholar API (venue filter)
    patents_applied        → (no free API — Tavily handles this)
    grey_literature        → (no structured API — Tavily handles this)
    news_commentary        → (no structured API — Tavily handles this)
    """
    if domain == "preprints":
        return fetch_arxiv(query, max_results=max_results)

    if domain == "academic_papers":
        return fetch_semantic_scholar(query, max_results=max_results)

    if domain == "conference_proceedings":
        # Semantic Scholar covers most major CS/ML/Bio venues
        return fetch_semantic_scholar(query, max_results=max_results)

    # Other domains handled by Tavily only
    return []
