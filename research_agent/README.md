# 🔬 Research Agent

A **PhD-level autonomous research agent** built on LangGraph that performs exhaustive, systematic literature reviews from a single topic prompt. It searches across academic papers, preprints, patents, grey literature, and news — then synthesises everything into a structured, publication-quality research report.

---

## What It Does

Given a research topic, the agent:

1. **Decomposes** the topic into subtopics, disciplines, and 10–20 targeted search queries
2. **Validates** the query plan through a guardrail layer (Guardrail-1)
3. **Pauses for human review** (HITL-1) — you can approve, edit, or abort the query plan before any API calls are made
4. **Searches in parallel** across Tavily (web) + arXiv API + Semantic Scholar API
5. **Filters and cleans** the evidence corpus through a second guardrail (Guardrail-2)
6. **Critically evaluates** consensus areas, contested claims, key debates, and replication concerns
7. **Synthesises** three parallel outputs: Literature Summary, Knowledge Map, and Annotated Bibliography
8. **Assembles and saves** a final Markdown report to the `outputs/` folder

---

## Project Structure

```
research_agent/
├── agent.py          # Entry point: run_research_agent() — HITL resume loop
├── graph.py          # LangGraph graph definition and compilation
├── nodes.py          # All node functions (topic_decomposer, search_worker, etc.)
├── schemas.py        # Pydantic models and TypedDicts for all state objects
├── prompts.py        # All LLM system/user prompts (separated for easy iteration)
├── source_apis.py    # arXiv + Semantic Scholar API clients
├── logger.py         # Coloured console logger + rotating file logger
├── main.py           # CLI entry point
├── requirements.txt  # Python dependencies
├── outputs/          # Generated Markdown reports saved here
└── logs/             # Rotating log files
```

---

## Quickstart

### 1. Install dependencies

```bash
cd research_agent
pip install -r requirements.txt
```

### 2. Set up your `.env` file

Create a `.env` file in the **project root** (the folder above this one):

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. Run

```bash
# From the project root
python research_agent/main.py --topic "Transformer attention mechanisms"

# With optional scope and focus
python research_agent/main.py \
    --topic "CRISPR-Cas9 off-target effects in gene therapy" \
    --scope "Peer-reviewed biomedical research 2018–2025" \
    --focus "Emphasise safety, clinical trials, and regulatory landscape"

# Increase recursion limit for very broad topics
python research_agent/main.py --topic "Climate change and machine learning" --recursion-limit 300
```

### CLI Arguments

| Argument | Short | Default | Description |
|---|---|---|---|
| `--topic` | `-t` | *(required)* | The research topic |
| `--scope` | `-s` | None | Optional scope constraint (e.g. date range, domain) |
| `--focus` | `-f` | None | Optional synthesis angle |
| `--recursion-limit` | — | 250 | LangGraph recursion cap |

---

## Human-in-the-Loop (HITL)

After topic decomposition, the agent **pauses and shows you the full query plan** before executing any searches. You have three options:

| Key | Action |
|---|---|
| `a` | Approve — continue with the generated queries |
| `e` | Approve with edits — provide a JSON patch to add/remove/modify queries |
| `x` | Abort — stop the pipeline with an optional reason |

**Example edit patch:**
```json
{"add": [{"query": "speculative decoding llm inference", "domain": "academic_papers", "rationale": "Added manually"}], "remove": [2, 5], "modify": {"0": "large language model inference optimisation survey"}}
```

---

## Output

Reports are saved to `outputs/` as Markdown files, named:

```
YYYYMMDD_<topic_slug>_phd_research_report.md
```

Each report contains:
- **Output 1 — Literature Summary** (field overview, historical development, themes, debates, gaps)
- **Output 2 — Knowledge Map** (glossary, concept relationships, key researchers, landmark papers)
- **Output 3 — Annotated Bibliography** (30–50+ entries across 7 categories)

---

## Configuration

| Parameter | Values | Default | Description |
|---|---|---|---|
| `hitl_mode` | `full` / `none` | `full` | `none` skips the human review checkpoint |
| `guardrails_mode` | `strict` / `warn` / `off` | `strict` | `warn` downgrades blocks to warnings; `off` disables guardrails |

These can be passed programmatically via `run_research_agent()`:

```python
from research_agent import run_research_agent

state = run_research_agent(
    topic="Speculative decoding and continuous batching",
    hitl_mode="none",           # skip human review (for automation)
    guardrails_mode="warn",     # don't block, just log warnings
)
print(state["output_path"])
```

---

## APIs Used

| API | Purpose | Key Required |
|---|---|---|
| OpenAI (GPT-4o-mini) | Topic decomposition, evaluation, synthesis | Yes — `OPENAI_API_KEY` |
| Tavily Search | Web + academic web search | Yes — `TAVILY_API_KEY` |
| arXiv API | Preprint retrieval | No |
| Semantic Scholar API | Academic paper + citation data | No (rate-limited) |

---

## Standalone vs. Orchestrated

This agent is designed to **run independently** — just provide a topic and get a report.

It can also be invoked by the **Master Orchestrator** as part of a larger pipeline, where the orchestrator coordinates this agent alongside the Research Writer to produce a complete end-to-end research and writing workflow.

See the [main project README](../README.md) for the full architecture.
