# 🧠 AI Research & Writing Suite

A modular, agent-based system for **end-to-end academic research and long-form writing** — powered by LangGraph, GPT-4, and real academic APIs.

The suite is composed of three independent components that can each be used standalone, or coordinated together by a central orchestrator.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    master_orchestrator                      │
│                                                             │
│   Coordinates the full research → writing pipeline.        │
│   Routes user intent, passes outputs between agents,       │
│   and exposes a single unified interface.                   │
└───────────────┬──────────────────────────┬──────────────────┘
                │                          │
                ▼                          ▼
┌──────────────────────┐      ┌────────────────────────┐
│    research_agent    │      │    research_writer     │
│                      │      │                        │
│  Given a topic:      │      │  Given research input: │
│  • Decomposes topic  │      │  • Writes long-form    │
│  • Searches sources  │      │    articles, papers,   │
│  • Evaluates evidence│      │    or reports          │
│  • Synthesises report│      │  • Adapts to audience  │
│                      │      │    and style           │
│  Output: .md report  │      │  Output: .md / .docx   │
└──────────────────────┘      └────────────────────────┘
```

Each component can run **independently** or be **orchestrated together**.

---

## Components

### 1. `research_agent/`

An autonomous PhD-level literature review agent. Feed it a topic; it searches arXiv, Semantic Scholar, Tavily, and more — then synthesises a structured report with a Literature Summary, Knowledge Map, and Annotated Bibliography.

- Human-in-the-loop query review before any searches run
- Guardrails on query quality and source quality
- Parallel search and synthesis workers via LangGraph fan-out
- Output: structured Markdown report

→ [research_agent README](research_agent/README.md)

---

### 2. `research_writer/` *(coming soon)*

A long-form writing agent that takes research material as input and produces polished written output — articles, blog posts, academic papers, or structured reports — in a specified style and for a specified audience.

- Outline planning before writing
- Section-by-section generation with internal consistency checks
- Citation formatting
- Output: Markdown or Word document

→ `research_writer/README.md` *(to be added)*

---

### 3. `master_orchestrator/` *(coming soon)*

The coordination layer. When a user wants to go from a raw topic all the way to a finished written piece, the orchestrator manages the handoff between `research_agent` and `research_writer`.

**Orchestration flow:**
```
User prompt
    │
    ▼
master_orchestrator
    │
    ├──► research_agent  →  literature review report
    │         │
    │         └──► (report passed as context)
    │
    └──► research_writer →  finished written output
```

The orchestrator also supports **partial runs** — you can invoke just the research agent, just the writer (if you already have research), or the full pipeline.

→ `master_orchestrator/README.md` *(to be added)*

---

## Project Structure

```
research_agent/          ← Project root
│
├── research_agent/      ← PhD literature review agent (standalone ✅)
│   ├── agent.py
│   ├── graph.py
│   ├── nodes.py
│   ├── schemas.py
│   ├── prompts.py
│   ├── source_apis.py
│   ├── logger.py
│   ├── main.py
│   └── outputs/
│
├── research_writer/     ← Long-form writing agent (planned 🔧)
│
├── master_orchestrator/ ← Orchestration layer (planned 🔧)
│
├── shared/              ← Shared utilities (guardrails, HITL CLI, exceptions)
│   ├── guardrails.py
│   ├── hitl_cli.py
│   └── exceptions.py
│
├── .env                 ← API keys (never commit this)
├── pyproject.toml
└── README.md            ← You are here
```

---

## Quickstart

### Prerequisites

- Python 3.10+
- An OpenAI API key
- A Tavily API key ([free tier available](https://tavily.com))

### Setup

```bash
# Clone or navigate to the project
cd research_agent

# Install dependencies
pip install -r research_agent/requirements.txt

# Add your API keys to .env
echo 'OPENAI_API_KEY=sk-...' >> .env
echo 'TAVILY_API_KEY=tvly-...' >> .env
```

### Run the research agent (standalone)

```bash
python research_agent/main.py --topic "Speculative Decoding and Continuous Batching"
```

### Run the full pipeline via orchestrator *(once built)*

```bash
python master_orchestrator/main.py \
    --topic "Speculative Decoding and Continuous Batching" \
    --output-format "blog post" \
    --audience "ML engineers"
```

---

## Roadmap

| Component | Status |
|---|---|
| `research_agent` — core pipeline | ✅ Complete |
| `research_agent` — HITL query review | ✅ Complete |
| `research_agent` — guardrails (G1 + G2) | ✅ Complete |
| `research_agent` — arXiv + Semantic Scholar APIs | ✅ Complete |
| `research_writer` — outline + section generation | 🔧 Planned |
| `research_writer` — citation formatting | 🔧 Planned |
| `master_orchestrator` — agent coordination | 🔧 Planned |
| `master_orchestrator` — unified CLI | 🔧 Planned |
| Web UI / API server | 🔧 Future |

---

## Environment Variables

| Variable | Required By | Description |
|---|---|---|
| `OPENAI_API_KEY` | `research_agent`, `research_writer`, `master_orchestrator` | OpenAI API key |
| `TAVILY_API_KEY` | `research_agent` | Tavily search API key |

Keep these in the `.env` file at the project root. Never commit `.env` to version control.

---

## Design Principles

**Modular by default.** Each agent (`research_agent`, `research_writer`) is fully functional on its own. The orchestrator adds coordination without coupling the components.

**Human-in-the-loop first.** Before expensive operations (parallel search, long synthesis), the system pauses for human review. You always see what the agent is about to do before it does it.

**Shared infrastructure.** Guardrails, HITL CLI utilities, and exceptions live in `shared/` so both agents benefit from the same validation and safety layer.

**LangGraph for orchestration.** All multi-step pipelines use LangGraph for reliable state management, checkpointing, and fan-out/fan-in parallelism.
