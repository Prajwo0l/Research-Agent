# 🎓 PhD Research Agent

> An advanced, multi-phase AI research agent built with **LangGraph + GPT-4.1 + Tavily Search** that conducts exhaustive, PhD-level literature research on any topic and auto-generates three structured academic outputs.

---

## 📁 Project Structure

```
phd_research_agent/
│
├── __init__.py          ← Package entry-point (exposes run_research_agent)
├── main.py              ← CLI entry-point  (python main.py --topic "...")
│
├── agent.py             ← High-level orchestrator  (run_research_agent function)
├── graph.py             ← LangGraph builder  (nodes + edges + fan-out routers)
├── nodes.py             ← All 7 node functions
├── schemas.py           ← All Pydantic models + TypedDict state
├── prompts.py           ← All LLM system/user prompts (one place to edit)
├── logger.py            ← Centralised logger  (colourised console + rotating file)
│
├── outputs/             ← Generated reports saved here (.md files)
└── README.md            ← This file
```

**Log file location:** `logs/phd_research_agent.log` (auto-created on first run)

---

## 🏗 Architecture

```
START
  └─► topic_decomposer            Phase 1 — GPT-4.1-mini decomposes topic
                                            into 10–20 targeted queries
        │
        └─► [N × search_worker]   Phase 2 — Parallel Tavily searches
              (via Send)                    one per query, all domains
                │
                └─► evidence_aggregator    Deduplicates & counts sources
                      │
                      └─► critical_evaluator   Phase 3 — GPT-4.1 identifies
                                                consensus, debates, replication
                            │
                            └─► [3 × synthesis_worker]  Phase 4 — Three parallel
                                  (via Send)              GPT-4.1 writers
                                    ├─ literature_summary
                                    ├─ knowledge_map
                                    └─ annotated_bibliography
                                  │
                                  └─► final_assembler    Merges all 3 outputs
                                        │
                                        └─► save_outputs  Saves .md to outputs/
                                              │
                                              └─► END
```

**Key LangGraph patterns:**
- `Send()` for parallel fan-out of search workers and synthesis writers
- `Annotated[List, operator.add]` for fan-in aggregation of parallel results
- `functools.partial` to inject LLM instances into nodes without global state

---

## 📄 Three Outputs Generated

| Output | What it contains |
|--------|-----------------|
| **OUTPUT 1 — Literature Summary** | Field overview · Historical development · Core themes & findings · Methodological landscape · Debates & controversies · Applied/commercial landscape · Future directions |
| **OUTPUT 2 — Knowledge Map** | Core concepts glossary (15–30 terms) · Concept relationship map · Key research questions · Key researchers & institutions · Landmark papers & milestones · Datasets, benchmarks & tools |
| **OUTPUT 3 — Annotated Bibliography** | 30–50+ critically annotated references in 7 categories: Foundational · Recent high-impact · Preprints · Conference · Patents · Grey literature · Expert commentary |

---

## ⚙️ Prerequisites

All dependencies are already in your `requirements.txt`. Key ones:

```
langgraph>=0.6.4
langchain-openai>=0.3.29
langchain-community>=0.3.27
python-dotenv>=1.1.1
pydantic>=2.11
```

API keys needed (already in your `.env`):
- `OPENAI_API_KEY` — GPT-4.1 and GPT-4.1-mini
- `TAVILY_API_KEY` — Web search across all domains

---

## 🚀 How to Run

### Step 1 — Activate your environment

```bash
cd C:\Users\lamic\Desktop\LangGraph-Core-Components

# Windows CMD
myenv\Scripts\activate

# Windows PowerShell
myenv\Scripts\Activate.ps1
```

### Step 2 — Navigate to the agent folder

```bash
cd AI_Agents\phd_research_agent
```

---

### Option A — Run from CLI (recommended)

#### Minimal run
```bash
python main.py --topic "Transformer attention mechanisms"
```

#### Full run with all options
```bash
python main.py \
  --topic  "CRISPR-Cas9 off-target effects in therapeutic gene editing" \
  --scope  "Peer-reviewed biomedical research 2018–2025, clinical focus" \
  --focus  "Emphasise safety profiles, clinical trial results, and regulatory gaps"
```

#### All CLI flags
```
--topic  / -t    Research topic (REQUIRED)
--scope  / -s    Scope constraint — time period, geography, sub-domain (optional)
--focus  / -f    Synthesis angle to emphasise (optional)
--recursion-limit  LangGraph recursion cap, default 250 (increase for broad topics)
```

---

### Option B — Import as a Python module

```python
from phd_research_agent import run_research_agent

final_state = run_research_agent(
    topic = "Large Language Models in Scientific Discovery",
    scope = "Peer-reviewed work 2020–2025, natural sciences and biomedicine",
    focus = "Emphasise reproducibility concerns and real-world utility gaps",
)

# Access results
print(final_state["output_path"])          # Path to saved .md file
print(final_state["final_report"][:2000])  # First 2000 chars of the report

# Individual sections
lit_summary  = final_state["synthesis_outputs"][0]
knowledge_map = final_state["synthesis_outputs"][1]
bibliography  = final_state["synthesis_outputs"][2]

# Evidence corpus
for item in final_state["evidence_items"][:5]:
    print(item.title, "|", item.domain, "|", item.url)
```

---

## 📊 What the Console Output Looks Like

```
╔══════════════════════════════════════════════════════════════╗
║           🎓  PhD RESEARCH AGENT  (LangGraph)               ║
╚══════════════════════════════════════════════════════════════╝

16:42:01  [INFO]   phd_agent  Topic  : Large Language Models in Scientific Discovery
16:42:01  [INFO]   phd_agent  Scope  : Peer-reviewed work 2020–2025
16:42:01  [INFO]   phd_agent  ══════════════════════════════════════════════════════

  PHASE 1 — TOPIC DECOMPOSITION
  ════════════════════════════════════════════════════════════════
16:42:04  [INFO]   nodes   ✓  Subtopics identified : 7
16:42:04  [INFO]   nodes   ✓  Search queries built : 15
16:42:04  [INFO]   nodes   ✓  Disciplines          : Computer Science, Chemistry...

16:42:04  [INFO]   graph   Routing 15 parallel search tasks...
16:42:04  [INFO]   nodes   🔍 [academic_papers       ] LLM scientific discovery benchmarks
16:42:04  [INFO]   nodes   🔍 [preprints             ] site:arxiv.org LLM protein structure
   ...
16:42:18  [INFO]   nodes      → Found 8 results
16:42:18  [INFO]   nodes      → Found 7 results

  PHASE 2 — EVIDENCE AGGREGATION COMPLETE
  ════════════════════════════════════════════════════════════════
16:42:20  [INFO]   nodes   ✓  Total raw results   : 104
16:42:20  [INFO]   nodes   ✓  Unique sources      : 81
16:42:20  [INFO]   nodes      → academic_papers        28 sources
16:42:20  [INFO]   nodes      → preprints              19 sources
16:42:20  [INFO]   nodes      → news_commentary        16 sources
   ...

  PHASE 3 — CRITICAL EVALUATION
  ════════════════════════════════════════════════════════════════
16:42:28  [INFO]   nodes   ✓  Consensus areas        : 5
16:42:28  [INFO]   nodes   ✓  Contested claims       : 4
16:42:28  [INFO]   nodes   ✓  Key debates            : 3

16:42:28  [INFO]   graph   Routing 3 parallel synthesis tasks...
16:42:28  [INFO]   nodes   ── Writing OUTPUT 1: Literature Summary...
16:42:28  [INFO]   nodes   ── Writing OUTPUT 2: Knowledge Map...
16:42:28  [INFO]   nodes   ── Writing OUTPUT 3: Annotated Bibliography...
16:42:55  [INFO]   nodes   ✓  OUTPUT 1 → 2,847 words written
16:42:57  [INFO]   nodes   ✓  OUTPUT 2 → 3,102 words written
16:43:01  [INFO]   nodes   ✓  OUTPUT 3 → 4,201 words written

  PHASE 4 — FINAL ASSEMBLY
  ════════════════════════════════════════════════════════════════
16:43:02  [INFO]   nodes   ✓  Final report assembled : 10,412 words

16:43:02  [INFO]   nodes   ✅ Report saved → ...outputs\20260412_..._phd_research_report.md
══════════════════════════════════════════════════════════════
✅  Done.  Report → C:\...\outputs\20260412_..._phd_research_report.md
```

---

## 📂 Output File

Reports are saved to:
```
phd_research_agent/outputs/YYYYMMDD_<topic_slug>_phd_research_report.md
```

Open with: **VS Code**, **Obsidian**, **Typora**, or any Markdown reader.

---

## 🔧 Configuration

### Editing prompts
All prompts live in `prompts.py`. Edit any `*_SYSTEM` or `*_USER` string to change
agent behaviour — no need to touch node logic.

### Changing the LLMs
In `graph.py`, edit `_make_llms()`:
```python
def _make_llms():
    llm_strong = ChatOpenAI(model="gpt-4o",      temperature=0.2)   # heavier model
    llm_fast   = ChatOpenAI(model="gpt-4.1-mini", temperature=0.1)
    return llm_strong, llm_fast
```

### Changing the number of search results per query
In `nodes.py`, find `TavilySearchResults(max_results=8, ...)` and change `max_results`.

### Changing output location
In `nodes.py`, find `save_outputs()` and edit `output_dir`.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `AuthenticationError` | Check `.env` has valid `OPENAI_API_KEY` and `TAVILY_API_KEY` |
| `RecursionError` | Add `--recursion-limit 350` to the CLI command |
| Empty evidence corpus | Check Tavily API credits; try narrowing `--topic` |
| `[VERIFY]` tags in output | Intentional — those citations need manual confirmation |
| Synthesis output misidentified | Outputs are matched by "OUTPUT 1/2/3" keyword; re-run if blank |
| Log file not created | The `logs/` folder is created automatically next to `main.py` |

---

## 📝 Logging Details

| Handler | Level | Location |
|---------|-------|----------|
| Console (colourised) | INFO+ | Terminal |
| Rotating file | DEBUG+ | `logs/phd_research_agent.log` (5 MB × 3 backups) |

To increase console verbosity to DEBUG, edit `logger.py`:
```python
ch.setLevel(logging.DEBUG)
```

---

## ⚠️ Academic Use Disclaimer

This tool **assists** research — it does not replace researcher judgment.

- All `[VERIFY]` tags must be independently confirmed
- Do not submit AI-generated text to academic venues without review
- Bibliography entries should be cross-checked against actual sources
- The agent may hallucinate details when evidence is sparse — always verify

---

*Built with LangGraph · GPT-4.1 · Tavily Search · Python*
