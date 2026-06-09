# Multi-Agent Company & Finance Research Assistant

A multi-agent NLP application that answers questions about companies and finance
by routing each query to the right specialist. Built with **LangGraph** and
**Google Gemini**, it extends the single-agent pattern from the course reference
(`module5/01-multi-agent_system`) into a genuine **supervisor + specialists**
architecture.

> **Course**: BBS-AIIM — LLM and Generative AI, final project.
> **Track**: *AI agent with tool use for a defined task* (combining the
> data-analyst, document-Q&A, and web-research agent examples into one system).
> **Group**: Alessio Capobianco, Elena La Chiusa, Francesco Pirovano.

---

## 1. What it does

Ask a finance question. A **supervisor** classifies it and dispatches to one (or
several, in sequence) of three specialist agents, then a **synthesizer** writes
the final answer:

| Specialist     | Tool(s)                              | Good for                                            |
|----------------|--------------------------------------|-----------------------------------------------------|
| `document_qa`  | RAG over a Chroma vector store       | Definitions, company background, history            |
| `data_analyst` | pandas tools over a bundled dataset  | Rankings, aggregations, comparisons (verifiable)    |
| `web_research` | live DuckDuckGo search (no API key)  | Current prices, recent news, today's events         |

Example multi-hop query: *"Explain what market capitalization means, then list
the top 3 companies by it in the dataset."* → supervisor calls `document_qa`
(definition) then `data_analyst` (ranking), and the synthesizer merges both.

```
                        ┌──────────────┐
       user query ─────▶│  SUPERVISOR  │◀───────────────┐
                        │  (router)    │                │
                        └──────┬───────┘                │ specialist
           ┌──────────────────┼──────────────────┐      │ reports
           ▼                  ▼                  ▼      │
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
   │ DOCUMENT-Q&A │  │ DATA-ANALYST │  │ WEB-RESEARCH │─┘
   │  (RAG/Chroma)│  │  (pandas)    │  │ (DuckDuckGo) │
   └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
          └─────────────────┴──────────────────┘
                            │ supervisor says FINISH
                            ▼
                     ┌──────────────┐
                     │  SYNTHESIZE  │ ─▶ final answer
                     └──────────────┘
```

---

## 2. Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your free Google Gemini key (https://aistudio.google.com/apikey)
cp .env.example .env        # then paste your key into .env

# 3a. Command-line chat
python -m src.app_cli
python -m src.app_cli "Which company has the highest market cap in the dataset?"

# 3b. Web interface
streamlit run streamlit_app.py

# 3c. Reproducible notebook
jupyter notebook notebooks/demo.ipynb

# 4. Run the evaluation harness
python -m eval.run_eval --all
```

The **first** run scrapes 11 Wikipedia pages and embeds them into a Chroma
store under `.chroma/` (one-time, ~1 min). Later runs load that store instantly.

---

## 3. Design decisions — the *why*

This section is the heart of our submission: the reasoning behind each choice,
the alternatives we considered, and why we rejected them.

### 3.1 Why a supervisor multi-agent system (not one ReAct agent)
The course reference is a single agent with one RAG tool. We could have just
added more tools to one agent. We rejected that because:
- **Separation of concerns.** Each specialist has a focused system prompt and a
  small, relevant tool set. A single agent juggling RAG + pandas + web search
  has a larger, noisier action space and chooses tools less reliably.
- **Inspectability & evaluation.** With an explicit router we can measure
  *routing accuracy* on its own — a concrete, debuggable metric. In a monolithic
  agent the routing decision is hidden inside one opaque tool-selection step.
- **Extensibility.** Adding a new capability = adding a node, not enlarging one
  prompt. This matches the LangGraph "supervisor" pattern used in practice.

The cost is extra LLM calls (router + specialist + synthesizer). For an
interactive assistant that trade-off is worth the reliability and clarity.

### 3.2 Why these three specialists
They map directly onto the three agent archetypes the assignment names
(data-analyst, document-Q&A, web-research) and onto **three genuinely different
knowledge sources**: *static curated text* (RAG), *static structured numbers*
(dataset), and *live facts* (web). The split is not cosmetic — it dictates which
source can answer "what is a P/E ratio" vs. "top company by revenue" vs.
"today's price".

### 3.3 Why RAG with these parameters
- **Chroma + Google embeddings**, `chunk_size=1024`, `overlap=128`, `k=4` — we
  deliberately kept the reference agent's retrieval settings so our results are
  comparable to the course baseline, then measured them (Section 4).
- **Corpus = Wikipedia company + concept pages.** Real, citable, license-clean
  text. We persist embeddings to disk so the corpus is embedded once.
- *Alternative considered:* a financial-filings (10-K) corpus. Rejected for this
  timeframe: heavier parsing, and Wikipedia gives broad coverage of both
  companies and concepts in one consistent format.

### 3.4 Why focused analyst tools instead of a code interpreter
We gave the analyst five typed tools (`top_companies`, `aggregate`, …) rather
than a "run arbitrary Python" tool. Trade-off:
- **+** Reliable, safe, and — crucially — it gives us an **objective ground
  truth**: every analyst question has one correct answer computable from the
  CSV, so we can measure correctness exactly (Section 4.3).
- **−** Less flexible than free-form code. Acceptable, because the dataset and
  question types are bounded.

### 3.5 Why DuckDuckGo for web search
Keyless and free → the project is **fully reproducible** for the grader with a
single API key (Gemini). Paid search APIs (Tavily, SerpAPI) would add a second
key and a cost barrier with no pedagogical benefit.

### 3.6 Why Google Gemini Flash, temperature 0
Same model family as the course reference (comparability), generous free tier,
and fast. `temperature=0` makes routing and answers **reproducible**, which the
evaluation harness depends on.

### 3.7 Loop safety
The supervisor can chain specialists, so we cap it at `MAX_SUPERVISOR_STEPS=6`
and tell the router never to call the same specialist twice for one sub-question
— a confused router can never loop forever.

---

## 4. Evaluation methodology

We evaluate four things, each targeting a distinct failure mode. Run with
`python -m eval.run_eval --all`; the labeled set is `eval/eval_dataset.json`
(21 queries, balanced across the three agents).

### 4.1 Routing accuracy (quantitative)
Does the supervisor pick the right specialist? Each query is hand-labeled with
its expected agent; we compare the router's first decision. Reported as overall
accuracy + per-class accuracy + a list of misroutes.

### 4.2 RAG retrieval — Hit-rate@k and MRR (quantitative)
For Document-Q&A queries we know the **gold Wikipedia page** that should answer
them. We retrieve the top-`k` chunks and check whether the gold source appears
(Hit-rate@k) and how high (Mean Reciprocal Rank). This isolates *retrieval*
quality from *generation* quality.

### 4.3 Data-Analyst correctness (quantitative, objective)
The strongest metric: each analyst question has a **deterministic ground truth**
computed directly from `companies.csv` (e.g. highest market cap → Apple). We run
the analyst agent and check exact-match. No LLM judging needed — it is provably
right or wrong.

### 4.4 Answer quality — LLM-as-judge (qualitative→quantitative)
For the final synthesized answers we use Gemini as a judge scoring
**faithfulness** (is it grounded, no fabrication?) and **relevance** (does it
answer the question?) on a 1–5 scale. We treat this as a soft signal and read
the judge's comments, not just the numbers.

### 4.5 Results

> Reproduce with `python -m eval.run_eval --all`; results are written to
> `eval/results/metrics.json`. **The numbers below are filled in from our own
> run** (see that file for the exact JSON and the per-query breakdown).

| Metric                      | Value        |
|-----------------------------|--------------|
| Routing accuracy            | _see metrics.json_ |
| RAG Hit-rate@4              | _see metrics.json_ |
| RAG MRR                     | _see metrics.json_ |
| Analyst correctness         | _see metrics.json_ |
| Avg. faithfulness (1–5)     | _see metrics.json_ |
| Avg. relevance (1–5)        | _see metrics.json_ |

*(We report the actual figures in the slide deck and `eval/results/metrics.json`.
This table is intentionally tied to a re-runnable command rather than
hand-typed, so the grader can verify it.)*

---

## 5. Project structure

```
multi-agent-finance-research/
├── src/
│   ├── config.py          # models, paths, corpus list, routing constants
│   ├── knowledge_base.py  # async Wikipedia scraper + Chroma store (RAG)
│   ├── tools.py           # RAG / pandas / web-search tools
│   ├── agents.py          # supervisor + specialists + synthesizer graph
│   └── app_cli.py         # command-line entry point
├── data/
│   ├── companies.csv      # illustrative 25-company dataset
│   └── DATASET.md         # dataset card + honesty disclaimer
├── eval/
│   ├── eval_dataset.json  # 21 labeled queries (gold sources / answers)
│   ├── run_eval.py        # the four-metric evaluation harness
│   └── results/           # metrics.json output
├── notebooks/demo.ipynb   # reproducible end-to-end walkthrough
├── streamlit_app.py       # web interface with live routing trace
├── docs/                  # presentation outline + slides source
├── requirements.txt
└── .env.example
```

---

## 6. Limitations & honesty

- **The dataset is illustrative.** `companies.csv` holds approximate, rounded
  figures for demonstration — *not investment advice* (see `data/DATASET.md`).
  Current/precise numbers are intentionally delegated to the web-research agent.
- **Routing is LLM-based** and occasionally ambiguous (e.g. "who founded Apple"
  could be RAG *or* web). Our eval surfaces exactly these edge cases.
- **Web results are noisy** and time-dependent, so the web agent's answers are
  not deterministic and aren't part of the objective metrics.

---

## 7. AI usage disclosure

Per the course AI policy, here is where AI tools assisted us:

- **Claude (Anthropic)** — pair-programming assistant. Used for: scaffolding the
  LangGraph boilerplate (state, nodes, conditional edges), drafting docstrings
  and this README, and shaping the evaluation harness structure.
- **The course reference agent** (`agent.py`) — we adapted its async
  scraper and Chroma RAG pattern; lineage is noted in `knowledge_base.py`.

**What is our own intellectual work** (and what the policy asks us to own): the
problem framing (combining three agent archetypes under a supervisor), the
architecture and routing design, the choice and design of the four evaluation
metrics, the labeled evaluation set, the dataset compilation, and the
interpretation of results. Every design decision in Section 3 reflects choices
we debated as a group, with alternatives we explicitly rejected.

We are responsible for everything in this repository, including any errors.
