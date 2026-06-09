# Presentation outline — Multi-Agent Company & Finance Research Assistant

> Source for the PowerPoint deck. Each `##` is one slide; bullets are slide
> content, *Notes:* are speaker notes (the "why" the policy asks us to defend).

---

## Slide 1 — Title
- **Multi-Agent Company & Finance Research Assistant**
- LLM and Generative AI — BBS-AIIM final project
- Alessio Capobianco · Elena La Chiusa · Francesco Pirovano
- Track: *AI agent with tool use* (LangGraph + Google Gemini)

*Notes:* We chose the agent track and combined all three suggested archetypes
(data-analyst, document-Q&A, web-research) under one supervisor.

## Slide 2 — Problem framing
- A finance question can need very different knowledge:
  - a **definition** ("what is a P/E ratio?")
  - a **computed fact** ("top company by revenue")
  - a **live fact** ("today's Nvidia news")
- No single source/tool serves all three well.
- **Goal:** one assistant that sends each question to the right specialist.

*Notes:* This framing is the core design insight and it directly motivates a
multi-agent split rather than a monolithic agent.

## Slide 3 — From the course reference to our system
- Reference (`module5`): *one* ReAct agent, *one* RAG tool, CLI.
- Ours: a **supervisor** routing to **three specialists** + a **synthesizer**.
- Reused: async Wikipedia scraper + Chroma RAG pattern (lineage documented).

*Notes:* We deliberately kept the reference's retrieval settings so our numbers
are comparable to the baseline.

## Slide 4 — Architecture (diagram)
- Supervisor (router) → {document_qa | data_analyst | web_research} → back to
  supervisor → … → synthesizer → answer.
- LangGraph `StateGraph`; conditional edges; shared message state.
- Safety valve: max 6 supervisor hops.

*Notes:* Show the multi-hop example: define market cap (RAG) → top 3 by it
(analyst) → merged answer.

## Slide 5 — The three specialists
- **document_qa** — RAG over Chroma (Wikipedia: 7 companies + 4 concepts).
- **data_analyst** — 5 typed pandas tools over a 25-company dataset.
- **web_research** — keyless DuckDuckGo live search.
- Each is a small tool-using loop with a focused prompt + tool set.

*Notes:* Three genuinely different knowledge sources: static text, static
numbers, live web.

## Slide 6 — Key design decisions (the *why*)
- **Supervisor vs. monolith:** focused prompts, measurable routing, easy to extend.
- **Typed analyst tools vs. code interpreter:** safe + gives objective ground truth.
- **Curated CSV + DuckDuckGo:** fully reproducible with one free API key.
- **Gemini Flash, temp 0:** comparable to baseline, reproducible.

*Notes:* For each, name the alternative we rejected and why (see README §3).

## Slide 7 — Evaluation methodology (qualitative + quantitative)
- 21 hand-labeled queries, balanced across agents.
- Four metrics, four failure modes:
  1. **Routing accuracy** — right specialist?
  2. **RAG Hit-rate@4 / MRR** — right document retrieved?
  3. **Analyst correctness** — exact match vs. deterministic ground truth.
  4. **LLM-as-judge** — faithfulness & relevance (1–5).

*Notes:* Metric 3 is provably objective; metric 4 is a soft signal we read with
its comments.

## Slide 8 — Results
- Routing accuracy: **<fill>**
- RAG Hit-rate@4 / MRR: **<fill>** / **<fill>**
- Analyst correctness: **<fill>**
- Avg faithfulness / relevance: **<fill>** / **<fill>**
- (Reproduce: `python -m eval.run_eval --all`)

*Notes:* Interpret, don't just report — discuss any misroutes (e.g. ambiguous
"who founded Apple") and what they reveal about LLM routing.

## Slide 9 — Interpretation & limitations
- Routing errors concentrate on **ambiguous** queries → expected, informative.
- Dataset is **illustrative** (not investment advice); live numbers delegated to
  the web agent by design.
- Web answers are non-deterministic → excluded from objective metrics.

*Notes:* Honest about what the numbers do and don't show.

## Slide 10 — Demo & deliverables
- CLI, Streamlit UI, reproducible notebook, eval harness.
- Incremental git history.
- AI-usage disclosure in README (Claude for boilerplate/docs; our own design,
  eval, and interpretation).

## Slide 11 — Conclusion
- A clean, inspectable, reproducible multi-agent pattern.
- Evaluation that separates routing, retrieval, computation, and generation.
- Easy to extend with new specialists.
