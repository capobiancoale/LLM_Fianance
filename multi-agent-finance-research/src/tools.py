# =============================================================================
# TOOLS — the concrete capabilities each specialist agent can call
# =============================================================================
# Design note: rather than giving an agent a single "run arbitrary code" tool,
# we expose a small set of *focused, validated* tools. This makes tool calls
# reliable (the LLM fills in a few typed arguments), keeps the data-analyst
# answers deterministic, and gives us an objective ground truth for evaluation.
# =============================================================================

from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool

from . import config

# =============================================================================
# DOCUMENT-Q&A AGENT TOOL — semantic search over the Wikipedia corpus
# =============================================================================


@tool
def search_company_knowledge(query: str) -> str:
    """Search the curated company & finance knowledge base (Wikipedia) for
    background information. Use for definitions, history, business descriptions
    and other reference facts — e.g. "what does Nvidia make?", "what is a P/E
    ratio?". Returns the most relevant passages with their source titles.

    Args:
        query: A natural-language question or topic to look up.
    """
    # Imported lazily so that merely importing this module never triggers the
    # (slow, API-key-requiring) vector-store build.
    from .knowledge_base import get_retriever

    docs = get_retriever().invoke(query)
    if not docs:
        return "No relevant passages found in the knowledge base."

    blocks = []
    for d in docs[: config.RETRIEVER_K]:
        title = d.metadata.get("title", "Unknown source")
        blocks.append(f"[Source: {title}]\n{d.page_content}")
    return "\n---\n".join(blocks)


# =============================================================================
# DATA-ANALYST AGENT TOOLS — pandas over the bundled companies dataset
# =============================================================================

# Numeric columns the analyst can rank / aggregate on.
_NUMERIC_COLS = {
    "employees",
    "revenue_usd_b",
    "net_income_usd_b",
    "market_cap_usd_b",
    "founded",
}


@lru_cache(maxsize=1)
def _load_df() -> pd.DataFrame:
    """Load the companies dataset once and cache it."""
    return pd.read_csv(config.COMPANIES_CSV)


@tool
def dataset_overview() -> str:
    """Describe the companies dataset: its columns, row count and a few sample
    rows. Call this first when you are unsure what data is available."""
    df = _load_df()
    cols = ", ".join(df.columns)
    sample = df.head(3).to_string(index=False)
    return (
        f"The dataset has {len(df)} companies and these columns: {cols}.\n"
        f"Numeric columns you can rank/aggregate: {sorted(_NUMERIC_COLS)}.\n"
        f"Sample rows:\n{sample}"
    )


@tool
def top_companies(
    metric: str, n: int = 5, ascending: bool = False, sector: str | None = None
) -> str:
    """Rank companies by a numeric metric and return the top N.

    Args:
        metric: One of employees, revenue_usd_b, net_income_usd_b,
            market_cap_usd_b, founded.
        n: How many companies to return (default 5).
        ascending: False = largest first (default); True = smallest first.
        sector: Optional sector filter, e.g. "Technology".
    """
    if metric not in _NUMERIC_COLS:
        return f"'{metric}' is not a numeric column. Choose from {sorted(_NUMERIC_COLS)}."
    df = _load_df()
    if sector:
        df = df[df["sector"].str.lower() == sector.lower()]
        if df.empty:
            return f"No companies found in sector '{sector}'."
    ranked = df.sort_values(metric, ascending=ascending).head(n)
    out = ranked[["company", "sector", "country", metric]]
    return out.to_string(index=False)


@tool
def aggregate(group_by: str, metric: str, agg: str = "mean") -> str:
    """Group companies and aggregate a numeric metric.

    Args:
        group_by: A categorical column: sector, country or ticker.
        metric: A numeric column (e.g. revenue_usd_b, market_cap_usd_b).
        agg: One of sum, mean, max, min, count (default mean).
    """
    df = _load_df()
    if group_by not in {"sector", "country", "ticker"}:
        return "group_by must be one of: sector, country, ticker."
    if metric not in _NUMERIC_COLS:
        return f"'{metric}' is not numeric. Choose from {sorted(_NUMERIC_COLS)}."
    if agg not in {"sum", "mean", "max", "min", "count"}:
        return "agg must be one of: sum, mean, max, min, count."
    result = df.groupby(group_by)[metric].agg(agg).sort_values(ascending=False)
    return result.round(2).to_string()


@tool
def company_lookup(name: str) -> str:
    """Return all recorded figures for a single company by (partial) name.

    Args:
        name: Full or partial company name, e.g. "Apple" or "JPMorgan".
    """
    df = _load_df()
    hits = df[df["company"].str.contains(name, case=False, na=False)]
    if hits.empty:
        return f"No company matching '{name}' in the dataset."
    return hits.to_string(index=False)


@tool
def compare_companies(names: list[str], metric: str) -> str:
    """Compare several companies on one numeric metric.

    Args:
        names: List of (partial) company names to compare.
        metric: A numeric column, e.g. market_cap_usd_b.
    """
    if metric not in _NUMERIC_COLS:
        return f"'{metric}' is not numeric. Choose from {sorted(_NUMERIC_COLS)}."
    df = _load_df()
    pattern = "|".join(names)
    hits = df[df["company"].str.contains(pattern, case=False, na=False)]
    if hits.empty:
        return f"None of {names} were found in the dataset."
    out = hits[["company", metric]].sort_values(metric, ascending=False)
    return out.to_string(index=False)


# =============================================================================
# WEB-RESEARCH AGENT TOOL — live, keyless web search via DuckDuckGo
# =============================================================================


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the live web for *current* information not in the static
    knowledge base — recent news, today's figures, latest events. Returns a
    list of result titles, snippets and URLs.

    Args:
        query: The search query.
        max_results: Number of results to return (default 5).
    """
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS  # type: ignore

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:  # network / rate-limit failures shouldn't crash the agent
        return f"Web search failed ({e}). Try rephrasing or rely on other sources."

    if not results:
        return "No web results found."

    lines = []
    for r in results:
        title = r.get("title", "")
        body = r.get("body", "")
        href = r.get("href", "")
        lines.append(f"- {title}\n  {body}\n  {href}")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Tool groupings consumed by the specialist agents (see agents.py)
# -----------------------------------------------------------------------------
DOCUMENT_QA_TOOLS = [search_company_knowledge]
DATA_ANALYST_TOOLS = [
    dataset_overview,
    top_companies,
    aggregate,
    company_lookup,
    compare_companies,
]
WEB_RESEARCH_TOOLS = [web_search]
