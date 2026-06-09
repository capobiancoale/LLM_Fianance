# =============================================================================
# MULTI-AGENT SYSTEM — supervisor that orchestrates three specialist agents
# =============================================================================
#
# Architecture (LangGraph):
#
#                         ┌──────────────┐
#        user query ─────▶│  SUPERVISOR  │◀───────────────┐
#                         │  (router)    │                │
#                         └──────┬───────┘                │
#            ┌───────────────────┼───────────────────┐    │ specialist
#            ▼                   ▼                   ▼    │ reports
#   ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
#   │  DOCUMENT-Q&A  │ │  DATA-ANALYST  │ │  WEB-RESEARCH  │
#   │  (RAG / Chroma)│ │  (pandas)      │ │  (DuckDuckGo)  │
#   └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
#            └──────────────────┴───────────────────┘
#                                │ (when supervisor says FINISH)
#                                ▼
#                         ┌──────────────┐
#                         │  SYNTHESIZE  │ ─▶ final answer
#                         └──────────────┘
#
# Each specialist is itself a small tool-using (ReAct-style) loop, exactly the
# pattern from the course reference agent — the supervisor just decides *which*
# specialist should act next, and can chain several before synthesising.
# =============================================================================

import operator
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from . import config, tools

# -----------------------------------------------------------------------------
# Shared LLM
# -----------------------------------------------------------------------------
# A client-side rate limiter keeps us under the Gemini free-tier per-minute cap
# during multi-call evaluation runs (it paces requests; it does NOT raise the
# per-day quota). Tune via config.LLM_REQUESTS_PER_SECOND.
config.require_api_key()
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=config.LLM_REQUESTS_PER_SECOND,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)
_base_llm = ChatGoogleGenerativeAI(
    model=config.LLM_MODEL,
    temperature=config.LLM_TEMPERATURE,
    rate_limiter=_rate_limiter,
)


# -----------------------------------------------------------------------------
# Graph state
# -----------------------------------------------------------------------------
class AgentState(TypedDict):
    """State flowing through the multi-agent graph.

    * messages — the running transcript: the user's question plus each
      specialist's report (Annotated with operator.add so updates accumulate).
    * next     — the supervisor's routing decision for the upcoming step.
    * steps    — how many specialist hops we've taken (loop safety valve).
    """

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    steps: int


# =============================================================================
# SPECIALIST AGENTS — generic bounded ReAct loop, one instance per specialist
# =============================================================================
def _run_react(system_prompt: str, task: str, tool_list, max_iters: int = 4) -> str:
    """Run a bounded tool-using loop and return the agent's final text.

    Mirrors the course reference's llm_node/tools cycle, but kept self-contained
    so each specialist is independent and easy to reason about.
    """
    llm_with_tools = _base_llm.bind_tools(tool_list)
    tools_by_name = {t.name: t for t in tool_list}

    messages: list[BaseMessage] = [
        HumanMessage(content=f"{system_prompt}\n\nUser request:\n{task}")
    ]

    for _ in range(max_iters):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        tool_calls = getattr(ai_msg, "tool_calls", None)
        if not tool_calls:
            # No more tools requested → this is the agent's final answer.
            return _text_of(ai_msg)

        # Execute every requested tool call and feed results back in.
        for call in tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                result = f"Unknown tool: {call['name']}"
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            messages.append(
                ToolMessage(
                    content=str(result), name=call["name"], tool_call_id=call["id"]
                )
            )

    # Ran out of iterations — ask the model to wrap up with what it has.
    messages.append(
        HumanMessage(content="Provide your best final answer now, no more tools.")
    )
    return _text_of(_base_llm.invoke(messages))


def _text_of(message: BaseMessage) -> str:
    """Normalise Gemini's content (which can be str or a list of parts)."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return " ".join(p for p in parts if p).strip()
    return str(content)


_DOC_QA_PROMPT = (
    "You are the DOCUMENT-Q&A specialist. Answer questions about companies and "
    "finance concepts using ONLY the curated knowledge base. Always call "
    "search_company_knowledge before answering, and ground your answer in the "
    "retrieved passages. Cite the source titles. If the knowledge base lacks the "
    "answer, say so plainly."
)

_ANALYST_PROMPT = (
    "You are the DATA-ANALYST specialist. Answer quantitative questions about the "
    "bundled companies dataset by calling the analysis tools (never guess "
    "numbers). Call dataset_overview first if unsure of the schema. Report the "
    "concrete figures you computed and state they come from the bundled dataset."
)

_WEB_PROMPT = (
    "You are the WEB-RESEARCH specialist. Answer questions needing CURRENT or "
    "real-time information by calling web_search. Summarise findings concisely "
    "and include the source URLs. Note that web results may be noisy or dated."
)


def document_qa_node(state: AgentState) -> dict:
    task = _latest_user_question(state)
    report = _run_react(_DOC_QA_PROMPT, task, tools.DOCUMENT_QA_TOOLS)
    return {"messages": [HumanMessage(content=report, name=config.AGENT_DOCUMENT_QA)]}


def data_analyst_node(state: AgentState) -> dict:
    task = _latest_user_question(state)
    report = _run_react(_ANALYST_PROMPT, task, tools.DATA_ANALYST_TOOLS)
    return {"messages": [HumanMessage(content=report, name=config.AGENT_DATA_ANALYST)]}


def web_research_node(state: AgentState) -> dict:
    task = _latest_user_question(state)
    report = _run_react(_WEB_PROMPT, task, tools.WEB_RESEARCH_TOOLS)
    return {"messages": [HumanMessage(content=report, name=config.AGENT_WEB_RESEARCH)]}


def _latest_user_question(state: AgentState) -> str:
    """The original user question is the first human message in the transcript."""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage) and not getattr(msg, "name", None):
            return _text_of(msg)
    return _text_of(state["messages"][0])


# =============================================================================
# SUPERVISOR — decides which specialist acts next, or that we are done
# =============================================================================
class Route(BaseModel):
    """Structured routing decision emitted by the supervisor."""

    next: Literal["document_qa", "data_analyst", "web_research", "FINISH"] = Field(
        description="Which specialist should act next, or FINISH if enough "
        "information has been gathered to answer the user."
    )
    reasoning: str = Field(description="One short sentence justifying the choice.")


_SUPERVISOR_PROMPT = """You are the SUPERVISOR of a finance research team. Route \
the user's request to ONE specialist at a time, then re-evaluate.

Specialists:
- document_qa  : background facts, definitions, company descriptions, history \
(from a curated Wikipedia knowledge base).
- data_analyst : quantitative questions about a bundled dataset of 25 large \
companies (revenue, market cap, employees, sectors, rankings, aggregations).
- web_research : current / real-time information (recent news, today's prices, \
latest events).

Rules:
- Pick the specialist that best advances the answer. A request may need several \
specialists in sequence (e.g. look up a definition, then analyse the dataset).
- Do NOT call the same specialist twice for the same sub-question.
- When the gathered reports are enough to fully answer the user, output FINISH.
"""

_router_llm = _base_llm.with_structured_output(Route)


def supervisor_node(state: AgentState) -> dict:
    steps = state.get("steps", 0)

    # Safety valve: never loop forever.
    if steps >= config.MAX_SUPERVISOR_STEPS:
        return {"next": config.FINISH, "steps": steps}

    question = _latest_user_question(state)

    # Summarise what specialists have already reported, so the supervisor can
    # decide whether more work is needed.
    prior = [
        f"- {getattr(m, 'name')} reported: {_text_of(m)[:400]}"
        for m in state["messages"]
        if isinstance(m, HumanMessage) and getattr(m, "name", None)
    ]
    prior_block = "\n".join(prior) if prior else "(no specialist has reported yet)"

    routing_input = (
        f"{_SUPERVISOR_PROMPT}\n\nUSER REQUEST:\n{question}\n\n"
        f"SPECIALIST REPORTS SO FAR:\n{prior_block}\n\n"
        "Decide the next action."
    )

    try:
        decision: Route = _router_llm.invoke(routing_input)
        nxt = decision.next
    except Exception:
        # Robust fallback: if structured output fails, default sensibly.
        nxt = config.AGENT_DOCUMENT_QA if not prior else config.FINISH

    return {"next": nxt, "steps": steps + 1}


def _route(state: AgentState) -> str:
    """Conditional-edge function: map the supervisor's decision to a node."""
    nxt = state.get("next", config.FINISH)
    if nxt in config.SPECIALISTS:
        return nxt
    return "synthesize"


# =============================================================================
# SYNTHESIZER — composes the final, user-facing answer from all reports
# =============================================================================
_SYNTH_PROMPT = (
    "You are the lead analyst. Using the specialist reports below, write a "
    "clear, well-structured final answer to the user's question. Integrate the "
    "findings, keep numbers accurate, and briefly attribute facts to their "
    "source (knowledge base / dataset / web). If reports conflict or are "
    "incomplete, say so."
)


def synthesize_node(state: AgentState) -> dict:
    question = _latest_user_question(state)
    reports = [
        f"### {getattr(m, 'name')}\n{_text_of(m)}"
        for m in state["messages"]
        if isinstance(m, HumanMessage) and getattr(m, "name", None)
    ]
    reports_block = "\n\n".join(reports) if reports else "(no specialist reports)"

    prompt = (
        f"{_SYNTH_PROMPT}\n\nUSER QUESTION:\n{question}\n\n"
        f"SPECIALIST REPORTS:\n{reports_block}"
    )
    final = _base_llm.invoke(prompt)
    return {"messages": [AIMessage(content=_text_of(final))]}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================
def build_graph():
    """Wire the supervisor, specialists and synthesizer into a runnable graph."""
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node(config.AGENT_DOCUMENT_QA, document_qa_node)
    builder.add_node(config.AGENT_DATA_ANALYST, data_analyst_node)
    builder.add_node(config.AGENT_WEB_RESEARCH, web_research_node)
    builder.add_node("synthesize", synthesize_node)

    builder.set_entry_point("supervisor")

    # Supervisor routes to a specialist or to the synthesizer.
    builder.add_conditional_edges(
        "supervisor",
        _route,
        {
            config.AGENT_DOCUMENT_QA: config.AGENT_DOCUMENT_QA,
            config.AGENT_DATA_ANALYST: config.AGENT_DATA_ANALYST,
            config.AGENT_WEB_RESEARCH: config.AGENT_WEB_RESEARCH,
            "synthesize": "synthesize",
        },
    )

    # Every specialist hands control back to the supervisor.
    for specialist in config.SPECIALISTS:
        builder.add_edge(specialist, "supervisor")

    builder.add_edge("synthesize", END)
    return builder.compile()


# Singleton compiled graph.
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str, verbose: bool = False) -> str:
    """Convenience entry point: run the full multi-agent system on one question
    and return the final synthesized answer."""
    graph = get_graph()
    initial: AgentState = {
        "messages": [HumanMessage(content=question)],
        "next": "",
        "steps": 0,
    }
    if verbose:
        answer = "(no answer produced)"
        for event in graph.stream(initial):
            for node, update in event.items():
                if node == "supervisor":
                    print(f"  [supervisor] → {update.get('next')}")
                elif node in config.SPECIALISTS:
                    print(f"  [{node}] produced a report")
                elif node == "synthesize":
                    answer = _text_of(update["messages"][-1])
        return answer

    result = graph.invoke(initial)
    return _text_of(result["messages"][-1])
