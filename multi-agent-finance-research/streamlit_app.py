# =============================================================================
# STREAMLIT WEB INTERFACE for the Multi-Agent Finance Research Assistant
# =============================================================================
# Run with:
#   streamlit run streamlit_app.py
# Shows, for each question, which specialists the supervisor invoked (the
# routing trace) and the final synthesized answer.
# =============================================================================

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Finance Research Assistant", page_icon="📊", layout="wide")

st.title("📊 Multi-Agent Company & Finance Research Assistant")
st.caption(
    "A LangGraph **supervisor** routes your question to specialist agents — "
    "**Document-Q&A** (RAG), **Data-Analyst** (dataset), **Web-Research** (live) — "
    "then synthesizes a final answer."
)

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **document_qa** — definitions & company background (Wikipedia RAG)\n"
        "- **data_analyst** — numbers from the bundled 25-company dataset\n"
        "- **web_research** — current / real-time info (DuckDuckGo)\n"
    )
    st.markdown("**Try:**")
    st.markdown(
        "- *What is a price-earnings ratio?*\n"
        "- *Which company has the highest market cap in the dataset?*\n"
        "- *What are the latest news about Nvidia?*\n"
        "- *Define market cap, then tell me the top 3 companies by it.*"
    )
    st.divider()
    st.caption("Dataset figures are illustrative — not investment advice.")


@st.cache_resource(show_spinner="Initializing agents & knowledge base (first run embeds the corpus)...")
def _get_graph():
    from src import config
    from src.agents import get_graph

    config.require_api_key()
    return get_graph()


def _text_of(message) -> str:
    content = message.content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content).strip()
    return str(content)


question = st.text_input("Ask a finance question:", placeholder="e.g. Which sector has the highest total revenue?")

if st.button("Ask", type="primary") and question.strip():
    try:
        graph = _get_graph()
    except Exception as e:
        st.error(f"Setup error: {e}")
        st.stop()

    from src import config

    initial = {"messages": [HumanMessage(content=question)], "next": "", "steps": 0}

    trace_box = st.container()
    answer = "(no answer)"
    with st.status("Running multi-agent system...", expanded=True) as status:
        for event in graph.stream(initial):
            for node, update in event.items():
                if node == "supervisor":
                    st.write(f"🧭 **Supervisor** → `{update.get('next')}`")
                elif node in config.SPECIALISTS:
                    report = _text_of(update["messages"][-1])
                    with st.expander(f"🔧 {node} report", expanded=False):
                        st.write(report)
                elif node == "synthesize":
                    answer = _text_of(update["messages"][-1])
        status.update(label="Done", state="complete", expanded=False)

    st.subheader("Answer")
    st.markdown(answer)
