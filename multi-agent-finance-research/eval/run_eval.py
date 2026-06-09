# =============================================================================
# EVALUATION HARNESS — quantitative methodology for the multi-agent system
# =============================================================================
# Four complementary metrics, each targeting a different failure mode:
#
#   1. ROUTING ACCURACY   — does the supervisor pick the right specialist?
#                           (classification accuracy + per-class breakdown)
#   2. RAG RETRIEVAL       — does the Document-Q&A agent retrieve the gold
#                           source? (Hit-rate@k and Mean Reciprocal Rank)
#   3. ANALYST CORRECTNESS — does the Data-Analyst agent return the answer that
#                           is objectively correct given the dataset?
#                           (exact-match against deterministic ground truth)
#   4. ANSWER QUALITY      — LLM-as-judge faithfulness & relevance (1-5) of the
#                           final synthesized answers on a sample.
#
# Usage:
#   python -m eval.run_eval --routing      # just routing
#   python -m eval.run_eval --rag          # just retrieval
#   python -m eval.run_eval --analyst      # just analyst correctness
#   python -m eval.run_eval --judge        # just LLM-as-judge (slower)
#   python -m eval.run_eval --all          # everything (default)
#
# Results are printed and written to eval/results/metrics.json.
# =============================================================================

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_PATH = EVAL_DIR / "results" / "metrics.json"


def load_items() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)["items"]


# -----------------------------------------------------------------------------
# 1. ROUTING ACCURACY
# -----------------------------------------------------------------------------
def eval_routing(items: list[dict]) -> dict:
    from langchain_core.messages import HumanMessage

    from src import agents

    print("\n=== 1. ROUTING ACCURACY ===")
    correct = 0
    per_class = defaultdict(lambda: {"correct": 0, "total": 0})
    confusion = []

    for item in items:
        expected = item["expected_agent"]
        state = {"messages": [HumanMessage(content=item["query"])], "next": "", "steps": 0}
        try:
            decision = agents.supervisor_node(state)["next"]
        except Exception as e:
            print(f"  [ERR ] {item['query'][:50]} -> {type(e).__name__}")
            confusion.append({"query": item["query"], "expected": expected, "got": f"ERROR:{type(e).__name__}"})
            per_class[expected]["total"] += 1
            continue
        hit = decision == expected
        correct += hit
        per_class[expected]["total"] += 1
        per_class[expected]["correct"] += int(hit)
        if not hit:
            confusion.append({"query": item["query"], "expected": expected, "got": decision})
        print(f"  [{'OK ' if hit else 'MISS'}] {expected:13s} <- got {decision:13s} | {item['query'][:55]}")

    acc = correct / len(items) if items else 0.0
    print(f"  Routing accuracy: {correct}/{len(items)} = {acc:.1%}")
    return {
        "accuracy": round(acc, 4),
        "n": len(items),
        "per_class": {k: round(v["correct"] / v["total"], 4) for k, v in per_class.items()},
        "misroutes": confusion,
    }


# -----------------------------------------------------------------------------
# 2. RAG RETRIEVAL (Hit-rate@k, MRR)
# -----------------------------------------------------------------------------
def eval_rag(items: list[dict]) -> dict:
    from src.knowledge_base import get_retriever

    print("\n=== 2. RAG RETRIEVAL (Hit-rate@k & MRR) ===")
    retriever = get_retriever()
    doc_items = [i for i in items if "gold_source" in i]

    hits = 0
    reciprocal_ranks = []
    for item in doc_items:
        gold = item["gold_source"]
        docs = retriever.invoke(item["query"])
        titles = [d.metadata.get("title", "") for d in docs]
        rank = next((r for r, t in enumerate(titles, 1) if t == gold), None)
        if rank:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
            status = f"rank {rank}"
        else:
            reciprocal_ranks.append(0.0)
            status = "MISS"
        print(f"  [{status:7s}] gold='{gold}' | {item['query'][:45]}")

    n = len(doc_items)
    hit_rate = hits / n if n else 0.0
    mrr = sum(reciprocal_ranks) / n if n else 0.0
    print(f"  Hit-rate@{len(titles) if doc_items else '?'}: {hit_rate:.1%} | MRR: {mrr:.3f}")
    return {"hit_rate": round(hit_rate, 4), "mrr": round(mrr, 4), "n": n}


# -----------------------------------------------------------------------------
# 3. DATA-ANALYST CORRECTNESS (exact match vs. deterministic ground truth)
# -----------------------------------------------------------------------------
def eval_analyst(items: list[dict]) -> dict:
    from langchain_core.messages import HumanMessage

    from src import agents

    print("\n=== 3. DATA-ANALYST CORRECTNESS ===")
    ana_items = [i for i in items if "ground_truth" in i]
    correct = 0
    misses = []
    for item in ana_items:
        gt = str(item["ground_truth"]).lower()
        state = {"messages": [HumanMessage(content=item["query"])], "next": "", "steps": 0}
        try:
            report = agents.data_analyst_node(state)["messages"][-1].content
        except Exception as e:
            print(f"  [ERR ] {item['query'][:50]} -> {type(e).__name__}")
            misses.append({"query": item["query"], "expected": item["ground_truth"], "error": type(e).__name__})
            continue
        hit = gt in report.lower()
        correct += hit
        if not hit:
            misses.append({"query": item["query"], "expected": item["ground_truth"]})
        print(f"  [{'OK ' if hit else 'MISS'}] expect '{item['ground_truth']}' | {item['query'][:50]}")

    n = len(ana_items)
    acc = correct / n if n else 0.0
    print(f"  Analyst correctness: {correct}/{n} = {acc:.1%}")
    return {"accuracy": round(acc, 4), "n": n, "misses": misses}


# -----------------------------------------------------------------------------
# 4. ANSWER QUALITY (LLM-as-judge: faithfulness & relevance, 1-5)
# -----------------------------------------------------------------------------
def eval_judge(items: list[dict], sample: int = 6) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from pydantic import BaseModel, Field

    from src import agents, config

    print(f"\n=== 4. ANSWER QUALITY — LLM-as-judge (sample={sample}) ===")

    class Judgement(BaseModel):
        faithfulness: int = Field(description="1-5: is the answer supported by evidence, no fabrication?")
        relevance: int = Field(description="1-5: does the answer actually address the question?")
        comment: str = Field(description="One short sentence of justification.")

    judge = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL, temperature=0
    ).with_structured_output(Judgement)

    # Evaluate a spread across the three agent types.
    chosen = items[:sample] if sample else items
    faiths, rels, rows = [], [], []
    for item in chosen:
        answer = agents.ask(item["query"])
        prompt = (
            "You are a strict evaluator. Rate the ASSISTANT ANSWER to the USER "
            "QUESTION on faithfulness and relevance (integers 1-5).\n\n"
            f"USER QUESTION: {item['query']}\n\nASSISTANT ANSWER: {answer}"
        )
        j = judge.invoke(prompt)
        faiths.append(j.faithfulness)
        rels.append(j.relevance)
        rows.append({"query": item["query"], "faithfulness": j.faithfulness, "relevance": j.relevance, "comment": j.comment})
        print(f"  faith={j.faithfulness} rel={j.relevance} | {item['query'][:45]}")

    avg_f = sum(faiths) / len(faiths) if faiths else 0.0
    avg_r = sum(rels) / len(rels) if rels else 0.0
    print(f"  Avg faithfulness: {avg_f:.2f}/5 | Avg relevance: {avg_r:.2f}/5")
    return {"avg_faithfulness": round(avg_f, 3), "avg_relevance": round(avg_r, 3), "n": len(rows), "rows": rows}


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent evaluation.")
    parser.add_argument("--routing", action="store_true")
    parser.add_argument("--rag", action="store_true")
    parser.add_argument("--analyst", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    # Default to --all if no specific flag given.
    run_all = args.all or not (args.routing or args.rag or args.analyst or args.judge)

    items = load_items()
    results = {"timestamp": datetime.now(timezone.utc).isoformat(), "n_items": len(items)}

    if run_all or args.routing:
        results["routing"] = eval_routing(items)
    if run_all or args.rag:
        results["rag_retrieval"] = eval_rag(items)
    if run_all or args.analyst:
        results["analyst_correctness"] = eval_analyst(items)
    if run_all or args.judge:
        results["answer_quality"] = eval_judge(items)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    sys.exit(main())
