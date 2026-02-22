#!/usr/bin/env python3
"""Run benchmark — 10 questions x 4 modes = 40 evaluations.

Evaluates all Cognee search modes (GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS,
SUMMARIES) against a 10-question benchmark set (5 EN + 5 RU).

Evaluation: keyword overlap judge with cross-language concept map (no external
API needed).
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

# Load .env BEFORE importing config (Settings reads env at import time)
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("benchmark")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from cog_rag_cognee.cognee_setup import apply_cognee_env  # noqa: E402
from cog_rag_cognee.config import get_settings  # noqa: E402
from cog_rag_cognee.service import PipelineService  # noqa: E402

# ---------------------------------------------------------------------------
# Cross-language concept map (RU ↔ EN)
# ---------------------------------------------------------------------------

CONCEPT_MAP: dict[str, str] = {
    "знаний": "knowledge",
    "память": "memory",
    "документы": "documents",
    "графовая": "graph",
    "граф": "graph",
    "встроенный": "embedded",
    "сущност": "entities",
    "хранения": "storage",
    "модел": "model",
    "поиск": "search",
    "вектор": "vector",
}

# ---------------------------------------------------------------------------
# Keyword overlap judge
# ---------------------------------------------------------------------------

MIN_ANSWER_LEN = 20
NO_KEYWORDS_MIN_LEN = 50
OVERLAP_THRESHOLD = 0.3


def evaluate_answer(question: dict, answer: str) -> bool:
    """Evaluate answer using keyword overlap with cross-language matching.

    Args:
        question: dict with ``expected_keywords`` (list[str]) and ``question``.
        answer: the answer text produced by the pipeline.

    Returns:
        True if the answer passes the keyword overlap check.
    """
    if not answer or len(answer) < MIN_ANSWER_LEN:
        return False

    keywords = question.get("expected_keywords", [])
    if not keywords:
        return len(answer) > NO_KEYWORDS_MIN_LEN

    answer_lower = answer.lower()

    matched = 0
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in answer_lower:
            matched += 1
            continue
        # Cross-language: RU keyword → EN equivalent
        for ru, en in CONCEPT_MAP.items():
            if ru in kw_lower and en in answer_lower:
                matched += 1
                break
            if en == kw_lower and ru in answer_lower:
                matched += 1
                break

    overlap = matched / len(keywords)
    return overlap >= OVERLAP_THRESHOLD


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODES = ["GRAPH_COMPLETION", "RAG_COMPLETION", "CHUNKS", "SUMMARIES"]
BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark")


async def run_benchmark() -> None:
    """Run the full benchmark suite."""
    settings = get_settings()
    apply_cognee_env(settings)

    svc = PipelineService()

    # Load questions
    questions_path = os.path.join(BENCH_DIR, "questions.json")
    with open(questions_path) as f:
        questions = json.load(f)

    en_count = sum(1 for q in questions if q.get("lang") == "en")
    ru_count = sum(1 for q in questions if q.get("lang") == "ru")
    total_evals = len(questions) * len(MODES)
    print(f"\nLoaded {len(questions)} questions (EN: {en_count}, RU: {ru_count})")
    print(f"Modes: {len(MODES)} → {total_evals} evaluations\n")

    results: dict[str, list[dict]] = {}

    for mode in MODES:
        mode_results: list[dict] = []
        print(f"{'─' * 60}")
        print(f"  Mode: {mode}")
        print(f"{'─' * 60}")

        for i, q in enumerate(questions):
            question_text = q["question"]

            t0 = time.time()
            try:
                qa = await svc.query(
                    question_text, search_type=mode, limit=5
                )
                answer = qa.answer
                confidence = qa.confidence
            except Exception as e:
                answer = f"ERROR: {e}"
                confidence = 0.0
            latency = time.time() - t0

            passed = evaluate_answer(q, answer)
            mark = "PASS" if passed else "FAIL"
            lang = q.get("lang", "?")
            cat = q.get("category", "?")
            print(
                f"  Q{i + 1:2d} [{lang}/{cat:8s}] {mark}  "
                f"({latency:.1f}s)  {answer[:80]}..."
            )

            mode_results.append({
                "question": question_text,
                "lang": lang,
                "category": cat,
                "answer": answer,
                "confidence": confidence,
                "latency": round(latency, 2),
                "passed": passed,
            })

        results[mode] = mode_results
        passed_count = sum(1 for r in mode_results if r["passed"])
        total = len(mode_results)
        pct = 100 * passed_count // total if total else 0
        print(f"\n  {mode}: {passed_count}/{total} ({pct}%)\n")

    _print_summary(settings, questions, results)


def _print_summary(
    settings, questions: list, results: dict[str, list[dict]]
) -> None:
    """Print benchmark summary table and save results."""
    total_q = len(questions)
    print(f"\n{'=' * 70}")
    print(
        f"BENCHMARK — {total_q} questions x {len(MODES)} modes "
        f"= {total_q * len(MODES)} evaluations"
    )
    print(f"LLM: {settings.llm_model}  |  Embeddings: {settings.embedding_model}")
    print(f"{'=' * 70}")

    header = f"  {'Mode':20s} {'Score':>8s} {'EN':>6s} {'RU':>6s} {'Avg Lat':>10s}"
    print(header)
    print(f"  {'─' * 54}")

    for mode in MODES:
        mr = results[mode]
        passed = sum(1 for r in mr if r["passed"])
        total = len(mr)
        en_pass = sum(
            1 for r in mr if r.get("lang") == "en" and r["passed"]
        )
        ru_pass = sum(
            1 for r in mr if r.get("lang") == "ru" and r["passed"]
        )
        en_total = sum(1 for r in mr if r.get("lang") == "en")
        ru_total = sum(1 for r in mr if r.get("lang") == "ru")
        avg_lat = sum(r["latency"] for r in mr) / total if total else 0
        pct = 100 * passed // total if total else 0

        print(
            f"  {mode:20s} {passed}/{total} ({pct:2d}%) "
            f"{en_pass}/{en_total:>2d}   {ru_pass}/{ru_total:>2d}   "
            f"{avg_lat:>7.1f}s"
        )

    total_passed = sum(
        sum(1 for r in mr if r["passed"]) for mr in results.values()
    )
    total_all = sum(len(mr) for mr in results.values())
    overall_pct = 100 * total_passed // total_all if total_all else 0
    print(f"  {'─' * 54}")
    print(f"  {'OVERALL':20s} {total_passed}/{total_all} ({overall_pct}%)")

    # Save results.json
    out_path = os.path.join(BENCH_DIR, "results.json")
    out_data = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "modes": MODES,
        "total_passed": total_passed,
        "total_all": total_all,
        "accuracy": round(total_passed / total_all, 4) if total_all else 0,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
