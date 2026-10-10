"""Reproducible Milestone 2 benchmark runner.

Offline is deliberately the default and never constructs the production agent.
Use ``--live`` only when network-backed evaluation is explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

from src.evaluation import (
    aggregate_results, evaluate_state, load_benchmark, render_evaluation_markdown,
)

ROOT = Path(__file__).parent
BENCHMARK = ROOT / "benchmark" / "milestone2_v1.json"
THRESHOLDS = ROOT / "benchmark" / "thresholds.json"


def _fixture(case):
    """Build a deterministic state; this function performs no I/O or network calls."""
    evidence = []
    if case.fixture not in {"missing_evidence", "tool_failure", "malformed_tool"} and case.expected_evidence_ids:
        for evidence_id in case.expected_evidence_ids:
            evidence.append({
                "evidence_id": evidence_id.strip("[]"), "source_type": "attachment",
                "source_id": f"fixture:{case.case_id}", "excerpt": "deterministic fixture",
            })
    answer = f"Deterministic answer for {case.case_id}."
    if case.fixture == "abstain":
        answer = "Insufficient evidence to answer this question reliably."
    elif case.fixture == "invalid_citation":
        answer += " [WEB-999]"
    elif case.fixture == "conflict":
        answer += " Sources disagree; the available evidence is inconclusive."
    if case.expected_evidence_ids and case.fixture not in {"abstain", "invalid_citation", "missing_evidence", "tool_failure"}:
        answer += " " + " ".join(case.expected_evidence_ids)
    messages = []
    if evidence:
        messages.append(SimpleNamespace(type="tool", content=json.dumps({
            "ok": True, "tool_name": "offline_fixture", "evidence": evidence,
        })))
    messages.append(SimpleNamespace(type="ai", tool_calls=[], content=answer))
    status = case.expected_status
    if case.fixture in {"tool_failure", "malformed_tool"}:
        messages.append(SimpleNamespace(type="tool", content=json.dumps({
            "ok": False, "tool_name": "offline_fixture", "error_type": "timeout",
            "error_message": "deterministic fixture failure",
        })))
    return {
        "run_status": status, "messages": messages,
        "recovered": case.expected_recovery, "recovery_attempts": 1 if case.expected_recovery else 0,
        "latency_ms": 0, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "cost": 0.0,
    }


def run_offline_evaluation(selected_cases=None) -> int:
    cases = load_benchmark(BENCHMARK)
    if selected_cases:
        cases = tuple(case for case in cases if case.category in selected_cases)
    if not cases:
        raise ValueError("no benchmark cases selected")
    if len(cases) != 50 and not selected_cases:
        raise ValueError(f"expected 50 benchmark cases, found {len(cases)}")
    results = [evaluate_state(_fixture(case), case) for case in cases]
    thresholds = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    summary = aggregate_results(results, thresholds)
    report = {"benchmark": "milestone2_v1", "mode": "offline", "deterministic": True,
              "external_calls": False,
              "metric_definitions": {
                  "expected_behavior_pass_rate": "expected-behavior checks passed / attempted cases; includes correctly handled failures and abstentions.",
                  "offline_fixture_task_success_rate": "successful controlled offline fixture cases / cases whose expected outcome is success; excludes failure and abstention fixtures and is not real-world research accuracy.",
                  "evidence_validity": "valid cited evidence IDs / cited evidence IDs; current-run structured evidence only.",
                  "evidence_coverage": "applicable required evidence references satisfied / applicable required references; cases without requirements excluded.",
                  "unsupported_claim_rate": "unsupported fixture-labeled claims / claims independently labeled in offline fixtures; not semantic accuracy.",
                  "recovery": "eligible recoverable-failure cases with observed recovery and expected outcome / eligible recoverable-failure cases.",
              },
              "limitations": [
                  "Offline fixtures validate contracts and recovery bookkeeping, not model quality.",
                  "Claim support labels do not establish factual truth.",
                  "Live latency, usage, and provider cost are not measured offline.",
              ],
              "summary": summary, "results": results}
    (ROOT / "evaluation-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_evaluation_markdown(results)
    markdown += (
        "\n## Metric definitions\n\n"
        "- **Expected-behavior pass rate:** correctly handled cases / attempted cases; includes expected failures and abstentions.\n"
        "- **Offline fixture task-success rate:** successful controlled fixture cases / cases whose expected outcome is success; excludes failure and abstention fixtures and is not real-world research accuracy.\n"
        "- **Evidence validity:** valid cited evidence references / cited references from current-run structured evidence.\n"
        "- **Evidence coverage:** satisfied applicable required evidence references / applicable required references; non-research and expected-failure cases are excluded.\n"
        "- **Unsupported-claim rate:** unsupported fixture-labeled claims / independently labeled claims; offline labels are contract checks, not semantic accuracy.\n"
        "- **Recovery rate:** eligible recoverable failures with observed recovery and expected outcome / eligible recoverable failures.\n\n"
        "Offline fixtures do not measure real-world model accuracy, source quality, or semantic claim entailment.\n\n"
        "## Aggregate metrics\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    )
    (ROOT / "EVALUATION.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if summary["regression_passed"] else 1


def _check_live_configuration() -> list[str]:
    required = ("GROQ_API_KEY", "TAVILY_API_KEY", "GEMINI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")
    return [name for name in required if not os.getenv(name)]


def run_live_evaluation(limit=None, categories=None) -> int:
    missing = _check_live_configuration()
    if missing:
        raise SystemExit("Live mode requires configuration; missing: " + ", ".join(missing))
    from src.agent import create_agent_graph
    cases = list(load_benchmark(BENCHMARK))
    if categories:
        cases = [case for case in cases if case.category in categories]
    if limit:
        cases = cases[:limit]
    graph = create_agent_graph()
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            state = graph.invoke({"messages": [{"role": "user", "content": case.question}]})
            state["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            results.append(evaluate_state(state, case))
        except Exception as error:
            results.append({
                "case_id": case.case_id, "status": "error", "outcome": "error",
                "passed": False, "task_success": False, "failures": [type(error).__name__],
            })
    summary = aggregate_results(results)
    report = {"benchmark": "milestone2_v1", "mode": "live", "external_calls": True,
              "summary": summary, "results": results,
              "limitations": ["Live results depend on configured providers and model behavior."]}
    (ROOT / "evaluation-results-live.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["regression_passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ResearchPilot's Milestone 2 benchmark.")
    parser.add_argument("--live", action="store_true", help="Opt in to network-backed evaluation.")
    parser.add_argument("--limit", type=int, default=None, help="Limit selected benchmark cases.")
    parser.add_argument("--category", action="append", help="Restrict cases by category.")
    args = parser.parse_args()
    selected = set(args.category or [])
    raise SystemExit(
        run_live_evaluation(args.limit, selected) if args.live
        else run_offline_evaluation(selected or None)
    )
