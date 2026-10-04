"""Offline evaluation helpers for answer and evidence integrity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from src.contracts import parse_tool_result

EVIDENCE_ID_PATTERN = re.compile(r"\[(?:KB|CALC|WEB|ATTACH)-[A-Z0-9]+\]")


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_status: str = "completed"
    expected_evidence_ids: tuple[str, ...] = ()
    mode: str = "Deep"


def extract_evidence_ids(answer: str) -> tuple[str, ...]:
    """Return unique structured evidence IDs in first-seen order."""

    return tuple(dict.fromkeys(EVIDENCE_ID_PATTERN.findall(answer or "")))


def available_evidence_ids(messages: Iterable[Any]) -> tuple[str, ...]:
    """Collect evidence IDs from successful structured tool messages."""

    found: list[str] = []
    for message in messages:
        if getattr(message, "type", None) != "tool":
            continue
        result = parse_tool_result(str(message.content))
        if result and result.ok:
            found.extend(item.evidence_id for item in result.evidence)
    return tuple(dict.fromkeys(found))


def evaluate_state(state: dict[str, Any], case: EvaluationCase) -> dict[str, Any]:
    """Evaluate graph invariants without judging open-ended language quality."""

    messages = state.get("messages", [])
    answer = ""
    for message in reversed(messages):
        if getattr(message, "type", None) == "ai" and not getattr(message, "tool_calls", []):
            answer = str(message.content)
            break
    cited = extract_evidence_ids(answer)
    available = available_evidence_ids(messages)
    available_tokens = {f"[{evidence_id}]" for evidence_id in available}
    invalid = tuple(evidence_id for evidence_id in cited if evidence_id not in available_tokens)
    missing = tuple(
        evidence_id
        for evidence_id in case.expected_evidence_ids
        if evidence_id not in cited
    )
    failures: list[str] = []
    if state.get("run_status") != case.expected_status:
        failures.append(f"expected status {case.expected_status}, got {state.get('run_status')}")
    if invalid:
        failures.append(f"invalid evidence IDs: {', '.join(invalid)}")
    if missing:
        failures.append(f"missing evidence IDs: {', '.join(missing)}")
    return {
        "case_id": case.case_id,
        "question": case.question,
        "mode": case.mode,
        "status": state.get("run_status", "missing"),
        "answer": answer,
        "cited_evidence_ids": list(cited),
        "available_evidence_ids": list(available),
        "failures": failures,
        "passed": not failures,
    }


def render_evaluation_markdown(results: Iterable[dict[str, Any]]) -> str:
    """Render compact, reviewable evaluation output."""

    rows = list(results)
    passed = sum(1 for row in rows if row["passed"])
    lines = [
        "# Offline evaluation results",
        "",
        f"**Passed:** {passed}/{len(rows)}",
        "",
        "| Case | Mode | Status | Evidence | Result |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        evidence = ", ".join(row["cited_evidence_ids"]) or "none"
        result = "PASS" if row["passed"] else "FAIL"
        lines.append(
            f"| {row['case_id']} | {row['mode']} | {row['status']} | {evidence} | {result} |"
        )
        for failure in row["failures"]:
            lines.append(f"|  |  |  |  | {failure} |")
    return "\n".join(lines) + "\n"
