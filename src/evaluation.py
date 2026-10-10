"""Offline evaluation helpers for answer and evidence integrity."""

from __future__ import annotations

import re
import json
import time
from pathlib import Path
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any, Iterable

from src.contracts import parse_tool_result

EVIDENCE_ID_PATTERN = re.compile(r"(?<!\w)\[[A-Z][A-Z0-9_-]*-\d+\](?!\w)")
MARKDOWN_URL_PATTERN = re.compile(r"\[[^\]]+\]\(([^)\s]+)\)")
BARE_URL_PATTERN = re.compile(r"(?<![\w\"'=])(https?://[^\s<>\"]+)")


def normalize_citation_url(url: str) -> str | None:
    """Normalize harmless surrounding punctuation without changing URL semantics."""

    candidate = (url or "").strip().rstrip(".,;:!?")
    while candidate.endswith((")", "]", "}")):
        if candidate.count("(") >= candidate.count(")"):
            break
        candidate = candidate[:-1]
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return candidate


def extract_citation_urls(answer: str) -> tuple[str, ...]:
    """Extract URL citations from Markdown links and bare HTTP(S) URLs."""

    urls = MARKDOWN_URL_PATTERN.findall(answer or "")
    urls.extend(BARE_URL_PATTERN.findall(answer or ""))
    return tuple(dict.fromkeys(urls))


def available_evidence_urls(messages: Iterable[Any], current_run_only: bool = True) -> tuple[str, ...]:
    """Collect valid source URLs from successful current-run evidence."""

    urls: list[str] = []
    for message in _current_run_messages(messages) if current_run_only else messages:
        if getattr(message, "type", None) != "tool":
            continue
        result = parse_tool_result(str(message.content))
        if not result or not result.ok:
            continue
        for evidence in result.evidence:
            normalized = normalize_citation_url(evidence.source_id) if evidence.source_type == "web" else None
            if normalized:
                urls.append(normalized)
    return tuple(dict.fromkeys(urls))


def _current_run_messages(messages: Iterable[Any]) -> list[Any]:
    message_list = list(messages)
    starts = [
        index for index, message in enumerate(message_list)
        if getattr(message, "type", None) == "human"
        and not str(getattr(message, "content", "")).startswith("Critic Feedback:")
    ]
    return message_list[starts[-1]:] if starts else message_list


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    question: str
    expected_status: str = "completed"
    expected_evidence_ids: tuple[str, ...] = ()
    mode: str = "Deep"
    category: str = "general"
    expected_claims: tuple[str, ...] = ()
    expected_recovery: bool = False
    fixture: str = "supported"
    expected_outcome: str = "success"
    description: str = ""
    required_source_types: tuple[str, ...] = ()
    citation_requirements: tuple[str, ...] = ()
    must_not_claim: tuple[str, ...] = ()
    expected_failure_behavior: str = ""
    grading_notes: str = ""


def _has_abstention(answer: str) -> bool:
    return bool(re.search(r"\b(insufficient|uncertain|cannot|can't|not enough evidence|unsupported)\b", answer, re.I))


def extract_evidence_ids(answer: str) -> tuple[str, ...]:
    """Return unique structured evidence IDs in first-seen order."""

    return tuple(dict.fromkeys(EVIDENCE_ID_PATTERN.findall(answer or "")))


def available_evidence_ids(messages: Iterable[Any], current_run_only: bool = True) -> tuple[str, ...]:
    """Collect evidence IDs from successful structured tool messages."""

    found: list[str] = []
    message_list = list(messages)
    if current_run_only:
        starts = [
            index for index, message in enumerate(message_list)
            if getattr(message, "type", None) == "human"
            and not str(getattr(message, "content", "")).startswith("Critic Feedback:")
        ]
        message_list = message_list[starts[-1]:] if starts else message_list
    for message in message_list:
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
    cited_urls = extract_citation_urls(answer)
    available_urls = set(available_evidence_urls(messages))
    invalid_urls = tuple(
        url for url in cited_urls
        if (normalized := normalize_citation_url(url)) is None or normalized not in available_urls
    )
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
    if invalid_urls:
        failures.append(f"invalid citation URLs: {', '.join(invalid_urls)}")
    if missing:
        failures.append(f"missing evidence IDs: {', '.join(missing)}")
    expected_failure = case.expected_outcome == "failure"
    expected_abstention = case.expected_outcome == "abstain"
    if expected_abstention and not _has_abstention(answer):
        failures.append("expected an explicit abstention or uncertainty statement")
    expected_behavior_passed = (
        not failures if case.expected_outcome == "success"
        else (
            state.get("run_status") != "completed"
            or bool(failures)
        ) if expected_failure
        else _has_abstention(answer)
    )
    outcome = "passed" if expected_behavior_passed else "failed"
    if state.get("run_status", "").endswith("_failed") and not expected_failure:
        outcome = "error"
    labels = claim_support_labels(answer, cited, available_tokens)
    unsupported = sum(1 for label in labels if label == "unsupported")
    valid = len(cited) - len(invalid)
    coverage = valid / len(case.expected_evidence_ids) if case.expected_evidence_ids else (
        1.0 if not invalid else 0.0
    )
    usage = state.get("usage") or {}
    return {
        "case_id": case.case_id,
        "question": case.question,
        "mode": case.mode,
        "expected_outcome": case.expected_outcome,
        "status": state.get("run_status", "missing"),
        "answer": answer,
        "cited_evidence_ids": list(cited),
        "available_evidence_ids": list(available),
        "failures": failures,
        "passed": outcome == "passed",
        "outcome": outcome,
        "expected_failure": expected_failure,
        "expected_abstention": expected_abstention,
        "expected_behavior_passed": expected_behavior_passed,
        "observed_failures": failures,
        "task_success": outcome == "passed" and case.expected_outcome == "success",
        "expected_recovery": case.expected_recovery,
        "evidence_validity": valid / len(cited) if cited else (1.0 if not available else 0.0),
        "evidence_validity_numerator": len(cited) - len(invalid),
        "evidence_validity_denominator": len(cited),
        "evidence_coverage": min(1.0, coverage),
        "evidence_coverage_numerator": valid,
        "evidence_coverage_denominator": (
            len(case.expected_evidence_ids)
            if case.expected_outcome == "success" else 0
        ),
        "evidence_coverage_applicable": (
            case.expected_outcome == "success" and bool(case.expected_evidence_ids)
        ),
        "claim_support_labels": labels,
        "unsupported_claims": unsupported,
        "claim_count": len(labels),
        "unsupported_claim_denominator": len(labels),
        "recovery": bool(
            case.expected_recovery
            and state.get("recovered", False)
            and state.get("recovery_attempts", 0) > 0
            and outcome == "passed"
        ),
        "recovery_eligible": case.expected_recovery,
        "latency_ms": state.get("latency_ms"),
        "usage": usage,
        "cost": state.get("cost"),
        "availability": {
            "latency": state.get("latency_ms") is not None,
            "usage": bool(usage),
            "cost": state.get("cost") is not None,
        },
        "limitations": [
            "Claim labels are citation-contract classifications, not factuality judgments.",
            "Latency, token usage, and cost are unavailable when the runner does not provide them.",
        ],
    }


def claim_support_labels(answer: str, cited: Iterable[str], available: set[str]) -> list[str]:
    """Classify citation-bearing claims without pretending to judge factual prose."""
    cited_set = set(cited)
    labels: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer or ""):
        if not sentence.strip():
            continue
        ids = set(extract_evidence_ids(sentence))
        if ids:
            labels.append("supported" if ids <= available else "unsupported")
        elif re.search(r"\b(is|are|was|were|has|have|will|shows|means)\b", sentence, re.I):
            labels.append("unsupported")
    return labels


def load_benchmark(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Load the versioned JSON benchmark, rejecting malformed case IDs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError("benchmark must contain a cases array")
    cases = []
    for item in payload["cases"]:
        if not isinstance(item, dict) or not item.get("case_id"):
            raise ValueError("each benchmark case needs a case_id")
        cases.append(EvaluationCase(
            case_id=str(item["case_id"]), question=str(item["question"]),
            expected_status=str(item.get("expected_status", "completed")),
            expected_evidence_ids=tuple(item.get("expected_evidence_ids", ())),
            mode=str(item.get("mode", "Deep")), category=str(item.get("category", "general")),
            expected_claims=tuple(item.get("expected_claims", ())),
            expected_recovery=bool(item.get("expected_recovery", False)),
            fixture=str(item.get("fixture", "supported")),
            expected_outcome=str(item.get("expected_outcome", "success")),
            description=str(item.get("description", "")),
            required_source_types=tuple(item.get("required_source_types", ())),
            citation_requirements=tuple(item.get("citation_requirements", ())),
            must_not_claim=tuple(item.get("must_not_claim", ())),
            expected_failure_behavior=str(item.get("expected_failure_behavior", "")),
            grading_notes=str(item.get("grading_notes", "")),
        ))
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("benchmark case IDs must be unique")
    valid_statuses = {"completed", "completed_after_revision", "critic_rejected",
                      "critic_failed", "executor_failed", "synthesis_failed"}
    invalid = [case.case_id for case in cases if case.expected_status not in valid_statuses]
    if invalid:
        raise ValueError(f"invalid expected_status for: {', '.join(invalid)}")
    return tuple(cases)


def aggregate_results(results: Iterable[dict[str, Any]], thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    rows = list(results)
    attempted = [r for r in rows if r.get("status") not in {"skipped", "not_assessed"}]
    n = len(attempted) or 1
    status_counts = {status: sum(r.get("outcome") == status for r in rows)
                     for status in ("passed", "failed", "error", "skipped", "not_assessed")}
    recovery_rows = [r for r in attempted if r.get("recovery_eligible")]
    coverage_rows = [r for r in attempted if r.get("evidence_coverage_applicable")]
    claim_rows = [r for r in attempted if r.get("unsupported_claim_denominator", 0)]
    validity_rows = [r for r in attempted if r.get("evidence_validity_denominator", 0)]
    recovery_denominator = len(recovery_rows) or 1
    raw_metrics = {
        "expected_behavior_pass_rate": sum(bool(r.get("expected_behavior_passed")) for r in attempted) / n,
        "offline_fixture_task_success_rate": (
            sum(bool(r.get("task_success")) for r in attempted if r.get("expected_outcome", "success") == "success")
            / max(1, sum(r.get("expected_outcome", "success") == "success" for r in attempted))
        ),
        "evidence_validity": (
            sum(r.get("evidence_validity_numerator", 0) for r in validity_rows)
            / max(1, sum(r.get("evidence_validity_denominator", 0) for r in validity_rows))
        ),
        "evidence_coverage": (
            sum(r.get("evidence_coverage_numerator", 0) for r in coverage_rows)
            / max(1, sum(r.get("evidence_coverage_denominator", 0) for r in coverage_rows))
        ),
        "recovery": sum(bool(r.get("recovery", False)) for r in recovery_rows) / recovery_denominator,
        "unsupported_claim_rate": (
            sum(r.get("unsupported_claims", 0) for r in claim_rows)
            / max(1, sum(r.get("unsupported_claim_denominator", 0) for r in claim_rows))
        ),
    }
    thresholds = thresholds or {}
    failures = []
    for name, minimum in thresholds.items():
        metric_name = (
            "offline_fixture_task_success_rate"
            if name in {"task_success", "task_success_rate"}
            else name
        )
        if name == "unsupported_claim_rate":
            if raw_metrics.get(name, 0) > minimum:
                failures.append(f"{name} {raw_metrics[name]:.3f} > {minimum:.3f}")
        elif raw_metrics.get(metric_name, 0) < minimum:
            failures.append(f"{metric_name} {raw_metrics.get(metric_name, 0):.3f} < {minimum:.3f}")
    metrics = {name: round(value, 4) for name, value in raw_metrics.items()}
    success_rows = [r for r in attempted if r.get("expected_outcome", "success") == "success"]
    return {"case_count": len(rows), "attempted": len(attempted),
            "expected_behavior_pass_rate_numerator": sum(bool(r.get("expected_behavior_passed")) for r in attempted),
            "expected_behavior_pass_rate_denominator": len(attempted),
            "offline_fixture_task_success_rate_numerator": sum(bool(r.get("task_success")) for r in success_rows),
            "offline_fixture_task_success_rate_denominator": len(success_rows),
            "evidence_coverage_numerator": sum(r.get("evidence_coverage_numerator", 0) for r in coverage_rows),
            "evidence_coverage_denominator": sum(r.get("evidence_coverage_denominator", 0) for r in coverage_rows),
            "evidence_validity_numerator": sum(r.get("evidence_validity_numerator", 0) for r in validity_rows),
            "evidence_validity_denominator": sum(r.get("evidence_validity_denominator", 0) for r in validity_rows),
            "recovery_numerator": sum(bool(r.get("recovery", False)) for r in recovery_rows),
            "recovery_denominator": len(recovery_rows),
            "unsupported_claim_numerator": sum(r.get("unsupported_claims", 0) for r in claim_rows),
            "unsupported_claim_denominator": sum(r.get("unsupported_claim_denominator", 0) for r in claim_rows),
            "observed_failure_count": sum(bool(r.get("observed_failures")) for r in rows),
            "status_counts": status_counts,
            "metrics": metrics, "raw_metrics": raw_metrics, "thresholds": thresholds,
            "regression_passed": not failures, "regression_failures": failures}


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
        result = (
            "PASS (expected failure)"
            if row["passed"] and row.get("observed_failures")
            else "PASS" if row["passed"] else "FAIL"
        )
        lines.append(
            f"| {row['case_id']} | {row['mode']} | {row['status']} | {evidence} | {result} |"
        )
        for failure in row["failures"]:
            lines.append(f"|  |  |  |  | {failure} |")
    return "\n".join(lines) + "\n"
