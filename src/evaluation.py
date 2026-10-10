"""Offline evaluation helpers for answer and evidence integrity."""

from __future__ import annotations

import re
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
