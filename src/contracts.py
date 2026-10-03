"""Serializable contracts shared by agent tools and the critic."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """A source or computation that can support a final answer claim."""

    evidence_id: str
    source_type: str
    source_id: str
    excerpt: str = ""
    score: float | None = None


@dataclass(frozen=True)
class ToolResult:
    """Stable tool response envelope understood by the executor and UI."""

    ok: bool
    tool_name: str
    data: Any = None
    error_type: str | None = None
    error_message: str | None = None
    evidence: tuple[Evidence, ...] = ()

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_tool_result(content: str) -> ToolResult | None:
    """Parse a structured tool response, returning None for legacy output."""

    try:
        value = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or not {"ok", "tool_name"} <= value.keys():
        return None
    evidence = tuple(Evidence(**item) for item in value.get("evidence", ()))
    return ToolResult(
        ok=bool(value["ok"]),
        tool_name=str(value["tool_name"]),
        data=value.get("data"),
        error_type=value.get("error_type"),
        error_message=value.get("error_message"),
        evidence=evidence,
    )
