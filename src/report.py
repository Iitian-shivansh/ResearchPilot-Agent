"""Markdown report export helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable


def build_markdown_report(
    query: str,
    answer: str,
    mode: str,
    trace: Iterable[dict[str, str]] = (),
    attachment_names: Iterable[str] = (),
) -> str:
    """Build a portable report from the verified answer and visible execution trace."""

    lines = [
        "# Research report",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Mode:** {mode}",
        "",
        "## Question",
        "",
        query.strip(),
        "",
        "## Answer",
        "",
        answer.strip() or "_No answer was produced._",
        "",
    ]
    names = list(attachment_names)
    if names:
        lines.extend(["## Task-scoped attachments", "", *[f"- `{name}`" for name in names], ""])
    trace_items = list(trace)
    if trace_items:
        lines.extend(["## Execution trace", ""])
        lines.extend(
            f"- **{item.get('node', 'step')}** -- {item.get('status', 'completed')}"
            for item in trace_items
        )
        lines.append("")
    return "\n".join(lines)
