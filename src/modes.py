"""Research mode configuration shared by the UI and agent prompts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchMode:
    name: str
    instruction: str
    max_tool_rounds: int


MODES = {
    "Quick": ResearchMode(
        "Quick",
        "Prefer a concise answer and the smallest sufficient evidence set.",
        2,
    ),
    "Deep": ResearchMode(
        "Deep",
        "Investigate thoroughly, cross-check important claims, and explain limitations.",
        4,
    ),
    "Compare": ResearchMode(
        "Compare",
        "Identify the alternatives, compare them using consistent criteria, and present trade-offs.",
        4,
    ),
    "Analyze": ResearchMode(
        "Analyze",
        "Break the question into assumptions, evidence, implications, and uncertainties.",
        3,
    ),
}


def get_mode(name: str | None) -> ResearchMode:
    """Return a supported mode, defaulting safely to Deep."""

    return MODES.get(name or "Deep", MODES["Deep"])
