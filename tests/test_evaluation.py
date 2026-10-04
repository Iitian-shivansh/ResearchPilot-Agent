import json
from types import SimpleNamespace
from unittest import TestCase

from src.evaluation import (
    EvaluationCase,
    available_evidence_ids,
    evaluate_state,
    extract_evidence_ids,
    render_evaluation_markdown,
)


class TestEvaluation(TestCase):
    def test_extracts_unique_structured_ids(self):
        self.assertEqual(
            extract_evidence_ids("See [KB-1], [KB-1], and [CALC-1]."),
            ("[KB-1]", "[CALC-1]"),
        )

    def test_available_evidence_ignores_unstructured_tool_output(self):
        messages = [
            SimpleNamespace(type="tool", content=json.dumps({
                "ok": True,
                "tool_name": "research",
                "evidence": [{
                    "evidence_id": "WEB-1",
                    "source_type": "web",
                    "source_id": "example",
                }],
            })),
            SimpleNamespace(type="tool", content="legacy text"),
        ]
        self.assertEqual(available_evidence_ids(messages), ("WEB-1",))

    def test_state_fails_on_fabricated_citation(self):
        state = {
            "run_status": "completed",
            "messages": [SimpleNamespace(type="ai", tool_calls=[], content="Claim [WEB-9].")],
        }
        result = evaluate_state(state, EvaluationCase("citation", "q"))
        self.assertFalse(result["passed"])
        self.assertIn("invalid evidence IDs", result["failures"][0])

    def test_markdown_reports_pass_count_and_failures(self):
        markdown = render_evaluation_markdown([
            {"case_id": "ok", "mode": "Deep", "status": "completed",
             "cited_evidence_ids": [], "failures": [], "passed": True},
            {"case_id": "bad", "mode": "Quick", "status": "failed",
             "cited_evidence_ids": [], "failures": ["bad status"], "passed": False},
        ])
        self.assertIn("**Passed:** 1/2", markdown)
        self.assertIn("bad status", markdown)
