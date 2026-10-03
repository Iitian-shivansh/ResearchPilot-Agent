"""Tests for the structured tool/evidence response contract."""

from unittest import TestCase

from src.contracts import Evidence, ToolResult, parse_tool_result


class TestContracts(TestCase):
    def test_tool_result_round_trips_evidence(self):
        original = ToolResult(
            ok=True,
            tool_name="query_knowledge_base",
            data={"results": "text"},
            evidence=(
                Evidence(
                    evidence_id="KB-1",
                    source_type="knowledge_base",
                    source_id="notes.md",
                    excerpt="text",
                    score=0.91,
                ),
            ),
        )

        parsed = parse_tool_result(original.to_json())

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.tool_name, "query_knowledge_base")
        self.assertEqual(parsed.evidence[0].evidence_id, "KB-1")
        self.assertEqual(parsed.evidence[0].score, 0.91)

    def test_legacy_plain_text_is_not_treated_as_structured_success(self):
        self.assertIsNone(parse_tool_result("plain tool output"))
