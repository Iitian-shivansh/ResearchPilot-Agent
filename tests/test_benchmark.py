import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

from src.evaluation import (
    EvaluationCase, aggregate_results, evaluate_state, load_benchmark,
)


ROOT = Path(__file__).parents[1]


class TestMilestone2Benchmark(TestCase):
    def test_dataset_is_versioned_and_has_fifty_cases(self):
        payload = json.loads((ROOT / "benchmark" / "milestone2_v1.json").read_text())
        self.assertEqual(payload["benchmark_version"], "milestone2_v1")
        self.assertEqual(len(load_benchmark(ROOT / "benchmark" / "milestone2_v1.json")), 50)
        self.assertEqual(len({case.case_id for case in load_benchmark(ROOT / "benchmark" / "milestone2_v1.json")}), 50)

    def test_result_reports_claim_labels_and_availability(self):
        state = {
            "run_status": "completed", "latency_ms": 12, "usage": {"total_tokens": 3},
            "cost": 0.01, "messages": [
                SimpleNamespace(type="tool", content=json.dumps({
                    "ok": True, "tool_name": "fixture", "evidence": [{
                        "evidence_id": "WEB-1", "source_type": "web",
                        "source_id": "https://example.test/source",
                    }],
                })),
                SimpleNamespace(type="ai", tool_calls=[], content="A claim [WEB-1]."),
            ],
        }
        result = evaluate_state(state, EvaluationCase("x", "q", expected_evidence_ids=("[WEB-1]",)))
        self.assertEqual(result["claim_support_labels"], ["supported"])
        self.assertEqual(result["evidence_coverage"], 1.0)
        self.assertEqual(result["availability"], {"latency": True, "usage": True, "cost": True})

    def test_regression_thresholds_fail_on_unsupported_claims(self):
        summary = aggregate_results([{
            "passed": False, "task_success": False, "evidence_validity": 0,
            "evidence_coverage": 0, "recovery": False, "unsupported_claims": 1,
        }], {"task_success": 1.0, "unsupported_claim_rate": 0.0})
        self.assertFalse(summary["regression_passed"])
        self.assertTrue(summary["regression_failures"])

    def test_expected_failure_is_not_counted_as_research_success(self):
        case = EvaluationCase(
            "failure", "q", expected_status="executor_failed",
            expected_outcome="failure", expected_evidence_ids=("[WEB-1]",),
        )
        result = evaluate_state({"run_status": "executor_failed", "messages": [
            SimpleNamespace(type="ai", tool_calls=[], content="No answer.")
        ]}, case)
        self.assertTrue(result["expected_behavior_passed"])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["evidence_coverage_denominator"], 0)

    def test_aggregate_reports_separate_denominators(self):
        rows = [
            {"outcome": "passed", "expected_behavior_passed": True, "expected_outcome": "success",
             "task_success": True, "evidence_coverage_applicable": False,
             "recovery_eligible": False, "unsupported_claim_denominator": 2,
             "unsupported_claims": 1},
            {"outcome": "passed", "expected_behavior_passed": True, "expected_outcome": "failure",
             "task_success": False, "evidence_coverage_applicable": False,
             "recovery_eligible": True, "recovery": True,
             "unsupported_claim_denominator": 0, "unsupported_claims": 0},
        ]
        summary = aggregate_results(rows)
        self.assertEqual(summary["expected_behavior_pass_rate_numerator"], 2)
        self.assertEqual(summary["offline_fixture_task_success_rate_numerator"], 1)
        self.assertEqual(summary["offline_fixture_task_success_rate_denominator"], 1)
        self.assertEqual(summary["unsupported_claim_denominator"], 2)
        self.assertEqual(summary["recovery_denominator"], 1)

    def test_evidence_validity_uses_reference_denominator_and_display_precision(self):
        rows = [{
            "outcome": "passed", "expected_behavior_passed": True,
            "expected_outcome": "success", "task_success": True,
            "evidence_validity_numerator": 35,
            "evidence_validity_denominator": 38,
            "evidence_coverage_applicable": False,
            "recovery_eligible": False,
            "unsupported_claim_denominator": 0,
        }]
        summary = aggregate_results(rows, {"evidence_validity": 0.92})
        self.assertAlmostEqual(summary["raw_metrics"]["evidence_validity"], 35 / 38)
        self.assertEqual(summary["metrics"]["evidence_validity"], 0.9211)
        self.assertEqual(summary["evidence_validity_numerator"], 35)
        self.assertEqual(summary["evidence_validity_denominator"], 38)
        self.assertTrue(summary["regression_passed"])
