# Offline evaluation results

**Passed:** 50/50

| Case | Mode | Status | Evidence | Result |
|---|---|---|---|---|
| m2-001 | Deep | completed | [WEB-1] | PASS |
| m2-002 | Deep | completed | [WEB-2] | PASS |
| m2-003 | Deep | completed | [WEB-999] | PASS (expected failure) |
|  |  |  |  | invalid evidence IDs: [WEB-999] |
|  |  |  |  | missing evidence IDs: [WEB-3] |
| m2-004 | Deep | completed | [WEB-4] | PASS |
| m2-005 | Deep | completed | [WEB-5] | PASS |
| m2-006 | Deep | completed | [WEB-6] | PASS |
| m2-007 | Deep | completed | [WEB-7] | PASS |
| m2-008 | Deep | completed | [WEB-8] | PASS |
| m2-009 | Deep | completed | [WEB-9] | PASS |
| m2-010 | Deep | completed | [WEB-10] | PASS |
| m2-011 | Deep | completed | [KB-11] | PASS |
| m2-012 | Deep | completed | none | PASS (expected failure) |
|  |  |  |  | missing evidence IDs: [KB-12] |
| m2-013 | Deep | completed | [KB-13] | PASS |
| m2-014 | Deep | completed | [KB-14] | PASS |
| m2-015 | Deep | completed | [KB-15] | PASS |
| m2-016 | Deep | completed | [KB-16] | PASS |
| m2-017 | Deep | completed | [WEB-999] | PASS (expected failure) |
|  |  |  |  | invalid evidence IDs: [WEB-999] |
|  |  |  |  | missing evidence IDs: [KB-17] |
| m2-018 | Deep | completed | [KB-18] | PASS |
| m2-019 | Deep | completed | [KB-19] | PASS |
| m2-020 | Deep | completed | [KB-20] | PASS |
| m2-021 | Deep | completed | [WEB-21] | PASS |
| m2-022 | Deep | completed | [WEB-22] | PASS |
| m2-023 | Deep | completed | [WEB-23] | PASS |
| m2-024 | Deep | completed | [WEB-24] | PASS |
| m2-025 | Deep | completed | [WEB-25] | PASS |
| m2-026 | Deep | completed | [WEB-26] | PASS |
| m2-027 | Deep | completed | [WEB-27] | PASS |
| m2-028 | Deep | completed | [WEB-28] | PASS |
| m2-029 | Deep | completed | [WEB-999] | PASS (expected failure) |
|  |  |  |  | invalid evidence IDs: [WEB-999] |
|  |  |  |  | missing evidence IDs: [WEB-29] |
| m2-030 | Deep | completed | [WEB-30] | PASS |
| m2-031 | Deep | completed | [WEB-31] | PASS |
| m2-032 | Deep | completed | [WEB-32] | PASS |
| m2-033 | Deep | completed | [WEB-33] | PASS |
| m2-034 | Deep | completed | [WEB-34] | PASS |
| m2-035 | Deep | completed | [WEB-35] | PASS |
| m2-036 | Deep | executor_failed | none | PASS (expected failure) |
|  |  |  |  | missing evidence IDs: [WEB-36] |
| m2-037 | Deep | executor_failed | none | PASS (expected failure) |
|  |  |  |  | missing evidence IDs: [WEB-37] |
| m2-038 | Deep | completed | [WEB-38] | PASS |
| m2-039 | Deep | completed | none | PASS |
| m2-040 | Deep | completed | none | PASS |
| m2-041 | Deep | completed | none | PASS |
| m2-042 | Deep | completed | none | PASS |
| m2-043 | Deep | completed | none | PASS |
| m2-044 | Deep | executor_failed | none | PASS |
| m2-045 | Deep | executor_failed | none | PASS |
| m2-046 | Deep | executor_failed | none | PASS |
| m2-047 | Deep | executor_failed | none | PASS |
| m2-048 | Deep | completed | [CALC-48] | PASS |
| m2-049 | Deep | completed | [CALC-49] | PASS |
| m2-050 | Deep | completed | [CALC-50] | PASS |

## Metric definitions

- **Expected-behavior pass rate:** correctly handled cases / attempted cases; includes expected failures and abstentions.
- **Offline fixture task-success rate:** successful controlled fixture cases / cases whose expected outcome is success; excludes failure and abstention fixtures and is not real-world research accuracy.
- **Evidence validity:** valid cited evidence references / cited references from current-run structured evidence.
- **Evidence coverage:** satisfied applicable required evidence references / applicable required references; non-research and expected-failure cases are excluded.
- **Unsupported-claim rate:** unsupported fixture-labeled claims / independently labeled claims; offline labels are contract checks, not semantic accuracy.
- **Recovery rate:** eligible recoverable failures with observed recovery and expected outcome / eligible recoverable failures.

Offline fixtures do not measure real-world model accuracy, source quality, or semantic claim entailment.

## Aggregate metrics

```json
{
  "case_count": 50,
  "attempted": 50,
  "expected_behavior_pass_rate_numerator": 50,
  "expected_behavior_pass_rate_denominator": 50,
  "offline_fixture_task_success_rate_numerator": 35,
  "offline_fixture_task_success_rate_denominator": 35,
  "evidence_coverage_numerator": 35,
  "evidence_coverage_denominator": 35,
  "evidence_validity_numerator": 35,
  "evidence_validity_denominator": 38,
  "recovery_numerator": 1,
  "recovery_denominator": 1,
  "unsupported_claim_numerator": 3,
  "unsupported_claim_denominator": 38,
  "observed_failure_count": 6,
  "status_counts": {
    "passed": 50,
    "failed": 0,
    "error": 0,
    "skipped": 0,
    "not_assessed": 0
  },
  "metrics": {
    "expected_behavior_pass_rate": 1.0,
    "offline_fixture_task_success_rate": 1.0,
    "evidence_validity": 0.9211,
    "evidence_coverage": 1.0,
    "recovery": 1.0,
    "unsupported_claim_rate": 0.0789
  },
  "raw_metrics": {
    "expected_behavior_pass_rate": 1.0,
    "offline_fixture_task_success_rate": 1.0,
    "evidence_validity": 0.9210526315789473,
    "evidence_coverage": 1.0,
    "recovery": 1.0,
    "unsupported_claim_rate": 0.07894736842105263
  },
  "thresholds": {
    "offline_fixture_task_success_rate": 1.0,
    "evidence_validity": 0.9,
    "evidence_coverage": 0.88,
    "unsupported_claim_rate": 0.1
  },
  "regression_passed": true,
  "regression_failures": []
}
```
