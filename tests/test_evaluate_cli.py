"""The offline evaluator must not leave partial results in tracked report files."""

import contextlib
import io
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import evaluate


class TestOfflineEvaluationOutputs(TestCase):
    def _run(self, *args, **kwargs):
        with patch.object(Path, "write_text") as write_text, contextlib.redirect_stdout(io.StringIO()):
            code = evaluate.run_offline_evaluation(*args, **kwargs)
        return code, write_text

    def test_category_subset_does_not_rewrite_tracked_reports(self):
        code, write_text = self._run({"web-search"})
        self.assertEqual(code, 0)
        write_text.assert_not_called()

    def test_no_write_flag_skips_report_files(self):
        code, write_text = self._run(None, write_outputs=False)
        self.assertEqual(code, 0)
        write_text.assert_not_called()

    def test_full_run_writes_both_reports(self):
        code, write_text = self._run(None)
        self.assertEqual(code, 0)
        self.assertEqual(write_text.call_count, 2)
