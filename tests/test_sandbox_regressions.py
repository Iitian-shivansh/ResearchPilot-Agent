"""Regression tests for sandbox defects found in the repository audit."""

import unittest

from src.sandbox import _check_blocked_patterns, _validate_ast, run_code_in_subprocess


class TestBlocklistIgnoresData(unittest.TestCase):
    """Retrieved text embedded as a string literal is data, not code."""

    def test_string_literal_words_are_not_blocked(self):
        for code in (
            "text = '''The API handles 10k requests per second.'''\nprint(len(text))",
            "text = 'Signal-to-noise improved; threading model changed.'\nprint(text.count(' '))",
            "print('the store will reopen(soon)')",
            "# count socket errors\nprint(1 + 1)",
            'print(f"{1 + 1} requests")',
        ):
            with self.subTest(code=code):
                self.assertIsNone(_check_blocked_patterns(code))

    def test_executed_text_runs_successfully(self):
        result = run_code_in_subprocess(
            "text = '''Servers saw 1200 requests and 3 socket timeouts.'''\n"
            "print(len(text.split()))"
        )
        self.assertTrue(result.success, result.stderr)
        self.assertEqual(result.stdout.strip(), "8")

    def test_code_outside_literals_is_still_blocked(self):
        self.assertEqual(_check_blocked_patterns("import socket"), "socket")
        self.assertEqual(_check_blocked_patterns("x = 'ok'\nimport subprocess"), "subprocess")
        # Code inside an f-string replacement field is executable and must be rejected
        # (by the blocklist on 3.12+, and by the AST validator on every version).
        self.assertFalse(run_code_in_subprocess('print(f"{open(1)}")').success)

    def test_untokenizable_code_falls_back_to_raw_scan(self):
        self.assertEqual(_check_blocked_patterns("x = '''unterminated socket"), "socket")


class TestFormatStringTraversal(unittest.TestCase):
    """str.format resolves '{0.attr}' with getattr, bypassing the dunder AST check."""

    def test_runtime_built_format_string_is_rejected(self):
        code = "print(('{0._' + '_class_' + '_._' + '_name_' + '_}').format(1))"
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ASTValidationError")

    def test_literal_dunder_format_string_is_rejected(self):
        self.assertIsNotNone(_validate_ast("print('{0.__class__}'.format(1))"))

    def test_format_map_and_formatter_are_rejected(self):
        self.assertIsNotNone(_validate_ast("t = '{x}'\nprint(t.format_map({'x': 1}))"))
        self.assertIsNotNone(_validate_ast("import string\nstring.Formatter().vformat('{0}', (1,), {})"))
        self.assertIsNotNone(_validate_ast("print(str.format('{0}', 1))"))

    def test_ordinary_formatting_still_works(self):
        result = run_code_in_subprocess(
            "x = 3.14159\nprint('{:.2f}'.format(x))\nprint(f'{x:.1f}')\nprint(format(x, '.3f'))"
        )
        self.assertTrue(result.success, result.stderr)
        self.assertEqual(result.stdout.split(), ["3.14", "3.1", "3.142"])


if __name__ == "__main__":
    unittest.main()
