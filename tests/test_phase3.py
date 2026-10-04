from types import SimpleNamespace
from unittest import TestCase

from src.attachments import (
    MAX_ATTACHMENT_BYTES,
    attachment_context,
    load_task_attachments,
)
from src.report import build_markdown_report


def uploaded(name: str, content: bytes):
    return SimpleNamespace(name=name, getvalue=lambda: content)


class TestTaskAttachments(TestCase):
    def test_valid_text_is_loaded_as_task_context(self):
        attachments = load_task_attachments([uploaded("notes.md", b"# Notes\nUse this.")])
        self.assertEqual(attachments[0].name, "notes.md")
        self.assertIn("Use this.", attachments[0].text)
        self.assertIn("reference only", attachment_context(attachments))

    def test_unsupported_and_oversized_files_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_task_attachments([uploaded("notes.pdf", b"no")])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_task_attachments([uploaded("notes.txt", b"x" * (MAX_ATTACHMENT_BYTES + 1))])

    def test_attachment_count_is_bounded(self):
        files = [uploaded(f"{index}.txt", b"x") for index in range(6)]
        with self.assertRaisesRegex(ValueError, "At most"):
            load_task_attachments(files)


class TestReport(TestCase):
    def test_report_contains_answer_mode_and_trace(self):
        report = build_markdown_report(
            "What changed?",
            "The answer [WEB-1].",
            "Deep",
            [{"node": "planner", "status": "completed"}],
            ["notes.md"],
        )
        self.assertIn("## Answer", report)
        self.assertIn("The answer [WEB-1].", report)
        self.assertIn("**Mode:** Deep", report)
        self.assertIn("`notes.md`", report)
        self.assertIn("**planner**", report)
