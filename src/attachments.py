"""Validation and task-scoped handling for uploaded research notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

SUPPORTED_ATTACHMENT_TYPES = frozenset({".txt", ".md"})
MAX_ATTACHMENT_COUNT = 5
MAX_ATTACHMENT_BYTES = 1_000_000
MAX_TOTAL_ATTACHMENT_BYTES = 4_000_000
MAX_CONTEXT_CHARS = 120_000


class UploadedFileLike(Protocol):
    name: str

    def getvalue(self) -> bytes:
        ...


@dataclass(frozen=True)
class TaskAttachment:
    """A validated attachment kept in the current task only."""

    name: str
    text: str
    size_bytes: int


def load_task_attachments(files: list[UploadedFileLike] | None) -> tuple[TaskAttachment, ...]:
    """Decode bounded text attachments without writing to or indexing Qdrant."""

    files = files or []
    if len(files) > MAX_ATTACHMENT_COUNT:
        raise ValueError(f"At most {MAX_ATTACHMENT_COUNT} attachments are allowed")

    total_bytes = 0
    attachments: list[TaskAttachment] = []
    total_chars = 0
    for uploaded in files:
        name = str(uploaded.name or "").strip()
        suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
        if suffix not in SUPPORTED_ATTACHMENT_TYPES:
            raise ValueError(f"Unsupported attachment '{name}'. Use .txt or .md files")
        raw = uploaded.getvalue()
        if not isinstance(raw, bytes):
            raise ValueError(f"Attachment '{name}' did not provide bytes")
        if len(raw) > MAX_ATTACHMENT_BYTES:
            raise ValueError(
                f"Attachment '{name}' exceeds the {MAX_ATTACHMENT_BYTES // 1_000_000} MB limit"
            )
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL_ATTACHMENT_BYTES:
            raise ValueError(
                f"Attachments exceed the {MAX_TOTAL_ATTACHMENT_BYTES // 1_000_000} MB total limit"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"Attachment '{name}' must be UTF-8 text") from error
        if not text.strip():
            raise ValueError(f"Attachment '{name}' is empty")
        if total_chars + len(text) > MAX_CONTEXT_CHARS:
            raise ValueError("Attachment text exceeds the task context limit")
        attachments.append(TaskAttachment(name=name, text=text, size_bytes=len(raw)))
        total_chars += len(text)
    return tuple(attachments)


def attachment_context(attachments: tuple[TaskAttachment, ...]) -> str:
    """Format attachments as temporary prompt context with clear boundaries."""

    if not attachments:
        return ""
    sections = [
        f"--- Untrusted task attachment (reference only): {attachment.name} ---\n{attachment.text}"
        for attachment in attachments
    ]
    return "\n\n".join(sections)
