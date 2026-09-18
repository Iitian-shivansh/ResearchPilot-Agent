"""
Sandbox module for isolated Python code execution.

Provides subprocess-based execution of LLM-generated Python code with:
    - Hard timeout enforcement (OS-level process kill)
    - Restricted builtins namespace
    - Environment variable sanitization
    - Structured result reporting
    - Safe metadata logging (no secrets, no code content)

This module is the ONLY place where execution limits are defined.
Import SandboxConfig to access or override limits.

Security guarantees and limitations are documented in docs/SECURITY.md.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration — single source of truth for all execution limits
# ---------------------------------------------------------------------------

@dataclass
class SandboxConfig:
    """
    Configurable limits for sandboxed Python execution.

    All limits are defined here and nowhere else. To change defaults,
    modify this class or pass overrides when constructing an instance.
    """

    # Maximum wall-clock time for code execution (seconds).
    # Enforced by subprocess timeout — the OS kills the process.
    # This is a HARD guarantee: the parent process cannot hang.
    max_execution_seconds: int = 10

    # Maximum length of generated code (characters).
    # Checked BEFORE launching a subprocess — no resources wasted.
    max_code_length: int = 50_000

    # Maximum length of captured stdout + stderr (characters).
    # Output beyond this limit is truncated with a warning.
    max_output_length: int = 10_000


# ---------------------------------------------------------------------------
# Structured result
# ---------------------------------------------------------------------------

@dataclass
class SandboxResult:
    """
    Structured result from sandboxed Python execution.

    Attributes:
        success: True if code executed without exceptions.
        stdout: Captured standard output from the code.
        stderr: Captured standard error / exception messages.
        error_type: Exception class name if an error occurred, else None.
        execution_time_ms: Wall-clock execution time in milliseconds.
    """

    success: bool
    stdout: str = ""
    stderr: str = ""
    error_type: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
        }


# ---------------------------------------------------------------------------
# Result delimiter — must match _sandbox_worker.py exactly
# ---------------------------------------------------------------------------
_RESULT_DELIMITER = "___SANDBOX_RESULT_ENVELOPE_8f3a1b2c___"


# ---------------------------------------------------------------------------
# Path to the worker script
# ---------------------------------------------------------------------------
_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "_sandbox_worker.py")


# ---------------------------------------------------------------------------
# Blocklist — checked BEFORE sending code to the subprocess
# ---------------------------------------------------------------------------
# This is a defense-in-depth layer. The primary isolation comes from the
# restricted builtins in the worker process. The blocklist catches common
# dangerous patterns early, but a determined attacker can bypass string
# matching. Do NOT rely on this as the sole security mechanism.
_BLOCKED_PATTERNS = [
    "os.system",
    "subprocess",
    "shutil",
    "open(",
    "eval(",
    "exec(",
    "__import__",
    "importlib",
    "pathlib",
    "socket",
    "requests",
    "urllib",
    "http.client",
    "ftplib",
    "smtplib",
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
]


def _check_blocked_patterns(code: str) -> Optional[str]:
    """
    Check code against the blocklist. Returns the matched pattern if found,
    or None if the code passes.

    NOTE: This is best-effort. String matching can be evaded.
    The real isolation comes from restricted builtins + subprocess boundary.
    """
    code_lower = code.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            return pattern
    return None


# ---------------------------------------------------------------------------
# Environment for the subprocess — strip secrets
# ---------------------------------------------------------------------------

_SENSITIVE_ENV_PATTERNS = [
    "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
    "PRIVATE_KEY", "AUTH", "DATABASE_URL", "CONNECTION_STRING",
]


def _build_clean_env() -> dict:
    """
    Build an environment dict for the subprocess with secrets removed.

    Keeps PATH, PYTHONPATH, and other non-secret variables so that the
    Python interpreter can start correctly. Removes anything that looks
    like a secret based on common naming patterns.

    This is best-effort — an unusual secret variable name might slip through.
    """
    clean = {}
    for key, value in os.environ.items():
        key_upper = key.upper()
        is_sensitive = any(p in key_upper for p in _SENSITIVE_ENV_PATTERNS)
        if not is_sensitive:
            clean[key] = value
    return clean


# ---------------------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------------------

def run_code_in_subprocess(
    code: str,
    config: Optional[SandboxConfig] = None,
) -> SandboxResult:
    """
    Execute Python code in an isolated subprocess.

    Args:
        code: The Python source code to execute.
        config: Execution limits. Uses defaults if not provided.

    Returns:
        SandboxResult with success status, output, errors, and timing.

    This function NEVER raises exceptions to the caller. All error
    conditions are captured in the returned SandboxResult.
    """
    if config is None:
        config = SandboxConfig()

    start_time = time.monotonic()

    # --- Pre-flight checks ---

    # Check for empty/whitespace-only code
    if not code or not code.strip():
        return SandboxResult(
            success=False,
            stderr="No code provided.",
            error_type="ValidationError",
            execution_time_ms=0.0,
        )

    # Check code length
    if len(code) > config.max_code_length:
        logger.warning(
            "Code rejected: length %d exceeds limit %d",
            len(code), config.max_code_length,
        )
        return SandboxResult(
            success=False,
            stderr=(
                f"Code too large: {len(code)} characters "
                f"(limit: {config.max_code_length})."
            ),
            error_type="CodeTooLarge",
            execution_time_ms=0.0,
        )

    # Check blocked patterns (defense-in-depth, not sole protection)
    blocked = _check_blocked_patterns(code)
    if blocked:
        logger.info(
            "Code blocked by pattern check: '%s' (code_length=%d)",
            blocked, len(code),
        )
        return SandboxResult(
            success=False,
            stderr=(
                f"Execution blocked: '{blocked}' is not allowed "
                f"in the sandbox for security reasons."
            ),
            error_type="BlockedPattern",
            execution_time_ms=0.0,
        )

    # --- Write code to a temp file ---
    tmp_file = None
    try:
        tmp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="sandbox_",
            delete=False,
            encoding="utf-8",
        )
        tmp_file.write(code)
        tmp_file.close()

        # --- Launch subprocess ---
        clean_env = _build_clean_env()

        try:
            proc = subprocess.run(
                [sys.executable, _WORKER_SCRIPT, tmp_file.name],
                capture_output=True,
                text=True,
                timeout=config.max_execution_seconds,
                env=clean_env,
                # Do not allow the child to inherit stdin
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.warning(
                "Code execution timed out after %ds (code_length=%d)",
                config.max_execution_seconds, len(code),
            )
            return SandboxResult(
                success=False,
                stderr=(
                    f"Execution timed out after "
                    f"{config.max_execution_seconds} seconds."
                ),
                error_type="Timeout",
                execution_time_ms=round(elapsed_ms, 2),
            )
        except OSError as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error("Failed to start sandbox subprocess: %s", e)
            return SandboxResult(
                success=False,
                stderr=f"Failed to start execution subprocess: {e}",
                error_type="ProcessStartupError",
                execution_time_ms=round(elapsed_ms, 2),
            )

        # --- Parse the result envelope ---
        raw_stdout = proc.stdout or ""
        raw_stderr = proc.stderr or ""

        # Find the result delimiter in stdout
        if _RESULT_DELIMITER in raw_stdout:
            parts = raw_stdout.split(_RESULT_DELIMITER, 1)
            # Everything before the delimiter is unexpected pre-output (should be empty)
            # Everything after is the JSON result
            json_part = parts[1].strip()
            try:
                result_data = json.loads(json_part)
            except json.JSONDecodeError:
                elapsed_ms = (time.monotonic() - start_time) * 1000
                return SandboxResult(
                    success=False,
                    stderr=(
                        "Worker produced malformed result. "
                        f"stderr: {raw_stderr[:500]}"
                    ),
                    error_type="WorkerError",
                    execution_time_ms=round(elapsed_ms, 2),
                )

            stdout_val = result_data.get("stdout", "")
            stderr_val = result_data.get("stderr", "")

            # Truncate oversized output
            if len(stdout_val) > config.max_output_length:
                stdout_val = (
                    stdout_val[:config.max_output_length]
                    + "\n\n... [output truncated: exceeded "
                    + f"{config.max_output_length} character limit]"
                )
            if len(stderr_val) > config.max_output_length:
                stderr_val = stderr_val[:config.max_output_length] + "\n... [truncated]"

            elapsed_ms = result_data.get(
                "execution_time_ms",
                (time.monotonic() - start_time) * 1000,
            )

            result = SandboxResult(
                success=result_data.get("success", False),
                stdout=stdout_val,
                stderr=stderr_val,
                error_type=result_data.get("error_type"),
                execution_time_ms=round(float(elapsed_ms), 2),
            )

        else:
            # Worker crashed before producing a result envelope
            elapsed_ms = (time.monotonic() - start_time) * 1000
            combined_err = raw_stderr or raw_stdout or "Unknown worker error"
            result = SandboxResult(
                success=False,
                stderr=f"Worker error: {combined_err[:1000]}",
                error_type="WorkerError",
                execution_time_ms=round(elapsed_ms, 2),
            )

        # --- Log safe metadata ---
        logger.info(
            "Sandbox execution: success=%s duration_ms=%.1f error_type=%s "
            "code_length=%d output_length=%d",
            result.success,
            result.execution_time_ms,
            result.error_type,
            len(code),
            len(result.stdout),
        )

        return result

    except Exception as e:
        # Catch-all: should never happen, but don't crash the parent
        elapsed_ms = (time.monotonic() - start_time) * 1000
        logger.error("Unexpected sandbox error: %s", e, exc_info=True)
        return SandboxResult(
            success=False,
            stderr=f"Unexpected sandbox error: {e}",
            error_type="InternalError",
            execution_time_ms=round(elapsed_ms, 2),
        )
    finally:
        # Clean up temp file
        if tmp_file is not None:
            try:
                os.unlink(tmp_file.name)
            except OSError:
                pass
