"""
Sandbox module for isolated Python code execution.

Provides subprocess-based execution of LLM-generated Python code with:
    - Hard timeout enforcement (OS-level process kill)
    - AST validation (primary source-level restriction)
    - String blocklist (defense-in-depth)
    - Restricted builtins namespace
    - Environment variable allowlist
    - Structured result reporting
    - Safe metadata logging (no secrets, no code content)

This module is the ONLY place where execution limits are defined.
Import SandboxConfig to access or override limits.

Security guarantees and limitations are documented in docs/SECURITY.md.
"""

import ast
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
# Blocklist — defense-in-depth, checked BEFORE sending code to subprocess
# ---------------------------------------------------------------------------
# NOTE: AST validation (below) is the primary source-level restriction.
# This blocklist is defense-in-depth only. String matching can be evaded
# by encoding tricks, concatenation, or indirect access. Do NOT rely on
# this as the sole security mechanism.
_BLOCKED_PATTERNS = [
    # Process and shell access
    "os.system", "os.popen", "os.exec", "os.spawn",
    "subprocess",
    # Filesystem
    "shutil",
    "open(",
    "pathlib",
    # Code execution
    "eval(",
    "exec(",
    "compile(",
    "breakpoint",
    # Import mechanisms
    "__import__",
    "importlib",
    "load_module",
    # Network
    "socket",
    "requests",
    "urllib",
    "http.client",
    "ftplib",
    "smtplib",
    # Low-level / unsafe
    "ctypes",
    "multiprocessing",
    "threading",
    "signal",
    # Introspection / escape (defense-in-depth; AST validation is primary)
    "__subclasses__",
    "__bases__",
    "__mro__",
    "__globals__",
    "__code__",
    "__closure__",
    "__builtins__",
    "__loader__",
]


def _check_blocked_patterns(code: str) -> Optional[str]:
    """
    Check code against the blocklist. Returns the matched pattern if found,
    or None if the code passes.

    NOTE: This is best-effort defense-in-depth. String matching can be evaded.
    AST validation is the primary source-level restriction.
    """
    code_lower = code.lower()
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            return pattern
    return None


# ---------------------------------------------------------------------------
# AST validation — primary source-level restriction
# ---------------------------------------------------------------------------
# AST validation is stronger than regex/string blocklists because it operates
# on the parsed syntax tree, not raw text. It cannot be bypassed by string
# concatenation, encoding tricks, comment injection, or variable naming.
# The regex blocklist above is retained as defense-in-depth.
# ---------------------------------------------------------------------------

# Module whitelist — must match _sandbox_worker.py _ALLOWED_MODULES
_ALLOWED_IMPORT_MODULES = frozenset([
    "math", "json", "time", "re", "collections", "statistics",
    "itertools", "functools", "string", "textwrap",
    "decimal", "fractions", "random", "datetime",
])

# Names that must not appear as bare references in user code
_DANGEROUS_NAMES = frozenset([
    "exec", "eval", "compile", "breakpoint",
    "__import__", "__builtins__",
    "getattr", "setattr", "hasattr", "delattr",
    "type", "globals", "locals", "vars",
    "open", "input",
])

# Non-dunder attributes that enable dynamic loading
_DANGEROUS_ATTRS = frozenset([
    "load_module", "find_module", "find_spec", "module_from_spec",
    "get_data",
])


def _validate_ast(code: str) -> Optional[str]:
    """
    Parse code as AST and reject dangerous constructs.

    Returns an error message if a dangerous construct is found, or None
    if the code passes validation.

    Checks performed:
        1. Dunder attribute access (e.g., obj.__class__, obj.__globals__)
        2. Dangerous bare names (e.g., exec, eval, getattr, type)
        3. Dangerous non-dunder attributes (e.g., load_module)
        4. Import of non-whitelisted modules
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Let the worker report syntax errors with proper line numbers
        return None

    for node in ast.walk(tree):
        # 1. Reject dunder attribute access
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return (
                    f"Access to dunder attribute '{node.attr}' is not allowed "
                    f"in the sandbox (line {getattr(node, 'lineno', '?')})"
                )
            if node.attr in _DANGEROUS_ATTRS:
                return (
                    f"Access to '{node.attr}' is not allowed "
                    f"in the sandbox (line {getattr(node, 'lineno', '?')})"
                )

        # 2. Reject dangerous bare names
        if isinstance(node, ast.Name) and node.id in _DANGEROUS_NAMES:
            return (
                f"Use of '{node.id}' is not allowed "
                f"in the sandbox (line {getattr(node, 'lineno', '?')})"
            )

        # 3. Reject imports of non-whitelisted modules
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in _ALLOWED_IMPORT_MODULES:
                    return (
                        f"Import of '{alias.name}' is not allowed. "
                        f"Allowed: {', '.join(sorted(_ALLOWED_IMPORT_MODULES))}"
                    )

        if isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top not in _ALLOWED_IMPORT_MODULES:
                    return (
                        f"Import from '{node.module}' is not allowed. "
                        f"Allowed: {', '.join(sorted(_ALLOWED_IMPORT_MODULES))}"
                    )

    return None


# ---------------------------------------------------------------------------
# Environment allowlist for the subprocess
# ---------------------------------------------------------------------------
# Uses an allowlist approach: only explicitly approved variables are passed
# to the worker. This prevents accidental exposure of API keys, tokens,
# passwords, cloud credentials, and other secrets regardless of naming.
#
# Platform-specific behavior:
#   Windows: requires SYSTEMROOT, WINDIR, COMSPEC for OS functionality
#   Linux:   requires HOME, LANG for locale and user context
# ---------------------------------------------------------------------------

_ALLOWED_ENV_KEYS = frozenset({
    # Windows OS requirements
    "SYSTEMROOT", "WINDIR", "COMSPEC",
    "TEMP", "TMP", "PATHEXT",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "APPDATA", "LOCALAPPDATA",
    "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES",
    "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS",
    # Linux OS requirements
    "HOME", "USER", "LOGNAME", "SHELL", "TERM",
    "TMPDIR",
    # Locale
    "LANG", "LANGUAGE",
    # Shared
    "PATH",
    # Python runtime
    "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV",
    "PYTHONDONTWRITEBYTECODE", "PYTHONUNBUFFERED",
    "PYTHONIOENCODING", "PYTHONHASHSEED",
})

_ALLOWED_ENV_PREFIXES = ("LC_",)


def _build_clean_env() -> dict:
    """
    Build a minimal environment for the worker subprocess.

    Uses an allowlist: only variables on the approved list are included.
    All other variables — including API keys, tokens, passwords, cloud
    credentials, and database connection strings — are silently dropped.
    """
    clean = {}
    for key, value in os.environ.items():
        key_upper = key.upper()
        if key_upper in _ALLOWED_ENV_KEYS:
            clean[key] = value
        elif any(key_upper.startswith(p) for p in _ALLOWED_ENV_PREFIXES):
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

    # Check AST for dangerous constructs (primary source-level restriction)
    ast_error = _validate_ast(code)
    if ast_error:
        logger.info(
            "Code blocked by AST validation: %s (code_length=%d)",
            ast_error, len(code),
        )
        return SandboxResult(
            success=False,
            stderr=f"Execution blocked: {ast_error}",
            error_type="ASTValidationError",
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
                # Explicitly close non-standard file descriptors
                close_fds=True,
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
