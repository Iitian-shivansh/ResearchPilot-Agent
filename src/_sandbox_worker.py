"""
Sandbox worker script — executed as a separate subprocess.

This script is NOT imported by the main application. It is invoked via:
    python src/_sandbox_worker.py <code_file_path>

It reads Python code from the specified file, executes it in a restricted
namespace, captures stdout/stderr, and outputs a JSON result envelope.

Security measures applied in this worker process:
    - Environment variables containing secrets are cleared (defense-in-depth)
    - __builtins__ is restricted to a safe subset
    - Only whitelisted modules can be imported
    - OS-level resource limits are applied where supported (Linux)
    - No dangerous builtins (getattr, setattr, hasattr, type) are available

Limitations (documented honestly):
    - A determined attacker could potentially bypass Python-level restrictions
      via techniques not yet discovered
    - There is no OS-level seccomp/namespace/cgroup isolation
    - Network access is not blocked at the OS level
    - Resource limits are only available on Linux/POSIX (not Windows)
"""

import sys
import io
import json
import os
import time


# ---------------------------------------------------------------------------
# 1. Sanitize environment — defense-in-depth secret removal
# ---------------------------------------------------------------------------
# The parent process uses an allowlist to construct a minimal environment.
# This blocklist-based cleanup is defense-in-depth: it catches any secrets
# that might slip through if the parent's allowlist is misconfigured.
_SENSITIVE_ENV_PATTERNS = [
    "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
    "PRIVATE_KEY", "AUTH", "DATABASE_URL", "CONNECTION_STRING",
    "ACCESS_KEY", "DSN",
]

def _sanitize_environment():
    """Remove environment variables that look like secrets (defense-in-depth)."""
    keys_to_remove = []
    for key in os.environ:
        key_upper = key.upper()
        for pattern in _SENSITIVE_ENV_PATTERNS:
            if pattern in key_upper:
                keys_to_remove.append(key)
                break
    for key in keys_to_remove:
        del os.environ[key]


# ---------------------------------------------------------------------------
# 2. Build restricted builtins
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 2a. Allowed modules whitelist for controlled __import__
# ---------------------------------------------------------------------------
# Only these modules can be imported by user code. Everything else raises
# ImportError. This is a stronger guarantee than the string blocklist alone,
# because it operates at the import machinery level inside the worker.
_ALLOWED_MODULES = frozenset([
    "math", "json", "time", "re", "collections", "statistics",
    "itertools", "functools", "string", "textwrap",
    "decimal", "fractions", "random", "datetime",
])

_real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

def _controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
    """
    A restricted __import__ that only allows whitelisted modules.
    This prevents user code from importing os, subprocess, socket, etc.
    even if they bypass the string blocklist.
    """
    top_level = name.split(".")[0]
    if top_level not in _ALLOWED_MODULES:
        raise ImportError(
            f"Import of '{name}' is not allowed in the sandbox. "
            f"Allowed modules: {', '.join(sorted(_ALLOWED_MODULES))}"
        )
    return _real_import(name, globals, locals, fromlist, level)


_SAFE_BUILTINS = {
    # Output
    "print": print,
    # Measurement
    "len": len,
    "range": range,
    # Types — only safe constructors, no metaclass access
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    # Numeric operations
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    # Predicates
    "any": any,
    "all": all,
    # Iteration
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    # Type checking (safe — no metaclass/hierarchy access)
    "isinstance": isinstance,
    # String representation
    "repr": repr,
    # Character/number conversions
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "format": format,
    # Exceptions — needed for try/except
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "ZeroDivisionError": ZeroDivisionError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ImportError": ImportError,
    "Exception": Exception,
    # Constants
    "True": True,
    "False": False,
    "None": None,
    # Controlled import — only allows whitelisted modules
    "__import__": _controlled_import,
    # ---------------------------------------------------------------
    # REMOVED for security (see audit 2026-09-19):
    #   getattr  — enables arbitrary attribute traversal
    #   setattr  — enables attribute mutation
    #   hasattr  — enables attribute probing (calls getattr internally)
    #   type     — enables metaclass/hierarchy access
    # ---------------------------------------------------------------
}


# ---------------------------------------------------------------------------
# 3. Unique delimiter for the JSON result envelope
# ---------------------------------------------------------------------------
# We use a unique delimiter so we can separate user stdout from our result.
_RESULT_DELIMITER = "___SANDBOX_RESULT_ENVELOPE_8f3a1b2c___"


# ---------------------------------------------------------------------------
# 4. OS-level resource limits (Linux/POSIX only)
# ---------------------------------------------------------------------------
def _apply_resource_limits():
    """
    Apply OS-level resource limits before executing user code.

    On Linux/POSIX, uses the resource module to set hard limits on:
        - CPU time (30 seconds — safety net above wall-clock timeout)
        - Virtual address space (512 MB)
        - File creation size (10 MB — prevent disk filling)
        - Core dump size (disabled)

    On Windows, equivalent limits are not available through Python's
    standard library. This is a documented limitation in docs/SECURITY.md.
    """
    if sys.platform == "win32":
        return
    try:
        import resource
        # CPU time: 30 seconds hard limit
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
        # Virtual address space: 512 MB
        _512MB = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (_512MB, _512MB))
        # File creation size: 10 MB
        _10MB = 10 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (_10MB, _10MB))
        # Core dumps: disabled
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, ValueError, OSError):
        pass  # Best-effort: not available on this platform


# ---------------------------------------------------------------------------
# 5. Main execution
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        result = {
            "success": False,
            "stdout": "",
            "stderr": "Worker called with wrong number of arguments.",
            "error_type": "WorkerError",
            "execution_time_ms": 0.0,
        }
        print(_RESULT_DELIMITER)
        print(json.dumps(result))
        sys.exit(1)

    code_file_path = sys.argv[1]

    # Read the code
    try:
        with open(code_file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        result = {
            "success": False,
            "stdout": "",
            "stderr": f"Failed to read code file: {e}",
            "error_type": "WorkerError",
            "execution_time_ms": 0.0,
        }
        print(_RESULT_DELIMITER)
        print(json.dumps(result))
        sys.exit(1)

    # Sanitize environment before executing user code
    _sanitize_environment()

    # Apply OS-level resource limits (Linux/POSIX only)
    _apply_resource_limits()

    # Build execution namespace.
    # math and json are pre-loaded for convenience (matches original behavior).
    # User code can also use 'import math' / 'import json' because the
    # controlled __import__ in _SAFE_BUILTINS allows whitelisted modules.
    import math as _math
    import json as _json

    safe_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "math": _math,
        "json": _json,
    }

    # Redirect stdout/stderr to capture user output
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = captured_stdout
    sys.stderr = captured_stderr

    start_time = time.monotonic()
    success = True
    error_type = None

    try:
        exec(code, safe_globals)
    except SyntaxError as e:
        success = False
        error_type = "SyntaxError"
        captured_stderr.write(f"SyntaxError: {e}\n")
    except NameError as e:
        success = False
        error_type = "NameError"
        # Provide a helpful hint for common mistakes
        err_str = str(e)
        if "query_knowledge_base" in err_str or "tavily_search_results_json" in err_str:
            captured_stderr.write(
                f"NameError: {e}\n"
                "Hint: Tool functions cannot be called inside the sandbox. "
                "Pass retrieved text as a string literal instead.\n"
            )
        else:
            captured_stderr.write(f"NameError: {e}\n")
    except Exception as e:
        success = False
        error_type = type(e).__name__
        captured_stderr.write(f"{type(e).__name__}: {e}\n")

    elapsed_ms = (time.monotonic() - start_time) * 1000

    # Restore stdout/stderr for result output
    sys.stdout = old_stdout
    sys.stderr = old_stderr

    stdout_val = captured_stdout.getvalue()
    stderr_val = captured_stderr.getvalue()

    result = {
        "success": success,
        "stdout": stdout_val,
        "stderr": stderr_val,
        "error_type": error_type,
        "execution_time_ms": round(elapsed_ms, 2),
    }

    # Output the result envelope on the real stdout
    print(_RESULT_DELIMITER)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
