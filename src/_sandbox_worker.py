"""
Sandbox worker script — executed as a separate subprocess.

This script is NOT imported by the main application. It is invoked via:
    python src/_sandbox_worker.py <code_file_path>

It reads Python code from the specified file, executes it in a restricted
namespace, captures stdout/stderr, and outputs a JSON result envelope.

Security measures applied in this worker process:
    - Environment variables containing secrets are cleared (best-effort)
    - __builtins__ is restricted to a safe subset (moderate protection)
    - No imports are pre-loaded beyond the minimum needed for the worker itself

Limitations (documented honestly):
    - A determined attacker could bypass __builtins__ restrictions via
      object introspection (e.g., ().__class__.__bases__[0].__subclasses__())
    - There is no OS-level seccomp/namespace/cgroup isolation
    - Network access is not blocked at the OS level
"""

import sys
import io
import json
import os
import time


# ---------------------------------------------------------------------------
# 1. Sanitize environment — remove known secret/sensitive variables
# ---------------------------------------------------------------------------
# This is best-effort: we clear common secret patterns. The parent process
# also avoids passing its full environment, but we defensively clear here too.
_SENSITIVE_ENV_PATTERNS = [
    "API_KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL",
    "PRIVATE_KEY", "AUTH", "DATABASE_URL", "CONNECTION_STRING",
]

def _sanitize_environment():
    """Remove environment variables that look like secrets."""
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
    "print": print,
    "len": len,
    "range": range,
    "int": int,
    "float": float,
    "str": str,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "isinstance": isinstance,
    "type": type,
    "repr": repr,
    "reversed": reversed,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "oct": oct,
    "bin": bin,
    "format": format,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
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
    "True": True,
    "False": False,
    "None": None,
    # Controlled import — only allows whitelisted modules
    "__import__": _controlled_import,
}


# ---------------------------------------------------------------------------
# 3. Unique delimiter for the JSON result envelope
# ---------------------------------------------------------------------------
# We use a unique delimiter so we can separate user stdout from our result.
_RESULT_DELIMITER = "___SANDBOX_RESULT_ENVELOPE_8f3a1b2c___"


# ---------------------------------------------------------------------------
# 4. Main execution
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
