"""
Security tests for the sandboxed Python execution module.

These tests verify that:
    1. Normal code executes correctly
    2. Syntax errors are reported
    3. Runtime exceptions are caught
    4. Infinite loops are terminated by timeout
    5. Oversized code is rejected
    6. Oversized output is truncated
    7. Network access is blocked
    8. Environment variable access is blocked
    9. Filesystem access is blocked
    10. Subprocess/shell creation is blocked
    11. AST validation rejects dangerous constructs
    12. Introspection escapes are blocked
    13. Indirect builtin access is blocked
    14. Dynamic import mechanisms are blocked
    15. Indirect filesystem access is blocked
    16. Indirect subprocess access is blocked
    17. Indirect network access is blocked
    18. Secret/environment discovery is blocked
    19. File descriptor inheritance is controlled
    20. Resource exhaustion is handled gracefully

All tests run WITHOUT any API keys or external services.
Run with: python -m pytest tests/test_sandbox.py -v
"""

import unittest
import time
import inspect
from src.sandbox import (
    run_code_in_subprocess,
    SandboxConfig,
    SandboxResult,
    _validate_ast,
    _build_clean_env,
)


# ===========================================================================
# Original test classes (1-10)
# ===========================================================================


class TestSandboxNormalExecution(unittest.TestCase):
    """Test 1: Normal valid Python execution."""

    def test_basic_print(self):
        result = run_code_in_subprocess('print("Hello, World!")')
        self.assertTrue(result.success)
        self.assertIn("Hello, World!", result.stdout)
        self.assertIsNone(result.error_type)

    def test_math_calculation(self):
        result = run_code_in_subprocess('print(847 * 32)')
        self.assertTrue(result.success)
        self.assertIn("27104", result.stdout)

    def test_math_module(self):
        result = run_code_in_subprocess('import math\nprint(math.sqrt(144))')
        self.assertTrue(result.success)
        self.assertIn("12", result.stdout)

    def test_json_module(self):
        code = 'import json\ndata = {"key": "value"}\nprint(json.dumps(data))'
        result = run_code_in_subprocess(code)
        self.assertTrue(result.success)
        self.assertIn('"key"', result.stdout)

    def test_list_comprehension(self):
        result = run_code_in_subprocess('print([x**2 for x in range(5)])')
        self.assertTrue(result.success)
        self.assertIn("[0, 1, 4, 9, 16]", result.stdout)

    def test_multiline_code(self):
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

print(fibonacci(10))
"""
        result = run_code_in_subprocess(code)
        self.assertTrue(result.success)
        self.assertIn("55", result.stdout)

    def test_no_output(self):
        result = run_code_in_subprocess('x = 42')
        self.assertTrue(result.success)
        self.assertEqual(result.stdout.strip(), "")

    def test_execution_time_reported(self):
        result = run_code_in_subprocess('print("timing test")')
        self.assertTrue(result.success)
        self.assertGreater(result.execution_time_ms, 0)


class TestSandboxSyntaxError(unittest.TestCase):
    """Test 2: Syntax error handling."""

    def test_syntax_error(self):
        result = run_code_in_subprocess('def foo(:\n  pass')
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "SyntaxError")
        self.assertIn("SyntaxError", result.stderr)

    def test_incomplete_expression(self):
        result = run_code_in_subprocess('print(1 +')
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "SyntaxError")


class TestSandboxRuntimeException(unittest.TestCase):
    """Test 3: Runtime exception handling."""

    def test_division_by_zero(self):
        result = run_code_in_subprocess('print(1 / 0)')
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ZeroDivisionError")
        self.assertIn("ZeroDivisionError", result.stderr)

    def test_name_error(self):
        result = run_code_in_subprocess('print(undefined_variable)')
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "NameError")

    def test_type_error(self):
        result = run_code_in_subprocess('print("hello" + 42)')
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "TypeError")

    def test_index_error(self):
        result = run_code_in_subprocess('x = [1, 2, 3]\nprint(x[10])')
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "IndexError")


class TestSandboxTimeout(unittest.TestCase):
    """Test 4: Infinite loop -> timeout."""

    def test_infinite_loop_timeout(self):
        config = SandboxConfig(max_execution_seconds=2)
        start = time.monotonic()
        result = run_code_in_subprocess('while True: pass', config=config)
        elapsed = time.monotonic() - start

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "Timeout")
        self.assertIn("timed out", result.stderr)
        self.assertLess(elapsed, config.max_execution_seconds + 3)

    def test_long_sleep_timeout(self):
        config = SandboxConfig(max_execution_seconds=2)
        code = "import time\ntime.sleep(60)\nprint('should not reach here')"
        result = run_code_in_subprocess(code, config=config)
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "Timeout")


class TestSandboxOversizedCode(unittest.TestCase):
    """Test 5: Oversized code -> rejected before execution."""

    def test_oversized_code_rejected(self):
        config = SandboxConfig(max_code_length=100)
        code = "x = 1\n" * 200
        result = run_code_in_subprocess(code, config=config)
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "CodeTooLarge")
        self.assertIn("Code too large", result.stderr)
        self.assertLess(result.execution_time_ms, 100)

    def test_empty_code_rejected(self):
        result = run_code_in_subprocess("")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ValidationError")

    def test_whitespace_only_code_rejected(self):
        result = run_code_in_subprocess("   \n\n  ")
        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "ValidationError")


class TestSandboxOversizedOutput(unittest.TestCase):
    """Test 6: Oversized output -> truncated."""

    def test_oversized_output_truncated(self):
        config = SandboxConfig(max_output_length=500)
        code = 'print("A" * 2000)'
        result = run_code_in_subprocess(code, config=config)
        self.assertTrue(result.success)
        self.assertIn("output truncated", result.stdout)
        self.assertLess(len(result.stdout), 700)


class TestSandboxNetworkBlocked(unittest.TestCase):
    """Test 7: Network access attempts are blocked."""

    def test_socket_blocked(self):
        code = 'import socket\ns = socket.socket()'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("socket", result.stderr.lower())

    def test_requests_blocked(self):
        code = 'import requests\nrequests.get("http://example.com")'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("requests", result.stderr.lower())

    def test_urllib_blocked(self):
        code = 'import urllib.request\nurllib.request.urlopen("http://example.com")'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        stderr_lower = result.stderr.lower()
        self.assertTrue(
            "urllib" in stderr_lower or "open(" in stderr_lower,
            f"Expected 'urllib' or 'open(' in stderr, got: {result.stderr}"
        )

    def test_http_client_blocked(self):
        code = 'import http.client\nconn = http.client.HTTPConnection("example.com")'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("http.client", result.stderr.lower())


class TestSandboxEnvVarBlocked(unittest.TestCase):
    """Test 8: Environment variable / secrets access is blocked."""

    def test_os_environ_blocked(self):
        code = 'import os\nprint(os.environ)'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)

    def test_env_var_not_in_worker_env(self):
        import os
        os.environ["TEST_API_KEY_SANDBOX_CHECK"] = "super_secret_value"
        try:
            clean = _build_clean_env()
            self.assertNotIn("TEST_API_KEY_SANDBOX_CHECK", clean)
        finally:
            del os.environ["TEST_API_KEY_SANDBOX_CHECK"]


class TestSandboxFilesystemBlocked(unittest.TestCase):
    """Test 9: Filesystem access attempts are blocked."""

    def test_open_blocked(self):
        code = 'f = open("/etc/passwd", "r")\nprint(f.read())'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("open(", result.stderr.lower())

    def test_pathlib_blocked(self):
        code = 'import pathlib\np = pathlib.Path(".")\nprint(list(p.iterdir()))'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("pathlib", result.stderr.lower())

    def test_shutil_blocked(self):
        code = 'import shutil\nshutil.rmtree("/tmp/test")'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("shutil", result.stderr.lower())


class TestSandboxSubprocessBlocked(unittest.TestCase):
    """Test 10: Subprocess/shell creation is blocked."""

    def test_subprocess_blocked(self):
        code = 'import subprocess\nsubprocess.run(["ls"])'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("subprocess", result.stderr.lower())

    def test_os_system_blocked(self):
        code = 'import os\nos.system("whoami")'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("os.system", result.stderr.lower())

    def test_exec_blocked(self):
        code = 'exec("print(42)")'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)

    def test_eval_blocked(self):
        code = 'result = eval("1 + 1")\nprint(result)'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)

    def test_dunder_import_blocked(self):
        code = '__import__("os").system("whoami")'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIn("__import__", result.stderr.lower())


class TestSandboxResultStructure(unittest.TestCase):
    """Verify the SandboxResult structure is correct."""

    def test_result_has_all_fields(self):
        result = run_code_in_subprocess('print("test")')
        self.assertIsInstance(result, SandboxResult)
        self.assertIsInstance(result.success, bool)
        self.assertIsInstance(result.stdout, str)
        self.assertIsInstance(result.stderr, str)
        self.assertIsInstance(result.execution_time_ms, float)

    def test_result_to_dict(self):
        result = run_code_in_subprocess('print("test")')
        d = result.to_dict()
        self.assertIn("success", d)
        self.assertIn("stdout", d)
        self.assertIn("stderr", d)
        self.assertIn("error_type", d)
        self.assertIn("execution_time_ms", d)

    def test_error_result_has_error_type(self):
        result = run_code_in_subprocess('print(1/0)')
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error_type)


# ===========================================================================
# New adversarial test classes (11-20)
# ===========================================================================


class TestASTValidator(unittest.TestCase):
    """Test 11: AST validator rejects dangerous constructs directly.

    Tests the _validate_ast function in isolation to verify that:
    - Safe data-analysis patterns are allowed
    - Dangerous introspection/execution patterns are rejected
    - The correct rejection reason is reported
    """

    # --- Safe patterns (must be ALLOWED) ---

    def test_safe_arithmetic(self):
        self.assertIsNone(_validate_ast("x = 1 + 2 * 3"))

    def test_safe_math_sqrt(self):
        self.assertIsNone(_validate_ast("import math\nprint(math.sqrt(25))"))

    def test_safe_list_append(self):
        self.assertIsNone(_validate_ast("data = []\ndata.append(42)"))

    def test_safe_string_methods(self):
        self.assertIsNone(_validate_ast("s = 'hello'\nprint(s.upper())"))

    def test_safe_dict_methods(self):
        self.assertIsNone(_validate_ast("d = {}\nd.update({'a': 1})\nprint(d.keys())"))

    def test_safe_statistics_analysis(self):
        code = "import statistics\ndata = [14, 8, 11]\nprint(statistics.mean(data))"
        self.assertIsNone(_validate_ast(code))

    def test_safe_allowed_imports(self):
        for mod in ["math", "json", "statistics", "re", "collections",
                     "itertools", "functools", "datetime", "random"]:
            self.assertIsNone(_validate_ast(f"import {mod}"))

    def test_safe_isinstance(self):
        self.assertIsNone(_validate_ast("print(isinstance(42, int))"))

    # --- Dunder attribute access (must be REJECTED) ---

    def test_dunder_class_rejected(self):
        result = _validate_ast("x = ().__class__")
        self.assertIsNotNone(result)
        self.assertIn("__class__", result)

    def test_dunder_globals_rejected(self):
        result = _validate_ast("def f(): pass\nf.__globals__")
        self.assertIsNotNone(result)
        self.assertIn("__globals__", result)

    def test_dunder_bases_rejected(self):
        result = _validate_ast("x = int.__bases__")
        self.assertIsNotNone(result)
        self.assertIn("__bases__", result)

    def test_dunder_subclasses_rejected(self):
        result = _validate_ast("x = object.__subclasses__()")
        self.assertIsNotNone(result)
        self.assertIn("__subclasses__", result)

    def test_dunder_code_rejected(self):
        result = _validate_ast("def f(): pass\nf.__code__")
        self.assertIsNotNone(result)
        self.assertIn("__code__", result)

    def test_dunder_mro_rejected(self):
        result = _validate_ast("int.__mro__")
        self.assertIsNotNone(result)
        self.assertIn("__mro__", result)

    def test_dunder_closure_rejected(self):
        result = _validate_ast("def f(): pass\nf.__closure__")
        self.assertIsNotNone(result)
        self.assertIn("__closure__", result)

    def test_dunder_loader_rejected(self):
        result = _validate_ast("x.__loader__")
        self.assertIsNotNone(result)
        self.assertIn("__loader__", result)

    def test_dunder_builtins_rejected(self):
        result = _validate_ast("x = __builtins__")
        self.assertIsNotNone(result)
        self.assertIn("__builtins__", result)

    def test_dunder_dict_rejected(self):
        result = _validate_ast("x.__dict__")
        self.assertIsNotNone(result)
        self.assertIn("__dict__", result)

    def test_dunder_init_subclass_rejected(self):
        result = _validate_ast("int.__init_subclass__")
        self.assertIsNotNone(result)
        self.assertIn("__init_subclass__", result)

    # --- Dangerous names (must be REJECTED) ---

    def test_getattr_rejected(self):
        result = _validate_ast("getattr(obj, 'x')")
        self.assertIsNotNone(result)
        self.assertIn("getattr", result)

    def test_setattr_rejected(self):
        result = _validate_ast("setattr(obj, 'x', 1)")
        self.assertIsNotNone(result)
        self.assertIn("setattr", result)

    def test_hasattr_rejected(self):
        result = _validate_ast("hasattr(obj, 'x')")
        self.assertIsNotNone(result)
        self.assertIn("hasattr", result)

    def test_delattr_rejected(self):
        result = _validate_ast("delattr(obj, 'x')")
        self.assertIsNotNone(result)
        self.assertIn("delattr", result)

    def test_eval_rejected(self):
        result = _validate_ast('eval("1 + 1")')
        self.assertIsNotNone(result)
        self.assertIn("eval", result)

    def test_exec_rejected(self):
        result = _validate_ast('exec("print(42)")')
        self.assertIsNotNone(result)
        self.assertIn("exec", result)

    def test_compile_rejected(self):
        result = _validate_ast('compile("1+1", "", "eval")')
        self.assertIsNotNone(result)
        self.assertIn("compile", result)

    def test_type_rejected(self):
        result = _validate_ast("type(42)")
        self.assertIsNotNone(result)
        self.assertIn("type", result)

    def test_open_rejected(self):
        result = _validate_ast('open("file.txt")')
        self.assertIsNotNone(result)
        self.assertIn("open", result)

    def test_input_rejected(self):
        result = _validate_ast("x = input('Enter: ')")
        self.assertIsNotNone(result)
        self.assertIn("input", result)

    def test_breakpoint_rejected(self):
        result = _validate_ast("breakpoint()")
        self.assertIsNotNone(result)
        self.assertIn("breakpoint", result)

    def test_globals_rejected(self):
        result = _validate_ast("globals()")
        self.assertIsNotNone(result)
        self.assertIn("globals", result)

    def test_locals_rejected(self):
        result = _validate_ast("locals()")
        self.assertIsNotNone(result)
        self.assertIn("locals", result)

    def test_vars_rejected(self):
        result = _validate_ast("vars()")
        self.assertIsNotNone(result)
        self.assertIn("vars", result)

    # --- Import validation ---

    def test_disallowed_import_rejected(self):
        result = _validate_ast("import os")
        self.assertIsNotNone(result)
        self.assertIn("os", result)

    def test_disallowed_from_import_rejected(self):
        result = _validate_ast("from os import path")
        self.assertIsNotNone(result)
        self.assertIn("os", result)

    def test_sys_import_rejected(self):
        result = _validate_ast("import sys")
        self.assertIsNotNone(result)

    def test_builtins_import_rejected(self):
        result = _validate_ast("import builtins")
        self.assertIsNotNone(result)

    # --- Dangerous attributes ---

    def test_load_module_rejected(self):
        result = _validate_ast("loader.load_module('os')")
        self.assertIsNotNone(result)
        self.assertIn("load_module", result)

    def test_find_spec_rejected(self):
        result = _validate_ast("loader.find_spec('os')")
        self.assertIsNotNone(result)
        self.assertIn("find_spec", result)

    def test_find_module_rejected(self):
        result = _validate_ast("loader.find_module('os')")
        self.assertIsNotNone(result)
        self.assertIn("find_module", result)

    # --- Edge cases ---

    def test_syntax_error_passes_validation(self):
        """Syntax errors pass AST validation — worker reports them."""
        self.assertIsNone(_validate_ast("def foo(:"))

    def test_empty_code_passes_validation(self):
        """Empty string is valid Python (no AST nodes)."""
        self.assertIsNone(_validate_ast(""))


class TestIntrospectionEscape(unittest.TestCase):
    """Test 12: Python introspection escape attempts are blocked."""

    def test_tuple_class(self):
        result = run_code_in_subprocess("print(().__class__)")
        self.assertFalse(result.success)
        self.assertIn("__class__", result.stderr)

    def test_tuple_class_bases(self):
        result = run_code_in_subprocess("print(().__class__.__bases__)")
        self.assertFalse(result.success)

    def test_int_mro(self):
        result = run_code_in_subprocess("print(int.__mro__)")
        self.assertFalse(result.success)

    def test_object_subclasses(self):
        result = run_code_in_subprocess(
            "print(().__class__.__bases__[0].__subclasses__())"
        )
        self.assertFalse(result.success)

    def test_function_globals(self):
        code = "def f(): pass\nprint(f.__globals__)"
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)

    def test_function_code(self):
        code = "def f(): pass\nprint(f.__code__)"
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)

    def test_full_jailbreak_chain(self):
        code = (
            "[c for c in ().__class__.__bases__[0].__subclasses__() "
            "if c.__name__ == 'BuiltinImporter']"
        )
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)

    def test_dict_class_chain(self):
        result = run_code_in_subprocess("print({}.__class__)")
        self.assertFalse(result.success)

    def test_string_class_chain(self):
        result = run_code_in_subprocess("print(''.__class__)")
        self.assertFalse(result.success)


class TestIndirectBuiltinAccess(unittest.TestCase):
    """Test 13: Indirect access to dangerous builtins is blocked."""

    def test_getattr_blocked(self):
        result = run_code_in_subprocess("getattr([], 'append')")
        self.assertFalse(result.success)

    def test_setattr_blocked(self):
        result = run_code_in_subprocess("class X: pass\nsetattr(X, 'a', 1)")
        self.assertFalse(result.success)

    def test_hasattr_blocked(self):
        result = run_code_in_subprocess("hasattr([], 'append')")
        self.assertFalse(result.success)

    def test_type_blocked(self):
        result = run_code_in_subprocess("print(type(42))")
        self.assertFalse(result.success)

    def test_chr_string_harmless(self):
        """chr() can build strings, but can't exploit without getattr/eval."""
        code = "s = chr(111) + chr(115)\nprint(s)"
        result = run_code_in_subprocess(code)
        self.assertTrue(result.success)
        self.assertIn("os", result.stdout)

    def test_globals_function_blocked(self):
        result = run_code_in_subprocess("print(globals())")
        self.assertFalse(result.success)

    def test_locals_function_blocked(self):
        result = run_code_in_subprocess("print(locals())")
        self.assertFalse(result.success)

    def test_vars_function_blocked(self):
        result = run_code_in_subprocess("print(vars())")
        self.assertFalse(result.success)

    def test_eval_blocked(self):
        result = run_code_in_subprocess('eval("1+1")')
        self.assertFalse(result.success)

    def test_exec_blocked(self):
        result = run_code_in_subprocess('exec("print(42)")')
        self.assertFalse(result.success)

    def test_compile_blocked(self):
        result = run_code_in_subprocess('compile("1+1", "", "eval")')
        self.assertFalse(result.success)

    def test_isinstance_allowed(self):
        """isinstance is safe — no metaclass access."""
        result = run_code_in_subprocess("print(isinstance(42, int))")
        self.assertTrue(result.success)
        self.assertIn("True", result.stdout)


class TestDynamicImport(unittest.TestCase):
    """Test 14: Dynamic import mechanisms are blocked."""

    def test_importlib_blocked(self):
        result = run_code_in_subprocess("import importlib")
        self.assertFalse(result.success)

    def test_os_import_blocked(self):
        result = run_code_in_subprocess("import os")
        self.assertFalse(result.success)

    def test_sys_import_blocked(self):
        result = run_code_in_subprocess("import sys")
        self.assertFalse(result.success)

    def test_builtins_import_blocked(self):
        result = run_code_in_subprocess("import builtins")
        self.assertFalse(result.success)

    def test_ctypes_import_blocked(self):
        result = run_code_in_subprocess("import ctypes")
        self.assertFalse(result.success)

    def test_from_os_import_blocked(self):
        result = run_code_in_subprocess("from os import path")
        self.assertFalse(result.success)

    def test_dunder_import_call_blocked(self):
        code = '__import__("os")'
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)

    def test_loader_access_blocked(self):
        result = run_code_in_subprocess("x.__loader__")
        self.assertFalse(result.success)

    def test_allowed_import_works(self):
        code = "import statistics\nprint(statistics.mean([1, 2, 3]))"
        result = run_code_in_subprocess(code)
        self.assertTrue(result.success)
        self.assertIn("2", result.stdout)

    def test_allowed_import_re_works(self):
        code = "import re\nm = re.match(r'(\\d+)', '42abc')\nprint(m.group(1))"
        result = run_code_in_subprocess(code)
        self.assertTrue(result.success)
        self.assertIn("42", result.stdout)


class TestIndirectFilesystemAccess(unittest.TestCase):
    """Test 15: Filesystem access through indirect mechanisms is blocked."""

    def test_open_via_name_blocked(self):
        result = run_code_in_subprocess('open("/etc/passwd")')
        self.assertFalse(result.success)

    def test_pathlib_import_blocked(self):
        result = run_code_in_subprocess(
            "import pathlib\npathlib.Path('.').iterdir()"
        )
        self.assertFalse(result.success)

    def test_io_import_blocked(self):
        result = run_code_in_subprocess("import io\nio.open('file.txt')")
        self.assertFalse(result.success)

    def test_introspection_to_open_blocked(self):
        code = "().__class__.__bases__[0].__subclasses__()"
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)

    def test_os_path_blocked(self):
        result = run_code_in_subprocess(
            "import os\nprint(os.path.exists('/etc/passwd'))"
        )
        self.assertFalse(result.success)


class TestIndirectSubprocessAccess(unittest.TestCase):
    """Test 16: Subprocess access through indirect mechanisms is blocked."""

    def test_subprocess_import_blocked(self):
        result = run_code_in_subprocess("import subprocess")
        self.assertFalse(result.success)

    def test_os_system_blocked(self):
        result = run_code_in_subprocess("import os\nos.system('ls')")
        self.assertFalse(result.success)

    def test_os_popen_blocked(self):
        result = run_code_in_subprocess("import os\nos.popen('ls')")
        self.assertFalse(result.success)

    def test_introspection_to_popen_blocked(self):
        code = (
            "[c for c in ().__class__.__bases__[0].__subclasses__() "
            "if 'Popen' in c.__name__]"
        )
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)


class TestIndirectNetworkAccess(unittest.TestCase):
    """Test 17: Network access through indirect mechanisms is blocked."""

    def test_socket_import_blocked(self):
        result = run_code_in_subprocess("import socket")
        self.assertFalse(result.success)

    def test_http_import_blocked(self):
        result = run_code_in_subprocess("import http.client")
        self.assertFalse(result.success)

    def test_xmlrpc_import_blocked(self):
        result = run_code_in_subprocess("import xmlrpc.client")
        self.assertFalse(result.success)

    def test_asyncio_import_blocked(self):
        result = run_code_in_subprocess("import asyncio")
        self.assertFalse(result.success)

    def test_introspection_to_socket_blocked(self):
        code = "().__class__.__bases__[0].__subclasses__()"
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)


class TestSecretDiscovery(unittest.TestCase):
    """Test 18: Secrets and environment variables are not discoverable."""

    def test_os_environ_not_accessible(self):
        result = run_code_in_subprocess("import os\nprint(os.environ)")
        self.assertFalse(result.success)

    def test_allowlist_strips_custom_vars(self):
        """Arbitrary environment variables are not passed to worker."""
        import os
        os.environ["MY_CUSTOM_VAR_sandbox_test"] = "test_value"
        try:
            clean = _build_clean_env()
            self.assertNotIn("MY_CUSTOM_VAR_sandbox_test", clean)
        finally:
            del os.environ["MY_CUSTOM_VAR_sandbox_test"]

    def test_allowlist_strips_secrets(self):
        import os
        os.environ["DATABASE_PASSWORD_sandbox_test"] = "db_secret"
        try:
            clean = _build_clean_env()
            self.assertNotIn("DATABASE_PASSWORD_sandbox_test", clean)
        finally:
            del os.environ["DATABASE_PASSWORD_sandbox_test"]

    def test_allowlist_keeps_path(self):
        clean = _build_clean_env()
        self.assertIn("PATH", clean)

    def test_allowlist_strips_api_keys(self):
        import os
        os.environ["ACME_CORP_API_GATEWAY_TOKEN"] = "token_value"
        try:
            clean = _build_clean_env()
            self.assertNotIn("ACME_CORP_API_GATEWAY_TOKEN", clean)
        finally:
            del os.environ["ACME_CORP_API_GATEWAY_TOKEN"]

    def test_allowlist_strips_cloud_credentials(self):
        import os
        os.environ["AWS_SECRET_ACCESS_KEY"] = "aws_secret"
        try:
            clean = _build_clean_env()
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", clean)
        finally:
            del os.environ["AWS_SECRET_ACCESS_KEY"]


class TestFileDescriptorAccess(unittest.TestCase):
    """Test 19: Inherited file descriptors are not accessible."""

    def test_proc_self_fd_blocked(self):
        result = run_code_in_subprocess(
            "import os\nprint(os.listdir('/proc/self/fd'))"
        )
        self.assertFalse(result.success)

    def test_os_fdopen_blocked(self):
        result = run_code_in_subprocess("import os\nos.fdopen(0)")
        self.assertFalse(result.success)

    def test_close_fds_in_source(self):
        """Verify close_fds=True is present in sandbox source."""
        source = inspect.getsource(run_code_in_subprocess)
        self.assertIn("close_fds=True", source)


class TestResourceExhaustion(unittest.TestCase):
    """Test 20: Resource exhaustion is handled gracefully."""

    def test_large_memory_allocation(self):
        """Excessive memory allocation fails without crashing parent."""
        config = SandboxConfig(max_execution_seconds=5)
        code = 'x = "A" * (10 ** 10)'
        result = run_code_in_subprocess(code, config=config)
        self.assertFalse(result.success)
        self.assertIsInstance(result, SandboxResult)

    def test_recursive_depth_exhaustion(self):
        """Deep recursion hits Python's recursion limit."""
        code = "def f(n): return f(n + 1)\nf(0)"
        result = run_code_in_subprocess(code)
        self.assertFalse(result.success)
        self.assertIsInstance(result, SandboxResult)

    def test_parent_stable_after_exhaustion(self):
        """Parent process remains stable after child resource exhaustion."""
        config = SandboxConfig(max_execution_seconds=2)
        result = run_code_in_subprocess("while True: pass", config=config)
        self.assertFalse(result.success)
        result2 = run_code_in_subprocess('print("still alive")')
        self.assertTrue(result2.success)
        self.assertIn("still alive", result2.stdout)

    def test_large_output_handled(self):
        """Very large output is truncated, not OOM."""
        config = SandboxConfig(max_output_length=500)
        code = 'for i in range(100000): print(i)'
        result = run_code_in_subprocess(code, config=config)
        self.assertLess(len(result.stdout), 1000)


if __name__ == "__main__":
    unittest.main()
