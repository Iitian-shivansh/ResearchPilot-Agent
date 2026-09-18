# Security: Python Code Execution Sandbox

## Threat Model

ResearchPilot-Agent uses a Plan-and-Execute architecture where an LLM (Planner) generates sub-tasks and an Executor resolves them using tools — including the `execute_python` tool, which runs LLM-generated Python code.

**Primary threat**: The LLM can generate arbitrary Python code. If this code runs in the main application process without isolation, it can:

- **Crash the application** (infinite loop, segfault, memory exhaustion)
- **Access secrets** (API keys in environment variables or `.env`)
- **Read/write the filesystem** (overwrite app code, read user data)
- **Make network requests** (exfiltrate data, connect to external services)
- **Spawn subprocesses** (execute arbitrary shell commands)
- **Interfere with Streamlit** (corrupt shared `sys.stdout`, break the event loop)

## Why Direct `exec()` Is Unsafe

The original implementation used:

```python
exec(code, safe_globals)
```

inside the main Streamlit/LangGraph process. Problems:

1. **No process isolation**: A crash or infinite loop kills the entire app.
2. **String blacklist only**: The `BLOCKED_KEYWORDS` list is trivially bypassed (e.g., `getattr(__builtins__, '__imp' + 'ort__')('os')`).
3. **Shared `sys.stdout`**: Redirecting stdout in the same process is not thread-safe and interferes with Streamlit.
4. **No hard timeout**: There is no way to kill an `exec()` that enters an infinite loop without killing the process itself.
5. **Full environment access**: The code has access to all environment variables, file descriptors, and memory in the parent process.

## New Architecture

### Subprocess Isolation

Generated code now runs in a **separate Python subprocess**:

```
┌──────────────────────────┐         ┌──────────────────────────┐
│   Main Process           │         │   Worker Subprocess      │
│   (Streamlit/LangGraph)  │         │   (_sandbox_worker.py)   │
│                          │         │                          │
│  execute_python(code)    │         │  1. Read code from file  │
│       │                  │         │  2. Sanitize env vars    │
│       ▼                  │         │  3. Restrict builtins    │
│  run_code_in_subprocess()│ ──────► │  4. exec() in namespace  │
│       │                  │  file   │  5. Capture stdout/err   │
│       │                  │  +      │  6. Output JSON result   │
│       ◄────────────────  │ stdout  │                          │
│       │                  │         └──────────────────────────┘
│  Parse JSON result       │               killed on timeout
│  Return string to agent  │
└──────────────────────────┘
```

### Execution Flow

1. `execute_python(code)` is called by LangGraph's `ToolNode`.
2. Pre-flight checks: empty code, code length, blocked patterns.
3. Code is written to a temp file.
4. A subprocess is launched with `subprocess.run(timeout=...)`.
5. The worker script:
   - Strips secret environment variables
   - Builds a restricted `__builtins__` namespace
   - Executes the code
   - Captures stdout/stderr
   - Outputs a JSON result envelope
6. The parent parses the result or handles timeout/crash.
7. Temp file is deleted.
8. A string result is returned to the agent (same format as before).

### Files Involved

| File | Purpose |
|------|---------|
| `src/sandbox.py` | `SandboxConfig`, `SandboxResult`, `run_code_in_subprocess()` |
| `src/_sandbox_worker.py` | Worker script executed in the subprocess |
| `src/tools.py` | `execute_python` tool — calls `run_code_in_subprocess()` |

## Timeout and Resource Limits

All limits are defined in `src/sandbox.py` → `SandboxConfig`:

| Limit | Default | Enforcement |
|-------|---------|-------------|
| `max_execution_seconds` | 10 seconds | `subprocess.run(timeout=...)` — **OS-level kill** |
| `max_code_length` | 50,000 characters | Checked before subprocess launch |
| `max_output_length` | 10,000 characters | Truncated after execution |

These are the **only** places where limits are defined. To change them, modify `SandboxConfig` defaults or pass a custom config instance.

## Isolation Guarantees

### What IS guaranteed

| Property | Mechanism | Strength |
|----------|-----------|----------|
| **Process isolation** | Separate subprocess (separate PID, memory space) | **Strong** |
| **Timeout enforcement** | `subprocess.run(timeout=...)` + OS process kill | **Strong** |
| **Parent stability** | Crash/hang in child cannot kill parent | **Strong** |
| **stdout separation** | Child has its own stdout, no interference with Streamlit | **Strong** |
| **Structured errors** | All failures return `SandboxResult`, never raise to caller | **Strong** |

### What is best-effort

| Property | Mechanism | Limitation |
|----------|-----------|------------|
| **Restricted builtins** | Limited `__builtins__` dict | Can be bypassed via `().__class__.__bases__[0].__subclasses__()` or similar object introspection |
| **Blocked patterns** | String-matching blocklist | Can be evaded with string concatenation, encoding tricks, etc. |
| **Environment sanitization** | Remove vars matching secret patterns | Unusual secret variable names may not be caught |
| **No network access** | Blocked by builtins + blocklist | No OS-level firewall; a bypass of builtins could make network calls |
| **No filesystem access** | `open()` not in builtins + blocklist | Same caveat as network — bypassable if builtins restriction is circumvented |
| **No subprocess creation** | `subprocess` not in builtins + blocklist | Same caveat |

## What Is NOT Guaranteed

> **Honest disclaimer**: A subprocess is not equivalent to a hardened container sandbox.

The following are **NOT** provided by this implementation:

1. **Container/VM isolation**: The subprocess runs on the same host OS with the same user permissions. There is no Docker, gVisor, or Firecracker boundary.

2. **OS-level system call filtering**: There is no seccomp, AppArmor, or SELinux profile restricting syscalls. A builtins bypass could make any syscall the OS user has permission for.

3. **Network namespace isolation**: There is no network namespace or firewall. Network access is blocked by the builtins restriction, not by the OS.

4. **Resource limits (CPU/memory)**: There is no cgroup, `ulimit`, or memory cap. A subprocess could theoretically allocate excessive memory before the timeout fires.

5. **Filesystem mount isolation**: The subprocess can see the entire filesystem accessible to the running user. Protection relies on `open()` not being in the safe builtins.

6. **Multi-tenant isolation**: This is designed for a single-user application. It does not provide isolation between multiple untrusted users.

## Remaining Security Risks

1. **Determined attacker bypass**: An attacker who can control the LLM output could craft code that bypasses the restricted builtins using Python introspection (e.g., walking the class hierarchy to find `os` or `subprocess`).

2. **Resource exhaustion**: Memory-intensive code could impact the host before the timeout fires. CPU-bound code will consume one core for up to `max_execution_seconds`.

3. **Timing attacks**: Execution time is reported. An attacker could infer information about the host from timing variations.

4. **Temp file race conditions**: The code is written to a temp file briefly. On a shared system, another user could theoretically read it.

## Recommended Future Improvements

For production deployments handling untrusted input:

1. **Container isolation**: Run the worker in a Docker/Podman container with `--network=none` and read-only filesystem.
2. **Resource limits**: Use cgroups or container resource limits to cap memory and CPU.
3. **seccomp profiles**: Restrict available system calls to the minimum needed.
4. **Ephemeral environments**: Create and destroy execution environments per request.

## How to Run the Security Tests

```bash
# Run all security tests
python -m pytest tests/test_sandbox.py -v

# Run a specific test category
python -m pytest tests/test_sandbox.py::TestSandboxTimeout -v
python -m pytest tests/test_sandbox.py::TestSandboxNetworkBlocked -v

# Run with output visible
python -m pytest tests/test_sandbox.py -v -s
```

The tests do NOT require any API keys or external services. They test the sandbox module directly.

### Test Categories

| # | Category | What It Verifies |
|---|----------|-----------------|
| 1 | Normal execution | `print()`, math, json, list comprehensions |
| 2 | Syntax errors | Malformed code returns `SyntaxError` |
| 3 | Runtime exceptions | Division by zero, NameError, TypeError, IndexError |
| 4 | Timeout | Infinite loop killed within timeout + buffer |
| 5 | Oversized code | Code exceeding `max_code_length` rejected pre-launch |
| 6 | Oversized output | Output exceeding `max_output_length` truncated |
| 7 | Network access | `socket`, `requests`, `urllib`, `http.client` blocked |
| 8 | Environment variables | Secrets stripped from subprocess environment |
| 9 | Filesystem access | `open()`, `pathlib`, `shutil` blocked |
| 10 | Subprocess/shell | `subprocess`, `os.system`, `exec()`, `eval()`, `__import__` blocked |
