# Security: Python Code Execution Sandbox

## Threat Model

ResearchPilot-Agent uses a Plan-and-Execute architecture where an LLM (Planner) generates sub-tasks and an Executor resolves them using tools — including the `execute_python` tool, which runs LLM-generated Python code.

**Primary threat**: The LLM can generate arbitrary Python code. If this code runs without isolation, it can:

- **Crash the application** (infinite loop, segfault, memory exhaustion)
- **Access secrets** (API keys in environment variables or `.env`)
- **Read/write the filesystem** (overwrite app code, read user data)
- **Make network requests** (exfiltrate data, connect to external services)
- **Spawn subprocesses** (execute arbitrary shell commands)
- **Interfere with Streamlit** (corrupt shared `sys.stdout`, break the event loop)

## Three-Tier Isolation Model

The sandbox provides defense-in-depth through three tiers of isolation. Each tier has different strength guarantees. **No single tier is relied upon as the sole security boundary.**

### Tier 1: Python-Level Controls (source-level restrictions)

These controls prevent dangerous code from reaching the Python interpreter:

| Control | Mechanism | Strength |
|---------|-----------|----------|
| **AST validation** | Parse code as AST; reject dunder attributes, dangerous names, non-whitelisted imports | **Strong** — cannot be bypassed by string tricks |
| **String blocklist** | Pattern matching against known dangerous strings | **Moderate** — defense-in-depth; evadable by encoding |
| **Restricted builtins** | Limited `__builtins__` dict (no `getattr`, `setattr`, `hasattr`, `type`) | **Moderate** — enforced at runtime in worker |
| **Import allowlist** | `_controlled_import` only allows whitelisted modules | **Strong** — runtime enforcement at import machinery level |

**AST validation is the primary source-level restriction.** It operates on the parsed syntax tree, not raw text, and cannot be bypassed by string concatenation, encoding tricks, comment injection, or variable naming. The regex blocklist is retained as defense-in-depth only.

### Tier 2: Process-Level Controls (execution isolation)

These controls isolate the execution environment:

| Control | Mechanism | Strength |
|---------|-----------|----------|
| **Subprocess isolation** | Separate PID, memory space, stdout | **Strong** |
| **Wall-clock timeout** | `subprocess.run(timeout=...)` + OS process kill | **Strong** |
| **Environment allowlist** | Only approved env vars passed to worker | **Strong** |
| **File descriptor closure** | `close_fds=True` in subprocess creation | **Strong** |
| **stdin isolation** | `stdin=subprocess.DEVNULL` | **Strong** |
| **Parent stability** | Crash/hang in child cannot kill parent | **Strong** |
| **Structured errors** | All failures return `SandboxResult`, never raise | **Strong** |

### Tier 3: OS-Level Controls (resource/system isolation)

These controls restrict the worker's use of system resources:

| Control | Mechanism | Availability |
|---------|-----------|--------------|
| **CPU time limit** | `resource.RLIMIT_CPU` (30 seconds) | **Linux only** |
| **Virtual memory limit** | `resource.RLIMIT_AS` (512 MB) | **Linux only** |
| **File size limit** | `resource.RLIMIT_FSIZE` (10 MB) | **Linux only** |
| **Core dump disabled** | `resource.RLIMIT_CORE` (0) | **Linux only** |

> **⚠️ Windows limitation**: OS-level resource limits are not available through Python's standard library on Windows. On Windows, the only resource control is the wall-clock timeout. Memory and CPU are not capped. This is a documented limitation, not a bug.

## Platform Differences

| Feature | Linux | Windows |
|---------|-------|---------|
| Subprocess isolation | ✅ Full | ✅ Full |
| Wall-clock timeout | ✅ SIGKILL | ✅ TerminateProcess |
| Resource limits (CPU/memory/file) | ✅ via `resource` module | ❌ Not available |
| `close_fds=True` | ✅ Closes all non-stdio FDs | ✅ No handle inheritance |
| Environment allowlist | ✅ Full | ✅ Full |
| AST validation | ✅ Full | ✅ Full |

## What Is NOT Guaranteed

> **Honest disclaimer**: A subprocess is NOT equivalent to a hardened container or VM sandbox.

The following are **NOT** provided by this implementation:

1. **Container/VM isolation**: The subprocess runs on the same host OS with the same user permissions. There is no Docker, gVisor, or Firecracker boundary.

2. **OS-level system call filtering**: There is no seccomp, AppArmor, or SELinux profile restricting syscalls.

3. **Network namespace isolation**: There is no network namespace or firewall. Network access is blocked by AST validation and import restrictions, not by the OS.

4. **Filesystem mount isolation**: The subprocess can see the entire filesystem accessible to the running user. Protection relies on AST validation blocking `open()` and filesystem modules.

5. **Memory/CPU limits on Windows**: On Windows, there are no resource limits beyond the wall-clock timeout. A subprocess could allocate excessive memory before timeout fires.

6. **Multi-tenant isolation**: This is designed for a single-user application. It does not provide isolation between multiple untrusted users.

## Execution Flow

```
┌──────────────────────────┐         ┌──────────────────────────┐
│   Main Process           │         │   Worker Subprocess      │
│   (Streamlit/LangGraph)  │         │   (_sandbox_worker.py)   │
│                          │         │                          │
│  1. Validate code length │         │  1. Read code from file  │
│  2. Check blocklist      │         │  2. Sanitize env vars    │
│  3. Validate AST         │         │  3. Apply resource limits│
│  4. Write temp file      │         │  4. Restrict builtins    │
│  5. Launch subprocess    │──file──►│  5. exec() in namespace  │
│  6. Enforce timeout      │         │  6. Capture stdout/err   │
│       │                  │         │  7. Output JSON result   │
│       ◄──────────────────│ stdout  │                          │
│  7. Parse JSON result    │         └──────────────────────────┘
│  8. Return SandboxResult │               killed on timeout
│  9. Delete temp file     │
└──────────────────────────┘
```

## Files Involved

| File | Purpose |
|------|---------|
| `src/sandbox.py` | `SandboxConfig`, `SandboxResult`, AST validation, `run_code_in_subprocess()` |
| `src/_sandbox_worker.py` | Worker script executed in the subprocess |
| `src/tools.py` | `execute_python` tool — calls `run_code_in_subprocess()` |

## Timeout and Resource Limits

All limits are defined in `src/sandbox.py` → `SandboxConfig`:

| Limit | Default | Enforcement |
|-------|---------|-------------|
| `max_execution_seconds` | 10 seconds | `subprocess.run(timeout=...)` — **OS-level kill** |
| `max_code_length` | 50,000 characters | Checked before subprocess launch |
| `max_output_length` | 10,000 characters | Truncated after execution |

Additional OS-level limits (Linux only, set in worker):

| Limit | Default | Enforcement |
|-------|---------|-------------|
| CPU time | 30 seconds | `resource.RLIMIT_CPU` |
| Virtual memory | 512 MB | `resource.RLIMIT_AS` |
| File creation size | 10 MB | `resource.RLIMIT_FSIZE` |
| Core dumps | Disabled | `resource.RLIMIT_CORE` |

## Security Audit Log

### Audit: 2026-09-19

Adversarial security review of the sandbox implementation.

**Vulnerabilities found and remediated:**

| # | Vulnerability | Severity | Status |
|---|--------------|----------|--------|
| 1 | `__subclasses__()` introspection escape via class hierarchy traversal | Critical | **Fixed** — AST validation blocks all dunder attribute access |
| 2 | `getattr` in builtins enables arbitrary attribute traversal | Critical | **Fixed** — removed from `_SAFE_BUILTINS`, blocked by AST |
| 3 | `type()` in builtins enables metaclass/hierarchy access | High | **Fixed** — removed from `_SAFE_BUILTINS`, blocked by AST |
| 4 | String blocklist bypass via concatenation/encoding | High | **Fixed** — AST validation is primary (cannot be string-evaded) |
| 5 | `exec`/`eval`/`compile` reachable via introspection | Medium | **Fixed** — AST blocks all introspection paths |
| 6 | Whitelisted module `__globals__` leaking namespace | Medium | **Fixed** — AST blocks all dunder attribute access |
| 7 | Inherited file descriptors accessible to worker | Low-Medium | **Fixed** — explicit `close_fds=True` |
| 8 | Resource exhaustion (memory/CPU beyond timeout) | Medium | **Mitigated** — Linux: `resource` limits; Windows: timeout only |
| 9 | Environment variable exposure via broad blocklist | Low | **Fixed** — switched from blocklist to allowlist approach |

**Remaining limitations (unresolved by design):**

| Limitation | Reason |
|-----------|--------|
| No container/VM isolation | Out of scope for subprocess architecture |
| No seccomp/AppArmor | Requires OS-level configuration |
| No network namespace | Requires container or OS-level isolation |
| No filesystem mount isolation | Requires container or chroot |
| No resource limits on Windows | Python `resource` module not available |
| No multi-tenant isolation | Single-user application by design |

## Recommended Future Improvements

For production deployments handling untrusted input:

1. **Container isolation**: Run the worker in a Docker/Podman container with `--network=none` and read-only filesystem.
2. **Resource limits**: Use cgroups or container resource limits to cap memory and CPU on all platforms.
3. **seccomp profiles**: Restrict available system calls to the minimum needed.
4. **Ephemeral environments**: Create and destroy execution environments per request.
5. **Network firewall**: Use OS-level firewall rules to block outbound connections.

## How to Run the Security Tests

```bash
# Run all security tests
python -m pytest tests/test_sandbox.py -v

# Run a specific test category
python -m pytest tests/test_sandbox.py::TestASTValidator -v
python -m pytest tests/test_sandbox.py::TestIntrospectionEscape -v
python -m pytest tests/test_sandbox.py::TestResourceExhaustion -v

# Run with output visible
python -m pytest tests/test_sandbox.py -v -s
```

The tests do NOT require any API keys or external services.

### Test Categories

| # | Category | What It Verifies |
|---|----------|--------------------|
| 1 | Normal execution | `print()`, math, json, list comprehensions |
| 2 | Syntax errors | Malformed code returns `SyntaxError` |
| 3 | Runtime exceptions | Division by zero, NameError, TypeError, IndexError |
| 4 | Timeout | Infinite loop killed within timeout + buffer |
| 5 | Oversized code | Code exceeding `max_code_length` rejected pre-launch |
| 6 | Oversized output | Output exceeding `max_output_length` truncated |
| 7 | Network access | `socket`, `requests`, `urllib`, `http.client` blocked |
| 8 | Environment variables | Secrets not passed to subprocess |
| 9 | Filesystem access | `open()`, `pathlib`, `shutil` blocked |
| 10 | Subprocess/shell | `subprocess`, `os.system`, `exec()`, `eval()` blocked |
| 11 | **AST validator** | Direct validation of safe/dangerous patterns |
| 12 | **Introspection escape** | `__class__`, `__bases__`, `__subclasses__`, `__globals__` |
| 13 | **Indirect builtins** | `getattr`, `type`, `globals`, `eval`, `compile` |
| 14 | **Dynamic import** | `importlib`, `__loader__`, `builtins` module |
| 15 | **Indirect filesystem** | Introspection chains to `open()` |
| 16 | **Indirect subprocess** | Introspection chains to `Popen` |
| 17 | **Indirect network** | Introspection chains to `socket` |
| 18 | **Secret discovery** | Environment allowlist verification |
| 19 | **File descriptor** | FD inheritance, `close_fds` verification |
| 20 | **Resource exhaustion** | Memory, recursion, parent stability |
