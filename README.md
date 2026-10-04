# ResearchPilot Agent

ResearchPilot is a self-correcting research assistant built with **LangGraph**, **Groq**, **Tavily**, **Qdrant**, **Gemini embeddings**, and **Streamlit**.

🌟 **Live demo:** [researchpilot-agent-ask-query.streamlit.app](https://researchpilot-agent-ask-query.streamlit.app/)

The agent accepts a natural-language question, creates a short plan, uses the appropriate tools, drafts an answer, and reviews its own result. It can combine web search, retrieval from a Qdrant knowledge base, and isolated Python calculations.

## What is implemented

- Plan-and-execute agent graph with a planner, executor, tools node, and critic.
- Tavily web search for current or general-purpose research.
- Qdrant vector retrieval for domain-specific documents.
- Gemini embeddings for knowledge-base queries.
- Sandboxed Python execution in a separate subprocess.
- Streamlit UI with expandable planner, tool-call, tool-output, draft, and critic traces.
- CLI entry point for running the agent without Streamlit.
- One critic-driven revision cycle and safeguards against unbounded tool loops.
- Explicit run, critic, and citation statuses so failed verification is not shown as approval.
- Structured tool-result envelopes with evidence identifiers for knowledge-base and calculation outputs.
- Task-scoped `.txt` and `.md` attachments with bounded size/count validation; attachments are temporary prompt context and are never inserted into shared Qdrant.
- Follow-up conversation state in Streamlit, with Quick, Deep, Compare, and Analyze research modes.
- Per-node execution progress and downloadable Markdown reports.

## Architecture

```mermaid
flowchart LR
    Q[User question] --> P[Planner]
    P --> E[Executor]
    E --> T{Tool call?}
    T -->|Web search| W[Tavily]
    T -->|Knowledge query| K[Qdrant + Gemini embeddings]
    T -->|Calculation| S[Python sandbox subprocess]
    W --> E
    K --> E
    S --> E
    E --> C[Critic]
    C -->|Revision requested, once| E
    C -->|Approved| A[Final answer]
```

The executor is configured with `qwen/qwen3.8-27b` through `langchain-groq`. Tool execution is capped at four tool-result rounds, and consecutive tool errors are capped at two. The critic can send the draft back to the executor for one revision.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Streamlit application with follow-up chat, task attachments, modes, progress, and Markdown export |
| `src/agent.py` | LangGraph workflow and retry/safeguard logic |
| `src/contracts.py` | Structured tool results and evidence models |
| `src/tools.py` | Tavily, Qdrant, and Python tools |
| `src/sandbox.py` | Sandbox validation, limits, subprocess orchestration, and result parsing |
| `src/_sandbox_worker.py` | Restricted worker process that executes generated Python |
| `src/main.py` | CLI runner with a textual execution trace |
| `src/attachments.py` | Bounded validation and formatting of temporary task attachments |
| `src/modes.py` | Research mode instructions and tool-budget defaults |
| `src/report.py` | Markdown report export |
| `tests/test_sandbox.py` | Sandbox and security tests |
| `tests/test_agent.py` | Mocked planner, executor, synthesis, and critic graph tests |
| `tests/test_contracts.py` | Tool-result and evidence contract tests |
| `test_connections.py` | Optional Qdrant and Gemini connectivity checks |
| `test_models.py` | Optional Groq model-list diagnostic |
| `evaluate.py` | Runs the sample evaluation questions and rewrites `EVALUATION.md` |
| `docs/SECURITY.md` | Threat model, controls, limitations, and security audit history |
| `demo_traces/` | Example plan/execute/self-correction trace |

## Requirements

- Python 3.10 or newer
- API keys for Groq, Tavily, and Google Gemini
- A Qdrant Cloud collection named `documents` for knowledge-base retrieval
- Windows, macOS, and Linux are supported; the strongest OS-level resource limits are available on Linux

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in the values:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your_qdrant_api_key
```

Never commit `.env` or real credentials. The application loads these variables with `python-dotenv`.

## Qdrant knowledge base

The retrieval tool expects:

- collection name: `documents`
- vector embeddings compatible with `models/gemini-embedding-001`
- payload field `text` containing the indexed chunk
- optional payload fields `document_id` and `chunk_index` for result labels
- `source_hash` and `ingestion_run_id` for lifecycle tracking

The repository includes a local document-ingestion command for `.txt` and `.md` files. It recursively scans a directory, normalizes and chunks text, creates or validates the collection, generates Gemini embeddings, and upserts deterministic Qdrant point IDs. Before replacing a source, it deletes that source's previous chunks, preventing stale content from surviving after a file is edited. Re-running the command for unchanged files is safe.

```bash
python ingest.py ./knowledge_base
```

The command expects `QDRANT_URL`, `QDRANT_API_KEY`, and `GEMINI_API_KEY` in `.env`. Optional settings are available for collection name and chunking:

```bash
python ingest.py ./knowledge_base --collection documents --chunk-size 1200 --chunk-overlap 200 --batch-size 64
```

Each point contains the payload fields used by retrieval:

- `text`: normalized document chunk
- `document_id`: path relative to the ingested directory
- `chunk_index`: zero-based chunk number

PDF ingestion is intentionally not included yet; add a parser and its dependency only when PDF support is required.

Retrieval can be tuned without code changes:

```env
KB_TOP_K=5
KB_SCORE_THRESHOLD=0.0
```

`KB_TOP_K` limits the number of Qdrant results and `KB_SCORE_THRESHOLD` filters low-scoring
matches. The retrieval tool returns structured evidence IDs such as `[KB-1]` with document,
chunk, score, and excerpt metadata.

## Run the application

Start the Streamlit UI:

```bash
streamlit run app.py
```

On Windows, `start.bat` creates no environment by itself; it expects an existing `venv` directory, installs `requirements.txt`, and starts Streamlit:

```bat
start.bat
```

The UI displays the plan, tool calls, tool outputs, draft answer, critic result, and final answer. A query can take a minute on free-tier LLM limits.

The Streamlit UI supports follow-up questions in the same browser session. Choose a research
mode (Quick, Deep, Compare, or Analyze) in the sidebar. Attach up to five UTF-8 `.txt` or
`.md` files (1 MB each, 4 MB total); their contents are supplied only as temporary context for
that task and are not persisted or indexed in Qdrant. The last result can be exported as a
Markdown report containing the question, mode, answer, attachment names, and execution trace.
PDF attachments are intentionally not supported in this phase.

## Run from the CLI

With the virtual environment active:

```bash
python -m src.main "What is the capital of France?"
```

If no question is provided, the CLI prompts interactively. The CLI requires `GROQ_API_KEY`, `TAVILY_API_KEY`, and `GEMINI_API_KEY` to be present, even when a particular question does not use every integration.

## Tests and evaluation

The sandbox tests do not require API keys or external services:

```bash
python -m unittest tests.test_sandbox -v
```

The agent graph also has deterministic tests with injected fake LLMs and tools:

```bash
python -m unittest tests.test_agent -v
```

The ingestion tests use fake embedding and Qdrant clients and do not require API keys:

```bash
python -m unittest tests.test_ingestion -v
```

Retrieval configuration and evidence formatting are covered by mocked tests:

```bash
python -m unittest tests.test_tools -v
```

If `pytest` is installed, the equivalent command is:

```bash
python -m pytest tests/test_sandbox.py -v
```

Run the optional live evaluation only after configuring all services:

```bash
python evaluate.py
```

The evaluation script makes multiple LLM/tool calls, sleeps between questions to reduce rate-limit pressure, and rewrites `EVALUATION.md` with its results. The current checked-in evaluation records successful runs for five sample questions, with critic revisions on some complex questions.

## Sandbox security

Generated Python is not executed inside the Streamlit process. Before launching the worker, the parent process applies:

- AST validation for dangerous names, dunder attributes, and non-approved imports.
- A defense-in-depth string blocklist.
- Code-size, output-size, and wall-clock limits.
- A minimal environment-variable allowlist.

The worker additionally restricts builtins and imports, captures output, and returns a structured result. On Linux it applies CPU, virtual-memory, file-size, and core-dump limits. On Windows, Python's standard library does not provide equivalent resource limits, so the wall-clock timeout is the primary resource control.

This is a subprocess-based defense-in-depth sandbox, not a hardened container or VM. It does not provide seccomp, filesystem mount isolation, network namespaces, or multi-tenant isolation. See [`docs/SECURITY.md`](docs/SECURITY.md) before using it with untrusted workloads.

## Phase one reliability work

The agent graph now separates execution outcomes such as `completed`, `completed_after_revision`,
`critic_unavailable`, `planner_failed`, `executor_failed`, and `synthesis_failed`. If the tool
budget is exhausted, the graph runs a dedicated synthesis step instead of terminating on a tool
call. Tool outputs use a stable JSON envelope and may include evidence IDs such as `[KB-1]` or
`[CALC-1]`, which the critic checks before approval.

This design is intentionally resume-friendly: the graph can be tested without API keys by
injecting fake LLM/tool implementations, while production construction still uses Groq, Tavily,
Gemini, and Qdrant.

## Known limitations

- Groq free-tier rate limits can make requests slow or fail temporarily.
- Tool-call formatting errors from the LLM are retried once with simplified instructions.
- Knowledge-base retrieval depends on Qdrant and Gemini credentials.
- The ingestion pipeline currently supports `.txt` and `.md`; PDF parsing is not included.
- Streamlit browser-level integration tests are not yet included; attachment validation and report generation have deterministic unit coverage.
- Windows does not receive the Linux-only CPU and memory limits.
- Citation checking in `evaluate.py` is heuristic rather than a formal correctness metric.

## Project status and next gaps

The core agent, UI, integrations, sandbox, documentation, ingestion pipeline, structured evidence contracts, and mocked agent test suite are implemented. The most useful next improvements are:

1. Add Streamlit smoke/integration coverage.
2. Improve citation extraction and evaluation beyond heuristics.
3. Add optional PDF ingestion.
4. Consider container-based execution for stronger isolation in production.

## License

See [`LICENSE`](LICENSE).
