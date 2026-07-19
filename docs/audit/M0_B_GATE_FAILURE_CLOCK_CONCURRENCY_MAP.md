# M0-B Gate, Failure, Clock, and Concurrency Map

Baseline: `main` at `78544d74af84afed450014d599b360c9b4af4f03`

Status: evidence-only audit. M0-B does not change runtime behavior, defaults, persistence, model/vector/device activation, tests, or source control flow.

## Reproducible inventory

Run:

```bash
python scripts/audit/m0_b_controlflow_concurrency_inventory.py --pretty
```

The command scans tracked Python files with `ast.parse` and emits canonical JSON to stdout. `--output` is permitted only for an ephemeral validation artifact. Generated JSON must not be committed.

Every entry records repository-relative path, exact line range, enclosing callable, mechanical evidence, detector, manual classification, confidence, unresolved state, and whether the evidence is manual-only.

## Audit categories

| Category | Mechanical scope | Default classification |
|---|---|---|
| `gate` | `if`, conditional expression, or assertion predicates containing authorization, enablement, validation, safety, or consent signals | `RUNTIME_GATE_CANDIDATE` |
| `bypass` | force, override, bypass, unsafe, or skip-validation predicates, call names, and keyword arguments | `BYPASS_OR_OVERRIDE_CANDIDATE` |
| `output` | stdout/stderr, logging, emit/publish/send/respond surfaces, and generator yields | `OBSERVABLE_OUTPUT_SURFACE` |
| `exception` | all exception handlers, with broadness and silent swallowing distinguished | specific/broad/silent failure-path classifications |
| `clock` | wall clock, monotonic clock, process clock, date-now, and delay calls | `CLOCK_OR_DELAY_DEPENDENCY` |
| `queue` | queue/deque construction and queue-like put/get/join operations | `QUEUE_OR_BUFFER_BOUNDARY` |
| `concurrency` | threads, processes, tasks, executors, locks, events, waits, starts, and joins | `THREAD_PROCESS_TASK_OR_LOCK_BOUNDARY` |
| `nondeterminism` | random, secrets, UUID, urandom, process hash/id, and externally ordered directory/glob operations | nondeterministic or external-order candidate |
| `parse_error` | tracked Python that cannot be parsed | `UNRESOLVED_PARSE_ERROR` |

## Silent-failure rule

An exception handler is mechanically classified as silent when its subtree contains neither a re-raise nor an observable error/logging/diagnostic signal. This is a conservative candidate classification. It does not prove that the caller cannot infer failure through state, return value, or later behavior.

Broad handlers are bare `except`, `Exception`, or `BaseException`. Broad and silent handlers are classified `SILENT_BROAD_EXCEPTION_PATH` and remain unresolved until manual disposition.

## Manually confirmed high-impact surfaces

The scanner applies high-confidence overrides to these already identified runtime funnels:

| Path / callable | Category | Classification | Reason |
|---|---|---|---|
| `main.py` / `build_full_engine` | gate | `ACTIVE_RUNTIME_COMPOSITION_GATE` | Optional runtime capabilities and guarded model integration are composed here. |
| `main.py` / `repl` | output | `ACTIVE_INTERACTIVE_OUTPUT_SURFACE` | Operator commands and visible interactive responses converge here. |
| `adapters/live_loop.py` / `LiveLoop._run` | clock | `ACTIVE_BACKGROUND_CLOCK_DEPENDENCY` | The continuous loop is driven by time and delay operations. |
| `adapters/live_loop.py` / `LiveLoop.start` | concurrency | `ACTIVE_DAEMON_THREAD_BOUNDARY` | The live loop starts a daemon thread. |
| `language/streaming.py` / `StreamingEngine.chat_stream` | exception | `ACTIVE_CHAT_FAILURE_HANDLING_PATH` | Chat state transition and output generation contain failure-handling paths requiring manual disposition. |
| `core/autonomous.py` / `AutonomousLoop.step` | output | `ACTIVE_AUTONOMOUS_OUTPUT_SURFACE` | Autonomous state transition may emit proactive speech or actions. |

## Interpretation rules

- A gate candidate is not automatically a sufficient or correctly placed gate.
- A bypass candidate is not automatically a vulnerability; explicit maintenance and test overrides remain candidates until manually classified.
- Logging is observable output, but logging alone does not prove operational recovery.
- A clock dependency is unresolved until later work decides whether an injected clock is required.
- A concurrency boundary is unresolved until ownership, shutdown, cancellation, ordering, and exception propagation are mapped.
- Randomness and external ordering remain unresolved even when current call sites appear harmless.
- Parse failures are never silently excluded.

## Scope boundary

M0-B does not:

- add, remove, move, strengthen, or weaken any runtime gate;
- activate enforcement, persistence, runtime mapping, network, model, vector, database, sensor, or device behavior;
- change clocks, queues, thread lifecycle, exception handling, logging, output, or fallback behavior;
- delete, skip, xfail, weaken, or rewrite tests;
- commit generated inventory JSON;
- decide M0-C persistence or hormone-to-drive migration;
- decide M0-D module retirement or integrated architecture disposition.

Measured counts and exact high-impact findings will be added only after independent validation against the unchanged branch head.
