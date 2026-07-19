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

## Validated inventory snapshot

Validated on the M0-B branch after bypass-symbol refinement:

```text
tracked Python files: 580
total entries: 4,063
gate predicates: 386
bypass / override candidates: 37
observable output occurrences: 2,459
exception handlers: 741
  broad handlers: 614
  silent handlers: 597
  silent broad handlers: 525
clock / delay dependencies: 127
queue / buffer boundaries: 6
concurrency boundaries: 23
nondeterminism / external-order candidates: 53
test files: 229 KEEP / 0 REWRITE / 0 RETIRE
unresolved entries: 1,311
parse errors: 2
```

Counts are occurrence-level mechanical evidence, not counts of distinct architectural defects. Test files, legacy snapshots, operator scripts, and historical executables remain in the tracked scan unless they are outside the tracked Python set.

The two parse failures remain explicit unresolved evidence:

- `eve_foundation_v10_2.py` — `'[' was never closed at line 11557`
- `eve_foundation_v12_0.py` — `'[' was never closed at line 11542`

M0-B neither excludes nor repairs those tracked legacy snapshots.

## Audit categories

| Category | Mechanical scope | Default classification |
|---|---|---|
| `gate` | `if`, conditional expression, or assertion predicates containing authorization, enablement, validation, safety, or consent signals | `RUNTIME_GATE_CANDIDATE` |
| `bypass` | force, override, bypass, unsafe, or skip-validation predicates, symbol components, and keyword arguments | `BYPASS_OR_OVERRIDE_CANDIDATE` |
| `output` | stdout/stderr, logging, emit/publish/send/respond surfaces, and generator yields | `OBSERVABLE_OUTPUT_SURFACE` |
| `exception` | all exception handlers, with broadness and silent swallowing distinguished | specific/broad/silent failure-path classifications |
| `clock` | wall clock, monotonic clock, process clock, date-now, and delay calls | `CLOCK_OR_DELAY_DEPENDENCY` |
| `queue` | queue/deque construction and queue-like put/get/join operations | `QUEUE_OR_BUFFER_BOUNDARY` |
| `concurrency` | threads, processes, tasks, executors, locks, events, waits, starts, and joins | `THREAD_PROCESS_TASK_OR_LOCK_BOUNDARY` |
| `nondeterminism` | random, secrets, UUID, urandom, process hash/id, and externally ordered directory/glob operations | nondeterministic or external-order candidate |
| `parse_error` | tracked Python that cannot be parsed | `UNRESOLVED_PARSE_ERROR` |

The bypass detector tokenizes dotted and underscored call symbols. It therefore detects `force_alternative` and `endorse_or_override` without treating `reinforce` or `enforcement` as `force` evidence.

## Silent-failure rule

An exception handler is mechanically classified as silent when its subtree contains neither a re-raise nor an observable error/logging/diagnostic signal. This is a conservative candidate classification. It does not prove that the caller cannot infer failure through state, return value, or later behavior.

Broad handlers are bare `except`, `Exception`, or `BaseException`. Broad and silent handlers are classified `SILENT_BROAD_EXCEPTION_PATH` and remain unresolved until manual disposition.

The scan found 597 silent handlers, including 525 broad silent handlers. High-density paths include runtime smoke orchestration, self-learning, orchestration/context extraction, state debug export, web learning, visualization/server code, and the chat stream. M0-B records these sites; it does not replace their fallback or error policies.

## Manually confirmed high-impact surfaces

The scanner applies high-confidence occurrence classifications within these already identified runtime funnels:

| Path / callable | Category | Classification | Reason |
|---|---|---|---|
| `main.py` / `build_full_engine` | gate | `ACTIVE_RUNTIME_COMPOSITION_GATE` | Optional runtime capabilities and guarded model integration are composed here. |
| `main.py` / `repl` | output | `ACTIVE_INTERACTIVE_OUTPUT_SURFACE` | Operator commands and visible interactive responses converge here. |
| `adapters/live_loop.py` / `LiveLoop._run` | clock | `ACTIVE_BACKGROUND_CLOCK_DEPENDENCY` | The continuous loop is driven by time and delay operations. |
| `adapters/live_loop.py` / `LiveLoop.start` | concurrency | `ACTIVE_DAEMON_THREAD_BOUNDARY` | The live loop starts a daemon thread. |
| `language/streaming.py` / `StreamingEngine.chat_stream` | exception | `ACTIVE_CHAT_FAILURE_HANDLING_PATH` | Chat state transition and output generation contain failure-handling paths requiring manual disposition. |
| `core/autonomous.py` / `AutonomousLoop.step` | output | `ACTIVE_AUTONOMOUS_OUTPUT_SURFACE` | Autonomous state transition may emit proactive speech or actions. |

## Initial architectural findings

1. `LiveLoop` is the primary active clock/queue/thread convergence point. It owns a `queue.Queue`, three `threading.Event` objects, queue put/get/qsize operations, timed waits, daemon-thread start, and thread join.
2. `OpenAIServerAdapter` owns a separate readiness event and background thread; `VisualizerServer` owns another background thread. Legacy `airi_server` code creates and cancels multiple asyncio tasks.
3. The active proactive path has explicit `force` flow through `StreamingEngine.proactive_stream`, the DMN adapter, the legacy DMN, and callers in `main.py`, `LiveLoop`, and `AutonomousLoop`. These are bypass candidates requiring later intent and authorization review, not automatically vulnerabilities.
4. The scan found 53 nondeterminism/order candidates. Current-schema files use `uuid.uuid4`; JSON-native cycle checks use process-local `id`; legacy/tuning executables contain `random` and `numpy.random`; suite runners and cleanup code depend on unsorted filesystem enumeration.
5. The 2,459 output occurrences include logging, print calls, sends/emits, and yields across active, legacy, operator, and test code. Their count is an observability surface measure, not a claim that all outputs are externally exposed.
6. Gate evidence is distributed across runtime loading, virtual-world policy schemas, external-seed validation, operator scripts, and tests. M0-B does not claim those predicates form a coherent authorization hierarchy.

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
