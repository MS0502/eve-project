# M0-A Runtime Entrypoint and Mutation Map

Baseline: `main` at `2064448f3b64edaaf6b21a4b6e5e5063a2040593`

Status: evidence-only audit. No runtime activation, persistence activation, default change, test weakening, or generated artifact is committed by M0-A.

## Reproducible mechanical inventory

Run:

```bash
python scripts/audit/m0_a_runtime_inventory.py --pretty
```

The command scans tracked Python files with `ast.parse` and emits canonical JSON to stdout. An operator may pass `--output` only for an ephemeral CI artifact. Generated JSON must not be committed.

Every emitted map entry contains:

- repository-relative `path`;
- exact `line_start` and `line_end`;
- enclosing `callable`;
- `mechanical_evidence` and detector;
- `manual_classification`;
- `confidence`;
- `unresolved`;
- `manual_only`.

The detectors cover module main guards, callable entrypoint candidates, imports, class-like dependency construction, attribute/subscript mutation, filesystem or persistence writes, and thread/process/server execution boundaries. Grep is not used as the sole evidence source.

## Manually confirmed high-impact map

| Path and range | Callable | Evidence | Manual classification | Confidence | Unresolved |
|---|---|---|---|---|---|
| `main.py:22-356` | `build_full_engine` | Constructs and connects the full adapter, engine, loop, store, server, sensory, learning, and diagnostic graph. | `ACTIVE_RUNTIME_COMPOSITION_ROOT` | high | no |
| `main.py:359-361` | `build_minimal_engine` | Constructs a bare `StreamingEngine`. | `ACTIVE_MINIMAL_RUNTIME_COMPOSITION_ROOT` | high | no |
| `main.py:270-279` | `build_full_engine` | Calls guarded medium30k load integration; default remains no-load unless explicit validation and authorization are supplied. | `EXPLICIT_GUARDED_MODEL_LOAD_BOUNDARY` | high | no |
| `main.py:374-741` | `repl` | Creates the full engine and exposes chat, autonomous, sensory, web, server, live-loop, save, and load commands. | `ACTIVE_INTERACTIVE_ENTRYPOINT` | high | no |
| `main.py:419-426` | `repl` | Assigns `~/eve.ckpt` and starts `LiveLoop` automatically for the interactive REPL. | `ACTIVE_DEFAULT_BACKGROUND_START_AND_AUTOSAVE_CONFIGURATION` | high | no |
| `main.py:714-725` | `repl` | `/save` and `/load` instantiate `PersistenceAdapter` and invoke persistence operations. | `EXPLICIT_OPERATOR_PERSISTENCE_BOUNDARY` | high | no |
| `main.py:735-741` | `<module>` | `if __name__ == "__main__"` selects minimal stdin mode or the full REPL. | `ACTIVE_MODULE_ENTRYPOINT` | high | no |
| `language/streaming.py:44-175` | `StreamingEngine.__init__` | Stores injected adapters and constructs language understanding, workspace, System1, System2, planner, generator, task solver, and mutable histories. | `ACTIVE_ENGINE_DEPENDENCY_CONSTRUCTION` | high | no |
| `language/streaming.py:177-362` | `StreamingEngine.chat_stream` | Mutates input, context, attention, learning, history, and adapter state while producing output. | `ACTIVE_CHAT_STATE_TRANSITION_PATH` | high | no |
| `adapters/live_loop.py:115-203` | `LiveLoop._run` | Repeated time-based hormone, salience, autonomy, proactive-output, and autosave cycle. | `ACTIVE_BACKGROUND_STATE_MUTATION_LOOP` | high | no |
| `adapters/live_loop.py:205-212` | `LiveLoop._do_autosave` | Creates `PersistenceAdapter` and calls `save`. | `ACTIVE_AUTOSAVE_WRITE_PATH` | high | no |
| `adapters/live_loop.py:216-237` | `LiveLoop.start` | Sets running state and starts a daemon `threading.Thread`. | `ACTIVE_BACKGROUND_THREAD_START` | high | no |
| `adapters/persistence_adapter.py:54-80` | `PersistenceAdapter.save` | Calls legacy persistence and writes a gzip/pickle sidecar. | `ACTIVE_PERSISTENCE_WRITE_PATH` | high | no |
| `adapters/persistence_adapter.py:84-101` | `PersistenceAdapter.load` | Loads and applies legacy state, then attempts sidecar restoration. | `ACTIVE_PERSISTENCE_RESTORE_PATH` | high | no |
| `adapters/persistence_adapter.py:130-151` | `PersistenceAdapter._load_sidecar` | Reads pickle data and clears engine history when a sidecar is found. | `ACTIVE_RESTORE_MUTATION_SITE` | high | no |
| `core/autonomous.py:70-176` | `AutonomousLoop.step` | Advances internal state, may mutate environment, attribute agency, perform curiosity work, append history, and emit proactive speech. | `ACTIVE_AUTONOMOUS_STATE_TRANSITION_PATH` | high | no |

## Initial architectural findings

1. `main.py` is not a thin launcher. `build_full_engine` is the central dependency-construction root and attaches optional network, sensory, persistence-adjacent, model/vector, autonomous, and server surfaces to one mutable engine object.
2. The full interactive entrypoint starts `LiveLoop` automatically and configures an autosave target. This is existing implementation evidence, not an M0-A activation.
3. Persistence has both explicit operator commands and an automatic loop path. `PersistenceAdapter.save` combines legacy persistence with a separate pickle sidecar.
4. `StreamingEngine.chat_stream`, `LiveLoop._run`, and `AutonomousLoop.step` are major mutation funnels. Later M0-B and M0-C work must map gates, exception swallowing, clocks, concurrency, and persistence semantics around these funnels.
5. The medium30k model path is present but guarded and defaults to no-load. M0-A records the construction boundary without changing authorization or defaults.
6. Multiple methods catch broad exceptions and continue. M0-A records the sites; silent-failure disposition belongs to M0-B.

## Classification rules for mechanically discovered entries

- Exact manually confirmed path/callable pairs use the high-confidence classifications above.
- Module main guards are active entrypoints.
- Other `main`, `repl`, `run`, `start`, `step`, stream, save, and load callables are entrypoint candidates unless manually confirmed.
- Class-like calls are dependency-construction candidates until reviewed.
- Attribute and subscript assignments are in-memory mutation sites.
- Write-capable `open`, serialization, path, filesystem, and persistence-like calls are direct-write sites.
- Thread, process, async runner, and server startup calls are execution boundaries.
- Parse failures remain unresolved and are never silently dropped.

## Scope boundary

M0-A does not:

- enable or disable any runtime feature;
- modify runtime code;
- execute network, device, model, vector, or persistence activation;
- delete, skip, xfail, weaken, or rewrite tests;
- commit generated JSON;
- decide M0-B concurrency or silent-failure policy;
- decide M0-C persistence or hormone-to-drive migration;
- decide M0-D module retirement.
