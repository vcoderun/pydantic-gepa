# Optimize Anything Omni

`pydantic_gepa.experimental.optimize_anything` adapts GEPA Optimize Anything
Omni to the same typed examples, application bindings, evaluation, candidates,
and normalized results used by the standard pydantic-gepa backend.

Use this backend when you need to choose or compose optimization engines. Keep
the standard backend when one direct GEPA run is sufficient. The namespace is
experimental because upstream engine and composition APIs are still evolving;
the common `Example`, `Candidate`, `Component`, evaluation, and result models
remain the package boundary.

## Install

The full extra includes upstream agent-engine dependencies:

```bash
uv add "pydantic-gepa[optimize-anything]"
```

The lighter `integrations` extra is enough for standard GEPA and for custom
Optimize Anything engines that do not need the built-in agent engines.

## First Engine Run

Build the application-facing optimization exactly once. Pydantic Evals stays
inside pydantic-gepa:

```python
from pydantic_gepa import Component, DataSplit, Example, Optimization
from pydantic_gepa.experimental.optimize_anything import (
    Engine,
    OptimizeAnythingConfig,
)

instructions = Component(
    name="instructions",
    initial_text="Classify the request.",
)
data = DataSplit.from_sets(
    train=[Example(name="train-1", inputs="refund order", expected_output="refund")],
    validation=[
        Example(name="val-1", inputs="money back please", expected_output="refund")
    ],
    test=[Example(name="test-1", inputs="return purchase", expected_output="refund")],
)

optimization = Optimization.from_examples(
    data=data,
    task=run_application,
    score=lambda ctx: float(ctx.output == ctx.expected_output),
    components=[instructions],
    injections=[instructions_injection],
    backend="optimize_anything",
    optimization_objective="Maximize routing accuracy.",
    background="The candidate controls the application instructions.",
)

result = optimization.optimize(
    config=OptimizeAnythingConfig(
        engine=Engine.gepa(
            gepa_config,
            candidate_mode="text",
            stop_at_score=1.0,
        ),
        component="instructions",
    )
)
```

`data.test` is held out from every optimizer engine. Pydantic-gepa evaluates
the seed and final candidate on it after optimization and records those calls
separately in `result.scores` and `result.budget`.

The complete executable version is
[`examples/experimental_optimize_anything.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/experimental_optimize_anything.py).
It uses a local Pydantic AI function model and requires no API key.

## Typed Engines

One `Engine` describes one bounded optimizer execution. Constructors expose
engine-specific settings directly; unknown settings cannot hide in a loose
dictionary.

### GEPA

```python
gepa = Engine.gepa(
    GEPAConfig(...),
    candidate_mode="components",
    max_evals=100,
    max_token_cost=5.0,
    max_concurrency=8,
    stop_at_score=0.95,
    output_dir="runs/output",
    run_dir="runs/gepa-state",
)
```

GEPA is the only built-in engine that can optimize a complete
`dict[str, str]` component candidate. Set `candidate_mode="text"` when it must
share a composition with text-only engines.

### AutoResearch

```python
autoresearch = Engine.autoresearch(
    model="claude-sonnet-4-6",
    max_evals=30,
    max_token_cost=8.0,
    max_concurrency=4,
    sandbox=True,
)
```

AutoResearch is text-only. Its CLI, provider credentials, and operating-system
sandbox prerequisites are runtime requirements; installing the Python extra
does not configure external executables or credentials.

### Meta-Harness

```python
meta = Engine.meta_harness(
    model="claude-sonnet-4-6",
    max_iterations=4,
    max_candidates_per_iteration=3,
    max_evals=30,
    max_token_cost=8.0,
)
```

### Best-of-N

```python
sampler = Engine.best_of_n(
    model="provider:model",
    temperature=0.8,
    max_samples=6,
    max_evals=30,
    max_token_cost=4.0,
)
```

### Custom Engine

A custom engine implements the upstream-compatible `run` and
`process_result` contract. It can evaluate through the shared server without
knowing Pydantic Evals:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gepa.oa.engine import Result
from pydantic_gepa.experimental.optimize_anything import (
    Engine,
    EngineResult,
    EvaluationServer,
    OptimizationTask,
)


@dataclass
class CandidateEngine:
    name: str
    candidate: str

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        del task
        score, evidence = server.evaluate_examples(self.candidate, split="val")
        return cast(
            "EngineResult",
            Result(
                best_candidate=self.candidate,
                best_score=score,
                metadata={"validation_evidence": evidence},
            ),
        )

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        del result, output_dir


engine = Engine.custom(
    CandidateEngine("candidate-generator", "Use the improved policy."),
    candidate_mode="text",
    max_evals=10,
    fingerprint="candidate-engine-v1",
)
```

Use `split="train"`, `split="val"`, or `split="all"`. Test examples are
intentionally unavailable through the engine-facing server.

## Composition

Pass one `composition` instead of `engine`. All engines in one composition
must use the same candidate mode.

### Sequential

```python
from pydantic_gepa.experimental.optimize_anything import Sequential

config = OptimizeAnythingConfig(
    composition=Sequential(engines=(first, second, third)),
    component="instructions",
)
```

Engines run in order. The running best candidate seeds the next engine, so a
regressing stage does not replace an earlier improvement.

### Parallel

```python
from pydantic_gepa.experimental.optimize_anything import Parallel

config = OptimizeAnythingConfig(
    composition=Parallel(engines=(first, second), max_workers=2),
    component="instructions",
)
```

Every engine receives the same parent. The result preserves all sibling
branches and deliberately selects no winner.

### Best Of

```python
from pydantic_gepa.experimental.optimize_anything import BestOf

config = OptimizeAnythingConfig(
    composition=BestOf(engines=(first, second), max_workers=2),
    component="instructions",
)
```

Best-of runs sibling engines and selects the greatest normalized search score.

### Vote

```python
from pydantic_gepa.experimental.optimize_anything import Vote

config = OptimizeAnythingConfig(
    composition=Vote(engines=(first, second), max_workers=2),
    component="instructions",
)
```

Vote performs a fair validation rescore after the engine runs. Search scores
remain in `EngineRunSummary.search_score`; vote scores remain in
`selection_score` and `SelectionSummary`. Selection evaluation calls are not
charged to an individual engine.

### Adaptive Sequential

```python
from pydantic_gepa.experimental.optimize_anything import AdaptiveSequential

config = OptimizeAnythingConfig(
    composition=AdaptiveSequential(
        engines=(first, second),
        plateau_evals=5,
        max_evals=30,
        patience=2,
        improvement_epsilon=0.01,
        cycle=True,
    ),
    component="instructions",
)
```

Adaptive sequential shares one evaluation budget and records each engine
slice, score transition, switch, and stop reason.

## Omni Pipeline

Omni is a composition recipe, not a special engine. Explore with `BestOf`,
then continue from the winner with a fresh engine:

```python
from pydantic_gepa.experimental.optimize_anything import BestOf, Pipeline, Single

omni = Pipeline(
    steps=(
        BestOf(engines=(gepa_explorer, autoresearch, meta_harness)),
        Single(engine=gepa_continuation),
    )
)
result = optimization.optimize(
    config=OptimizeAnythingConfig(
        composition=omni,
        component="instructions",
    )
)
```

Each step receives the selected output of the preceding step. `Parallel`
cannot appear directly in a pipeline because it has no selected output; use
`BestOf` or `Vote` at that boundary.

## Candidate Modes

- `components` passes the complete component mapping. GEPA supports this mode.
- `text` unwraps exactly one selected component and merges the returned text
  back into the full candidate.
- Set `component="name"` when a text engine should optimize one component
  while preserving its siblings.
- Mixing text and component engines in one composition fails before any model,
  agent, subprocess, or evaluator work starts.

There is no implicit JSON, multi-file, or prompt flattening format.

## Budgets And Cost

Optimize Anything separates two resources:

| Field | Meaning |
| --- | --- |
| `max_evals` | evaluator calls available to an engine or shared scheduler |
| `max_token_cost` | optimizer/proposer spend cap |
| `evaluation_cost` | optional evaluator-side cost |
| `final_rescore_calls` | vote selection work outside engine budgets |
| `heldout_evaluation_calls` | post-run protected test work |

Missing cost remains `None`; it is not converted to zero. Parallel, sequential,
best-of, and vote engines retain their own budgets. Adaptive sequential owns a
shared evaluation pool while each engine keeps its optimizer-cost cap.

## Durable Runs

Use the common `RunConfig`:

```python
from pydantic_gepa import RunConfig

config = OptimizeAnythingConfig(
    composition=omni,
    component="instructions",
    run=RunConfig(
        id="support-omni",
        directory="runs/support-omni",
        resume="if_exists",
    ),
)
```

Pydantic-gepa checkpoints completed pipeline steps and stores normalized
results. Resume compatibility includes the candidate, datasets, objective,
background, engine declarations, composition, budgets, package version, and
GEPA version. Use `fresh=True` to reset only a directory already owned by
pydantic-gepa. A completed checkpoint without its result artifact is treated
as corruption rather than silently rerun.

## Events And Progress

`TrackingConfig` observers receive the same typed event stream as the standard
backend, plus composition and engine correlation:

```python
from pydantic_gepa.configuration import TrackingConfig

events = []
config = OptimizeAnythingConfig(
    engine=engine,
    component="instructions",
    tracking=TrackingConfig(observers=(events.append,)),
)
```

Events identify run, pipeline, step, branch, engine execution, candidate,
evaluation, selection, budget, checkpoint, and held-out rescore boundaries.
GEPA engines additionally expose iteration, proposal, reflection, merge, and
Pareto detail through the package-owned callback bridge.

## Results

Persist `result.stable_dump()`, not upstream runtime objects:

```python
payload = result.stable_dump()
best = result.best_candidate
score = result.best_score
budget = result.budget
composition = result.composition
```

For composed runs, `result.composition` contains deterministic engine order,
input/output candidate lineage, search and selection scores, branch and step
IDs, adaptive schedules, artifact references, and normalized budgets. Raw GEPA
objects are excluded from stable serialization and checkpoints.

## Legacy Migration

The old experimental call still works for one deprecation cycle:

```python
result = optimization.optimize(config=GEPAConfig(...))
```

It emits `DeprecationWarning` and means exactly
`Single(Engine.gepa(config=legacy_config))`. Migrate to:

```python
result = optimization.optimize(
    config=OptimizeAnythingConfig(
        engine=Engine.gepa(legacy_config),
    )
)
```

The standard `PydanticGEPAOptimizer` and standard `GEPAConfig` API are not
deprecated.

## Failure Checklist

- Install `pydantic-gepa[optimize-anything]` when a built-in agent engine is
  missing.
- Use canonical split name `"val"`, not `"validation"`, in custom engines.
- Add an explicit `component` for text engines when the candidate has multiple
  values.
- Give every engine `max_evals`, `max_token_cost`, or both.
- Do not put an unselected `Parallel` step inside `Pipeline`.
- Use distinct validation data; test data is never available to engines.
- Treat an incompatible resume error as evidence that the run definition
  changed; do not bypass the fingerprint.
