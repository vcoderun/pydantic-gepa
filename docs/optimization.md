# Optimization

## One-shot API

Use `optimize(...)` when one function owns a complete run:

```python
result = optimize(
    train=train_examples,
    validation=validation_examples,
    task=run_subject,
    score=score_subject,
    components=components,
    injections=injections,
    reflection="openai:gpt-5-mini",
    budget=50,
    max_concurrency=5,
)
```

The shortcuts `reflection`, `budget`, `run`, and `on_event` construct a
`GEPAConfig`. Do not combine them with an explicit `config=`.

## Reusable pipeline

Use `Optimization.from_examples(...)` when a target is reused from Python, the
CLI, tests, or a larger orchestrator:

```python
from pydantic_gepa import Optimization

pipeline = Optimization.from_examples(
    examples=train_examples,
    val_examples=validation_examples,
    task=run_subject,
    score=score_subject,
    score_key="accuracy",
    components=components,
    injections=injections,
    dataset_name="support-routing",
)

result = pipeline.run(config=config)
```

`PydanticGEPAOptimization` is a compatibility alias for `Optimization`.

## What happens during a run

1. Examples become internal Pydantic Evals cases.
2. The component catalog produces the baseline candidate.
3. GEPA selects examples and components for reflection.
4. Candidate injections scope each task execution.
5. Evaluators return objective, constraint, and diagnostic evidence.
6. Reflection evidence is normalized for the proposer.
7. New candidates preserve parent and generation lineage.
8. Validation selects the best supported candidate.
9. Backend output becomes `PydanticGEPAResult`.

## Explicit initial candidate

Pass an explicit candidate when starting from a persisted or externally
tracked version:

```python
initial = Candidate(
    id="prompt-v7",
    values=components.values(),
    metadata={"source": "asset-registry"},
)

pipeline = Optimization.from_examples(
    examples=train,
    val_examples=validation,
    task=task,
    score=score,
    components=components,
    initial_candidate=initial,
)
```

## Objective selection

The default objective is `ScoreObjective(score_key="score")`. Use a named key
when the evaluator returns several metrics:

```python
from pydantic_gepa import ScoreObjective

objective = ScoreObjective(
    score_key="accuracy",
    direction="maximize",
    failure_score=0.0,
)
```

## Concurrency

`max_concurrency` limits example evaluation. Injection values are context-local,
but the application objects behind them must also tolerate concurrent calls. Set
the value to `1` for mutable or rate-limited subjects.

## Custom backend seam

`optimize_fn` exists for tests and advanced GEPA integration. Most applications
should use the installed backend through typed `GEPAConfig`; untyped backend
keyword bags are a migration path, not the preferred API.
