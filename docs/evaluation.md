# Evaluation

Evaluation converts application behavior into optimization evidence. The common
`optimize(...)` API builds an output evaluation internally. Use the explicit
API when you need repeated calls, cache control, validators, or multiple metric
roles.

## Output scoring

`Evaluation.output` runs the subject exactly once, then gives the result to the
scorer:

```python
from pydantic_gepa import Candidate, Evaluation, Example, MetricResult, Runtime

runtime = Runtime[str, str](
    lambda text: text.strip().lower(),
    identity="normalize-text-v1",
)

evaluation = Evaluation.output(
    runtime,
    lambda ctx: {
        "accuracy": float(ctx.output == ctx.example.expected_output),
        "length": MetricResult(
            score=float(len(ctx.output or "")),
            role="diagnostic",
        ),
    },
    objective="accuracy",
)

result = evaluation.run(
    Candidate(values={"instructions": "Normalize text."}),
    Example(inputs="  HELLO ", expected_output="hello"),
)
```

## Evaluator-controlled execution

`Evaluation.controlled` gives the evaluator a `Context`. The evaluator may call
`ctx.run()` or `await ctx.arun()` zero, one, or several times:

```python
from pydantic_gepa import Context

def stability(ctx: Context[str, str, None]) -> MetricResult:
    first = ctx.run()
    second = ctx.run()
    return MetricResult(
        score=float(first == second),
        feedback="Compared two independent executions.",
        side_info={"first": first, "second": second},
    )

evaluation = Evaluation.controlled(runtime, stability)
```

This form supports robustness checks, pairwise judges, repeated sampling, and
evaluators that reject an input without invoking the subject.

## Metric roles

`MetricResult` supports three roles:

| Role | Purpose |
| --- | --- |
| `objective` | Drives candidate selection |
| `constraint` | Represents a condition that must remain acceptable |
| `diagnostic` | Preserved for analysis and reflection without becoming the objective |

Use one named objective for GEPA. Preserve other metrics in feedback and side
information instead of collapsing every concern into an unexplained scalar.

```python
MetricResult(
    score=0.75,
    role="objective",
    feedback="Two of eight expected fields were wrong.",
    side_info={"wrong_fields": ["postal_code", "country"]},
)
```

## Failure and validation policy

`EvaluationConfig` controls invalid candidates, task failures, evaluator
failures, infrastructure errors, failure scores, score bounds, caching, and
validation hooks:

```python
from pydantic_gepa import EvaluationConfig

config = EvaluationConfig(
    on_task_error="record",
    on_evaluator_error="raise",
    invalid_score="raise",
    failure_score=0.0,
    min_score=0.0,
    max_score=1.0,
)
```

NaN, infinity, missing objectives, and out-of-range objective values are never
silently forwarded to GEPA.

## Caching

`InMemoryCache` and the `CacheStore` contract support repeated evaluation. The
cache identity includes the candidate, example, runtime, evaluator, stage, and
evaluation configuration. Mark an evaluation `deterministic=False` when output
may vary; nondeterministic evaluation is not cached unless explicitly enabled.

## Traces and artifacts

During controlled evaluation, `Context.capture()` and `Context.acapture()` retain
individual runtime attempts. `Context.artifact()` adds structured evidence for
reflection or external recording. Keep feedback concise and place large or
binary evidence in artifacts or attachments.

## Final rescoring

After search, evaluate a selected candidate against a stable split:

```python
from pydantic_gepa import rescore

final = rescore(evaluation, result.best_candidate, split, policy="validation")
```

Use `arun_rescore` in an async application. Final test data should not influence
candidate mutation or selection.
