# Evaluation

`Evaluation.output(...)` runs the subject once and then calls the scorer.
`Evaluation.controlled(...)` gives the evaluator a `Context`; it may call
`ctx.run()` zero, one, or many times.

```python
def stability(ctx):
    first = ctx.run()
    second = ctx.run()
    return MetricResult(
        score=compare(first, second),
        feedback="Compared two independent executions.",
        side_info={"first": first, "second": second},
    )
```

`EvaluationConfig` groups task/evaluator failure handling, invalid-score
behavior, cache policy, score bounds, and input/candidate/output/result
validators. NaN, infinity, missing objectives, and nonnumeric metrics never
reach GEPA silently.

The cache key includes candidate, example, runtime, evaluator, stage, and
configuration fingerprints. Nondeterministic evaluation is not cached unless
explicitly allowed.
