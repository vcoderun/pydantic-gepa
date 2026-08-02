# Common API

Use `optimize(...)` for a one-shot run and `Optimization.from_examples(...)`
when the configured pipeline must be reused.

```python
pipeline = Optimization.from_examples(
    examples=train,
    val_examples=validation,
    task=run_subject,
    score=score_subject,
    components=components,
    injections=injections,
)
result = pipeline.run(config=config)
```

Training and validation sets are explicit. Reusing training data for validation
requires visible opt-in in the compatibility API. `DataSplit` adds deterministic
partitioning, caps, a test set, and final rescoring.

Components describe candidate values. Injections bind those values to the subject
under evaluation. The one-shot and reusable APIs accept the same injection sequence,
so moving between them does not require rebuilding the runtime wiring.

`MetricResult` carries a scalar score, textual feedback, structured side
information, and a role: objective, constraint, or diagnostic. GEPA receives
the selected scalar while normalized results and reflection retain all metrics.
