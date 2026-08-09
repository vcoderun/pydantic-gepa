# Workflow

## Choose the API

- One run in one function: `optimize(...)`.
- Reusable Python/CLI target: `Optimization.from_examples(...)`.
- Repeated calls or custom cache/failure policy: explicit `Runtime` and `Evaluation`.
- Separate component ownership or budgets: `Plan` and `Stage`.
- Direct reflective data or optimizer implementation: low-level adapter.

## Common Sequence

```python
component = Component(name="instructions", initial_text=baseline, kind="instructions")
pipeline = Optimization.from_examples(
    examples=train,
    val_examples=validation,
    task=task,
    score=lambda ctx: float(ctx.output == ctx.expected_output),
    components=[component],
    injections=[injection],
)
result = pipeline.run(config=config)
```

Training drives mutation. Validation drives selection. Test data is reserved for final external
validation. Inspect `best_candidate`, `validation_scores`, `candidate_history`, budget, and stop
reason. Persist `stable_dump()`, not raw backend data.

Canonical docs: `docs/getting-started.md`, `docs/optimization.md`, `docs/results.md`.
