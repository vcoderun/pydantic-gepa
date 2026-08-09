# Low-Level Adapter

Most users should start with `optimize(...)` or `Optimization.from_examples`.
The low-level adapter exists for optimizer authors, custom evaluation harnesses,
and systems that need direct reflective datasets.

## PydanticGEPAAdapter

`PydanticGEPAAdapter.from_dataset(...)` binds an existing Pydantic Evals dataset,
task, injections, objective, component catalog, concurrency, and optional
recorder.

```python
adapter = PydanticGEPAAdapter.from_dataset(
    dataset=dataset,
    task=run_subject,
    injections=injections,
    objective=ScoreObjective(score_key="accuracy"),
    components=components,
    max_concurrency=5,
)
```

This is intentionally advanced: here the caller owns Pydantic Evals `Dataset`
and `Case` objects. The common API keeps them internal.

## Evaluate a candidate batch

```python
batch = adapter.evaluate(
    cases,
    candidate.to_gepa_dict(),
    capture_traces=True,
)
```

The normalized batch contains ordered scores, outputs, failures, and optional
trajectories.

## Build reflection evidence

```python
reflective = adapter.make_reflective_dataset(
    candidate=candidate.to_gepa_dict(),
    eval_batch=batch,
    components_to_update=["instructions"],
)
```

Evidence is grouped by component and includes normalized case records, feedback,
side information, traces, and failure categories.

## PydanticGEPAOptimizer

The optimizer wraps GEPA invocation and converts backend output into
`PydanticGEPAResult`:

```python
optimizer = PydanticGEPAOptimizer(
    adapter=adapter,
    initial_candidate=candidate,
)

result = optimizer.optimize(
    trainset=cases,
    valset=validation_cases,
    config=config,
)
```

## Recorder seam

Provide a candidate-evaluation recorder to forward normalized batches to an
external evidence system. The recorder receives candidate values, batch cases,
the report envelope, scores, and trajectories. It should not alter evaluation
outcomes.

## ASI construction

`PydanticEvalsASIBuilder` converts selected evaluation records into GEPA
reflective trajectories. `ComponentRecordSelector` and `SampleSelection`
control which records become reflection context. ASI is adapter plumbing; common
API users should reason about evaluation feedback, not build ASI manually.

## Compatibility adapter

`GEPAAdapter` and `EvaluationBatch` support legacy integration boundaries. New
code should use typed adapter and result models rather than expanding loose
callback dictionaries.
