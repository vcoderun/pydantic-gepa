# Candidates And Concurrency

GEPA evaluates many candidate-example pairs. Correct isolation is part of the
optimization contract, not only a performance setting.

## Runtime isolation

`Runtime` accepts a task, candidate scope, required components, optional
isolation strategy, concurrency limit, candidate normalization, and a stable
identity:

```python
runtime = Runtime(
    run_subject,
    required_components=("instructions",),
    max_concurrency=5,
    identity="support-agent-v2",
)
```

Candidate injections use context managers and context variables. A candidate's
instructions or output type are active only within its evaluation scope.

## Application safety

Context isolation does not make an arbitrary client thread-safe. Reduce
concurrency when the subject:

- mutates process-global state;
- uses a non-concurrent local model or database session;
- is constrained by provider rate limits;
- depends on ordering between examples.

## Batch evaluation

The adapter may evaluate examples concurrently while preserving result order
and case identity. Injection must therefore operate at example scope unless a
custom isolation strategy proves batch scope is safe.

## Candidate normalization

Candidate values should be normalized before entering the runtime. The common
component catalog handles raw and JSON-string serialization. Advanced runtimes
may supply a normalization callable, but should preserve stable component names
and text values expected by GEPA.

## Determinism and cache

Concurrency can reveal nondeterministic application behavior. Mark evaluations
nondeterministic when appropriate and do not cache them by default. A stable
runtime `identity` should change whenever behavior relevant to cached output
changes.

## Coupled components

Some components only make sense together, such as a tool description and its
parameter descriptions. Declare coupling in component metadata and use staged
or grouped selection when independent mutation would produce invalid states.
