# Autobench Bridge

Pydantic-gepa can run independently. When Autobench is present, optimization
evidence can be recorded beside benchmark runs without making Autobench part of
the optimizer's common API.

## Responsibility split

| pydantic-gepa | Autobench |
| --- | --- |
| candidate generation and injection | immutable experiment records |
| Pydantic Evals execution | semantic observations and assets |
| GEPA reflection and selection | replay, compare, export, and policies |
| normalized optimization result | durable evidence and run lineage |

## Native instrumentation

Autobench owns the concrete integration. Install its optional extra and enable
the native instrumentor in an Autobench benchmark:

```bash
uv add 'autobench[pydantic-gepa]'
```

```python
benchmark = Benchmark("optimize-routing").instrument_all()
```

The instrumentor subscribes to pydantic-gepa's typed event stream. It records
optimizer and engine spans, evaluation evidence, resource budgets, candidate
lineage, and component asset versions without an Autobench import in this
package or handwritten observer wiring.

## Generic recorder hooks

The low-level adapter accepts a `CandidateEvaluationRecorder`. The recorder is
called with the candidate, evaluated batch, normalized report, scores, and
optional trajectories. This remains useful for application-owned sinks that
need evaluation-level callbacks.

```python
adapter = PydanticGEPAAdapter.from_dataset(
    ...,
    recorder=application_recorder,
)
```

For serialized lifecycle events independent of a particular product, use
`callback_observer(callback)`. The legacy `autobench_observer()` helper remains
for one compatibility cycle and emits `DeprecationWarning`; new integrations
should not use it. pydantic-gepa never imports Autobench merely to run an
optimization.

## Semantic mapping

Useful evidence includes:

- candidate and parent ids
- component and tracked-asset references
- objective, constraint, and diagnostic metrics
- evaluation and reflection cost
- selected examples and failure categories
- candidate deltas and Pareto membership
- run, stage, checkpoint, and artifact references

Autobench assigns semantic types and persistence policy. pydantic-gepa retains
typed optimization meaning.

## Replay boundary

Autobench replay can regenerate reports from recorded optimization evidence. It
cannot replay a provider call unless its output was recorded. A pydantic-gepa
checkpoint resumes optimizer execution; an Autobench record replays evidence.
These are related but distinct guarantees.
