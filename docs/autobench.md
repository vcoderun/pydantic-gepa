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

## Recorder hook

The low-level adapter accepts a `CandidateEvaluationRecorder`. The recorder is
called with the candidate, evaluated batch, normalized report, scores, and
optional trajectories. An Autobench adapter can convert these into observations
and artifacts.

```python
adapter = PydanticGEPAAdapter.from_dataset(
    ...,
    recorder=autobench_recorder,
)
```

Keep the bridge in an integration layer. pydantic-gepa should not import
Autobench merely to run an optimization.

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
