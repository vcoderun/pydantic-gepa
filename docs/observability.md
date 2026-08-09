# Events And Observability

Optimization emits typed events independently of any one logging platform.
Observers receive those events and may render progress, persist evidence, or
forward telemetry.

## Event observer

```python
from pydantic_gepa import Event

events: list[Event] = []

result = optimize(
    ...,
    on_event=events.append,
)
```

Events identify the run, stage, kind, message, progress, candidate, score,
budget, and structured payload when available.

## Rich progress

```python
from pydantic_gepa.observers import rich_progress

result = plan.run(on_event=[rich_progress()])
```

Rich rendering is a presentation observer. It does not own optimization state.

## Logfire

```python
from pydantic_gepa.observers import logfire_observer

observer = logfire_observer()
result = plan.run(on_event=[observer])
```

Install the `logfire` extra and configure Logfire in the application. The
observer adds optimization events to the existing telemetry environment; it
does not configure credentials or globally instrument Pydantic AI.

## Reflection records

`CallableReflectionModel` and `PydanticAIReflectionModel` expose normalized
records including duration, retries, usage, cost, and error state. These records
help distinguish evaluation cost from reflection cost.

## Backend callbacks

`TrackingConfig.backend_callbacks` forwards supported GEPA callbacks. Use typed
package observers for portable behavior and backend callbacks only when a GEPA
feature has no normalized event yet.

## External recorders

`CandidateEvaluationRecorder`, `OptimizationEventRecorder`, and
`GEPAEventBridge` define package boundaries for systems such as Autobench. A
recorder should retain immutable evidence and must not modify candidate scores.

## Failure policy

Observer errors are configured separately through
`TrackingConfig.observer_errors`. Production runs should make the choice
explicit: telemetry loss may be non-fatal, while durable evidence recording may
need fail-fast behavior.
