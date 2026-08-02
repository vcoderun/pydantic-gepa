# Migration

## Injections

Existing injection classes remain compatibility adapters. Prefer `Runtime`
candidate scopes or Pydantic AI component constructors for new integrations.

## Loose backend options

Replace standalone GEPA keyword arguments with `GEPAConfig`. The legacy parser
warns, maps known keys, and rejects unknown keys.

## Recorder callbacks

Replace candidate/event recorder dictionaries with typed `on_event` callbacks.
Use `RunStore` for durable state rather than treating an observer as a sink.

## Single backend call

One optimization remains valid. Use `Plan` when multiple components or
subjects require deterministic sequential/grouped execution.

## Pydantic Evals

Common callers provide `list[Example]`, task/scorer callables, and optional
custom evaluators. Direct `Dataset`/`Case` construction belongs only in the
advanced compatibility layer.
