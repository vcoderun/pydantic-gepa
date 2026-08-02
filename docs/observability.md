# Events And Observability

Every lifecycle notification is a discriminated `Event` with a stable `kind`,
run/stage/candidate identifiers, sequence number, and event-specific payload.

Observers are ordinary typed callables. `compose_observers(...)` controls
observer failure behavior. Built-in adapters include:

- `rich_progress(...)` for interactive bars and deterministic noninteractive
  status lines;
- `logfire_observer(...)` for structured Logfire events;
- `autobench_observer(...)` for an optional recorder contract;
- `callback_observer(...)` for serialized payload consumers.

Events are observational. `RunStore` remains the source of truth for resume and
result persistence.
