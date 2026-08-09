# Checkpoint And Resume

Durable execution belongs to an explicit run directory. It stores enough state
to resume compatible work without pretending that changed code is the same run.

```python
from pydantic_gepa import RunConfig

run = RunConfig(
    id="support-2026-08-09",
    directory="runs/support-2026-08-09",
    resume="if_exists",
    checkpoint_interval=10,
    seed=17,
)

result = pipeline.run(config=config.model_copy(update={"run": run}))
```

## Resume modes

- `never`: start without loading prior state.
- `if_exists`: resume compatible state when present.
- `required`: fail unless compatible resumable state exists.

`fresh=True` requests a fresh owned run and conflicts with resume behavior that
requires old state.

## Compatibility

Run manifests fingerprint the optimization definition, component set,
callbacks, stages, scoring identity, and relevant configuration. Compatibility
checks prevent stale checkpoints from being applied after meaningful code or
configuration changes.

Give reusable callables stable ids such as `run_id`, `rescore_id`, and runtime
`identity`. Lambdas and renamed closures may be unsuitable for durable resume
because their identity is harder to prove.

## Atomic state

Manifest, checkpoint, result, and event writes use owned run paths and atomic
replacement. An interrupted write must not leave a partially valid result.

## Required resume example

```python
first = plan.run(run=RunConfig(id="demo", directory="runs/demo"))

same = plan.run(
    run=RunConfig(
        id="demo",
        directory="runs/demo",
        resume="required",
    )
)

assert same == first
```

## What resume does not promise

Resume does not make external model calls deterministic, reconstruct deleted
provider data, or permit incompatible code to reuse old state. It restores the
last compatible package-owned execution boundary.
