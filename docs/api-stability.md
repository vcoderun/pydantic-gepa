# API Stability

Pydantic-gepa is pre-1.0. Compatibility is managed by surface, not by treating
every import as equally stable.

## Common surface

The intended application-facing vocabulary is:

- `optimize` and `Optimization`
- `Example`, `Attachment`, and `DataSplit`
- `Component`, `ComponentCatalog`, and `Candidate`
- candidate injections
- `Evaluation`, `Runtime`, and `MetricResult`
- `GEPAConfig` and nested typed configuration
- `PydanticGEPAResult`
- `Plan`, `Stage`, `Budget`, and `RunConfig`

Changes to these names should include migration guidance and compatibility where
practical.

## Advanced surface

Adapters, ASI builders, report envelopes, custom cache stores, recorder
contracts, and backend callbacks are public for integration authors but may
evolve faster than the common API.

## Experimental surface

Anything under `pydantic_gepa.experimental` may change in a minor release as its
upstream backend changes. Experimental types still return common candidates and
results where possible.

## Compatibility aliases

`PydanticGEPAOptimization` aliases `Optimization`. Compatibility helpers exist
to migrate earlier code, but new docs use the shortest current names.

## Persistence

Persist candidate YAML and `PydanticGEPAResult.stable_dump()`, not raw GEPA
objects, Pydantic Evals reports, context manager instances, or reflection model
clients.

## Deprecation policy

Deprecated APIs should warn before removal and document the replacement.
Unknown untyped configuration is rejected rather than silently accepted, which
keeps upgrades observable.
