# Changelog

## Unreleased

- Added the typed Optimize Anything Omni engine, composition, result, budget, and event contracts.
- Added common ordered lifecycle events for standard GEPA, Optimize Anything engines and
  compositions, staged plans, evaluation cases, selections, budgets, checkpoints, and terminal
  outcomes, including occurrence timestamps and execution correlation.
- Added typed resource-specific evaluation, optimizer, and aggregate cost evidence plus normalized
  candidate/component lineage for external evidence systems.
- Deprecated `autobench_observer()` in favor of Autobench's native pydantic-gepa instrumentor or
  the backend-neutral `callback_observer()` helper.

All notable changes to this project are documented here. The project follows
Semantic Versioning; experimental APIs may change between minor releases.

## 0.1.0a0

Initial alpha release.

- Added the `Example`-first `optimize(...)` and `Optimization` APIs with
  internally managed Pydantic Evals execution.
- Added typed candidates, components, evaluation evidence, reflection models,
  GEPA configuration, normalized results, and candidate lineage.
- Added context-local, serialized, and factory-isolated runtimes.
- Added output-scoring and evaluator-owned execution with multimodal evidence,
  caching, validation, and final rescoring.
- Added sequential and grouped plans with carry-forward, frozen components,
  shared budgets, checkpoints, resume, and fresh-run behavior.
- Added typed events, Rich progress, optional Logfire and Autobench observers,
  a Click CLI, and versioned consumer contract schemas.
- Added the experimental `pydantic_gepa.experimental.optimize_anything`
  backend using the same typed configuration and canonical candidate codecs.
- Added one-shot candidate injection support and raw-text candidate serialization
  as the default GEPA-facing contract, while retaining explicit JSON-string codecs.
- Added compatibility filtering for typed GEPA options across installed backend
  signatures, with actionable errors for unsupported non-default settings.
- Added deterministic, live GEPA examples, strict documentation builds, a complete
  Python 3.11-3.13 quality matrix, and release CI.
