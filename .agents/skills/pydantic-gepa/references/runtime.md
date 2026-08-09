# Runtime And State

Build `GEPAConfig` from `ReflectionConfig`, `SelectionConfig`, `MergeConfig`, `BudgetConfig`,
`RunConfig`, `TrackingConfig`, `ProgressConfig`, and `EvaluationSetConfig`. Reject unknown loose
kwargs; use legacy mapping only during migration.

Reflection consumes normalized example/output/expected/metric/feedback/failure/component evidence.
Use a model id, `CallableReflectionModel`, or `PydanticAIReflectionModel`.

Use `Plan` only for actual ordered component ownership, budgets, or validation boundaries. Every
stage reports candidate, score, metric calls, acceptance, history, and checkpoint. Frozen values
must remain unchanged. Final rescore catches cross-component interactions.

Durable runs use `RunConfig` and stable callable ids. Resume validates target, component,
configuration, and stage compatibility. Never disable compatibility merely to consume stale state.

Canonical docs: `docs/configuration.md`, `docs/reflection.md`, `docs/plans.md`, `docs/state.md`.
