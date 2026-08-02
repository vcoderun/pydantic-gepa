# API Stability

`pydantic-gepa` exposes a small common vocabulary and keeps lower-level
integration contracts in their owning modules.

## Common API

The intended application-facing names are:

- `Candidate`
- `Budget`
- `CaseResult`
- `CacheStore`
- `Component`
- `Context`
- `DataSplit`
- `Evaluation`
- `EvaluationConfig`
- `Example`
- `GEPAConfig`
- `InMemoryCache`
- `MetricResult`
- `MetricRole`
- `Optimization`
- `OptimizationResult`
- `Plan`
- `PlanResult`
- `Runtime`
- `RunConfig`
- `Stage`
- `StageResult`
- `CallableReflectionModel`

`Optimization.run()` is the preferred execution spelling.

## Supported Python Versions

The `0.1` alpha supports and continuously validates Python 3.11, 3.12, and
3.13. Python 3.10 is outside the package's declared `>=3.11` baseline. Python
3.14 will be declared only after the complete GEPA, Pydantic AI, and Pydantic
Evals integration matrix is qualified; it is not claimed based on import-only
testing.

## Compatibility API

The following existing names remain supported while the common API evolves:

- `CandidateComponent`
- `PydanticGEPAOptimization`
- `PydanticGEPAResult`
- `PydanticGEPAOptimizer`
- `PydanticGEPAAdapter`
- candidate injection classes
- recorder contracts

Compatibility names retain their current behavior. Their presence does not
make them the preferred API for new applications.

## Advanced API

Advanced modules expose typed integration seams for:

- GEPA adapters and backend result normalization
- Pydantic Evals harness behavior
- candidate injection internals
- reflective dataset construction
- schema component extraction and application
- Optimize Anything experimental support
- staged and grouped optimization snapshots, runners, and aggregation callbacks
- reflection model adaptation and normalized reflection usage records

Advanced public contracts remain usable without being part of the common root
vocabulary. Experimental APIs remain under `pydantic_gepa.experimental`.

The remaining names currently exported from the root package are classified as
advanced while compatibility migration is in progress:

- ASI and reflection: `ASIBuilder`, `ComponentRecordSelector`,
  `PydanticEvalsASIBuilder`, `PydanticEvalTrajectory`, `SampleSelection`
- adapters and harnesses: `DataInstT`, `EvaluationBatch`, `GEPAAdapter`,
  `OptimizeFn`, `PydanticEvalsHarness`, `RolloutOutputT`, `run_awaitable_sync`
- component selection: `ComponentCatalog`, `ComponentKind`,
  `ComponentSelector`, `SelectorMode`, `SerializationMode`
- evaluation: `EvalContextView`, `EvaluationContext`, `EvaluationOutput`,
  `EvaluationReasonView`, `EvaluationScalar`, `EvaluationValue`,
  `ObjectiveDirection`, `OptimizationBackend`, `PydanticEvaluator`,
  `RescoreResult`, `ScoreFunction`, `ScoreObjective`, `ScoreOutput`,
  `arun_rescore`, `model_field_accuracy`, `rescore`
- schema components: `ModelSchemaCandidate`, `SchemaComponentTarget`,
  `SchemaDescription`, `ToolDefinitionView`, `ToolSchemaCandidate`,
  `apply_model_schema_candidate`, `apply_tool_schema_candidate`,
  `collect_model_components`, `collect_tool_components`,
  `collect_toolset_components`, `description_key`, `format_schema_path`,
  `iter_model_descriptions`, `iter_schema_descriptions`, `parameter_key`,
  `parse_schema_path`, `set_schema_description`
- errors: `CandidateComponentError`, `CandidateInjectionError`,
  `EvaluationHarnessError`, `EvidenceEncodingError`, `InfrastructureError`,
  `InvalidScoreError`, `OptimizationDependencyError`, `PydanticGEPAError`
- orchestration errors: `PlanError`
- result normalization: `CandidateSummary`, `result_from_gepa`
- metadata: `MetricSideInfoValue`, `__version__`

## Change Policy

- Stable common names require a documented migration path before removal.
- Compatibility names may be deprecated only after their replacement covers
  the same behavior.
- Advanced contracts follow semantic versioning but may evolve more quickly
  than the common API.
- Experimental contracts may change between minor releases.
- No compatibility alias may silently change score, candidate, dataset, or
  result semantics.
