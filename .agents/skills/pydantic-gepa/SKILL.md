---
name: pydantic-gepa
description: Design, implement, run, debug, review, or document typed GEPA optimizations with pydantic-gepa. Use for Example datasets, components and candidates, Pydantic AI instruction or structured-output injection, evaluation and scoring, reflection, typed GEPA configuration, candidate lineage, staged plans, concurrency, checkpoint/resume, Optimize Anything, Autobench recording, CLI targets, or pydantic-gepa repository development.
---

# pydantic-gepa

Use pydantic-gepa to optimize typed Pydantic applications while keeping GEPA and Pydantic Evals
plumbing behind a compact common API. Keep this file as the router; load only the reference and
repository example required by the task.

## Start Here

1. Identify what may change: instructions, prompt, tool description, output field descriptions,
   routing policy, or a coupled component group.
2. Identify what proves improvement: explicit train and validation examples, objective,
   constraints, diagnostics, and protected final data.
3. Read [references/workflow.md](references/workflow.md) for new optimizations.
4. Load only the relevant topic reference below.
5. Start from the nearest runnable `examples/*.py` program.
6. Prefer `optimize(...)` or `Optimization.from_examples(...)`; use direct Pydantic Evals and
   adapter types only for advanced integration work.
7. Validate result lineage and held-out evidence before suggesting promotion.

## Non-Negotiable Contracts

- Common users provide `list[Example]`; Pydantic Evals `Dataset` and `Case` stay internal.
- A `Component` describes a mutable dimension; a `Candidate` is one version of all values.
- Candidate values reach the application through scoped, concurrency-safe injections.
- Train and validation data are explicit. Protected test data must not drive mutation.
- Configuration is typed with `GEPAConfig`; do not grow arbitrary GEPA `**kwargs`.
- Preserve objective, constraint, diagnostic, feedback, and side information separately.
- `PydanticGEPAResult` and `stable_dump()` are the package boundary; do not persist raw GEPA
  objects.
- `Plan` is for real stage ownership, budget, or validation boundaries, not cosmetic layering.
- Standard GEPA and experimental Optimize Anything Omni share candidates, evaluation, and
  normalized results; select Omni engines and compositions through typed experimental models.
- pydantic-gepa does not deploy prompts, overwrite source files, or own promotion policy.

## Reference Router

- **New optimization, API selection, datasets, scoring, and result inspection**:
  [references/workflow.md](references/workflow.md)
- **Components, candidate naming, injection, tools, and output schemas**:
  [references/components.md](references/components.md)
- **Evaluation strategies, feedback, caching, repeated calls, and concurrency**:
  [references/evaluation.md](references/evaluation.md)
- **GEPA config, reflection, staged plans, budgets, and checkpoint/resume**:
  [references/runtime.md](references/runtime.md)
- **Pydantic AI, multimodal inputs, Optimize Anything, Autobench, and Autoptimize**:
  [references/integrations.md](references/integrations.md)
- **CLI, troubleshooting, repository map, tests, docs, and release gates**:
  [references/operations.md](references/operations.md)

## Example Router

- `examples/basic.py`: shortest instruction optimization through the common API.
- `examples/evaluation_strategies.py`: output scoring and evaluator-controlled repeated calls.
- `examples/schema_components.py`: tool description and parameter components.
- `examples/model_schema_components.py`: nested Pydantic output descriptions.
- `examples/dot_optimization.py`: multimodal Pydantic AI extraction with instructions and output
  schema optimization.
- `examples/staged_grouped.py`: ordered component groups and shared budgets.
- `examples/checkpoint_resume.py`: compatible durable resume.
- `examples/events_progress.py` and `examples/logfire_observer.py`: observers and progress.
- `examples/low_level_adapter.py`: direct adapter and reflective evidence for integration authors.
- `examples/experimental_optimize_anything.py`: real single GEPA and deterministic custom-engine
  Omni pipeline with held-out scoring and lineage.

## Implementation Workflow

### Application integration

1. Define typed `Example` values and distinct validation data.
2. Keep the task an ordinary input-to-output callable.
3. Define a stable component catalog and baseline candidate.
4. Bind values through built-in injections or one typed derived-value context.
5. Return a clear scalar or `MetricResult` with actionable feedback.
6. Construct typed config, set a bounded budget, and run.
7. Inspect candidate history, validation scores, stop reason, and stable result evidence.

### Framework change

1. Read `references/operations.md` and the domain reference.
2. Preserve common API simplicity and optional dependency boundaries.
3. Update source, public exports, result/state compatibility, docs, runnable examples, and tests.
4. Maintain meaningful 100% source line and branch coverage.
5. Run `make prod` and `make pre-commit`.

## Completion Criteria

A complete optimization has typed train/validation examples, scoped candidate application, a
bounded typed config, meaningful objective evidence, inspectable candidate lineage, a successful
run or clearly reproduced failure, and no automatic production promotion. Repository work also
requires current docs/examples and the full quality gate.
