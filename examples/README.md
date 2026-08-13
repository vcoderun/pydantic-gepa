# pydantic-gepa examples

These examples use the released `pydantic-ai`, `pydantic-evals`, and `gepa`
dependencies from the `examples` extra. Deterministic examples do not require API keys.

- `basic.py`: optimize agent instructions through the common `Example`, `Component`,
  and `optimize(...)` API.
- `low_level_adapter.py`: evaluate a Pydantic Evals `Dataset` through the advanced
  adapter API and build reflective examples.
- `optimizer.py`: run a real GEPA optimization loop over agent instructions.
- `recorder_hook.py`: attach a recorder hook to capture real evaluation batches and
  report aggregates.
- `schema_components.py`: extract optimizable tool/output schema description components
  from a Pydantic-AI-compatible tool definition shape and apply a candidate back to the
  schema payload.
- `model_schema_components.py`: extract optimizable Pydantic model field descriptions
  for structured outputs and apply candidate overrides back to the generated JSON schema.
- `dot_optimization.py`: real-world style rewrite of a multimodal extraction optimizer
  using `ImageUrl`, typed Pydantic AI output, `Example` records, callable scoring, and
  GEPA without exposing Pydantic Evals cases or datasets to the user script.
- `evaluation_strategies.py`: output scoring, evaluator-owned repeated execution,
  multi-metric results, feedback, and structured side information.
- `staged_grouped.py`: sequential carry-forward plus a grouped writer/tool stage.
- `checkpoint_resume.py`: durable run state and a completed-run resume without repeated work.
- `events_progress.py`: typed lifecycle events and Rich progress from the same plan.
- `logfire_observer.py`: optional structured Logfire delivery through typed events.
- `experimental_optimize_anything.py`: a real GEPA engine plus deterministic custom-engine
  BestOf/continuation Omni pipeline, held-out evaluation, lineage, and typed AutoResearch
  declaration under the explicit experimental namespace.

Run one example with:

```bash
uv run --extra examples python examples/basic.py
```

`dot_optimization.py` and live model-backed reflection require the provider credentials
used by their configured Pydantic AI models. Pass `--logfire` to the dot example only
when remote trace delivery is intended. The remaining examples are deterministic.
