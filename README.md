# pydantic-gepa

`pydantic-gepa` provides a typed optimization runtime over GEPA, Pydantic AI,
and internally managed Pydantic Evals execution.

```python
from pydantic_gepa import AgentInstructionsInjection, Component, Example, optimize

instructions = Component(name="instructions", initial_text="Answer precisely.")

result = optimize(
    train=training_examples,
    validation=validation_examples,
    task=run_subject,
    score=score_subject,
    components=[instructions],
    injections=[
        AgentInstructionsInjection(agent=agent, candidate_component=instructions)
    ],
    reflection="openai:gpt-5-mini",
    budget=50,
)
```

The package starts with a Python API:

- typed GEPA candidate wrappers
- typed candidate component catalogs with include/exclude selection
- raw-text candidate values by default, with explicit JSON-string codecs when required
- candidate injection helpers for Pydantic AI agents
- generic candidate context/value injections for runtime schema or config overrides
- high-level `optimize(...)`, `Example`, and `Optimization.from_examples(...)` APIs that keep
  Pydantic Evals as internal runtime plumbing for common optimization flows
- callable score functions and built-in `model_field_accuracy(...)` helpers, with custom
  Pydantic Evals evaluators still available as an advanced escape hatch
- Pydantic Evals harness normalization
- score extraction from named evaluator scores
- `MetricResult(score, feedback, side_info)` for richer reflection signals
- reflective dataset construction for GEPA side information, case metadata, expected output,
  assertion failures, metric feedback, metric side info, success flags, and failure categories
- YAML-first candidate save/load helpers
- Pydantic-AI-compatible tool/output schema description component extraction
- Pydantic model field-description component extraction for structured outputs
- optional recorder hooks for higher-level systems and GEPA callback bridging
- deterministic sequential/grouped plans, typed events, Rich progress, and
  compatibility-checked checkpoint/resume
- a Click CLI over typed Python targets, without a second optimization DSL

Install with `uv`:

```bash
uv add pydantic-gepa
```

Or with `pip`:

```bash
pip install pydantic-gepa
```

Development setup:

```bash
uv sync --extra dev --extra integrations
make prod
```

Build the documentation with:

```bash
make docs
```

Run a configured Python target from the CLI:

```bash
pydantic-gepa my_app.optimization:pipeline inspect target
pydantic-gepa my_app.optimization:pipeline run --run-dir runs/demo
```
