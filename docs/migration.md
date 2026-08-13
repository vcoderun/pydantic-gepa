# Migration

## From direct Pydantic Evals construction

Before:

```python
dataset = Dataset(cases=[], evaluators=[...])
cases = [Case(inputs=..., expected_output=...)]
adapter = PydanticGEPAAdapter.from_dataset(...)
```

After:

```python
pipeline = Optimization.from_examples(
    examples=[Example(inputs=..., expected_output=...)],
    val_examples=validation,
    task=task,
    score=score,
    components=components,
    injections=injections,
)
```

Keep direct datasets only when custom Pydantic Evals evaluator lifecycle is the
reason for the integration.

## From loose candidate dictionaries

Wrap values in `Candidate` and describe the search space with `Component` or
`ComponentCatalog`. This adds serialization, stable identity, lineage, schema
metadata, and injection validation.

## From manual output-type factories

Replace custom Pydantic model subclass builders with:

```python
output_schema = ModelOutputInjection(MyOutput)
```

Merge `output_schema.components`, include `output_schema` in injections, and
pass `output_schema.require()` directly as the Pydantic AI `output_type`.

## From untyped GEPA kwargs

Replace standalone options with `GEPAConfig` nested models. The compatibility
mapper recognizes known legacy names and rejects unknown values.

## From direct GEPA optimizer calls

Move candidate evaluation into `optimize(...)` or `Optimization`. Use the
low-level adapter only if the integration must explicitly inspect evaluation
batches or reflective datasets.

## From one monolithic optimizer

Use `Plan` only when component ownership, budgets, or validation differ by
stage. Do not split a simple optimization solely for style.

## From standard to Optimize Anything

Set `backend="optimize_anything"` when constructing `Optimization`, then pass
the typed experimental configuration:

```python
from pydantic_gepa.experimental.optimize_anything import (
    Engine,
    OptimizeAnythingConfig,
)

result = optimization.optimize(
    config=OptimizeAnythingConfig(
        engine=Engine.gepa(gepa_config),
    )
)
```

Passing `GEPAConfig` directly to this backend remains a one-cycle deprecated
compatibility path and means one GEPA engine. Use `composition=` for
sequential, parallel, best-of, vote, adaptive, or Omni pipeline behavior. Keep
the standard backend available until held-out evidence demonstrates parity for
the application.
