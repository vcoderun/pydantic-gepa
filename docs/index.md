# pydantic-gepa

`pydantic-gepa` turns typed examples, application callables, and candidate
components into GEPA optimization runs. Pydantic Evals and GEPA callback
plumbing stay behind the common API.

```python
from pydantic_gepa import Component, Example, optimize

result = optimize(
    train=[Example(inputs="2 + 2", expected_output="4")],
    validation=[Example(inputs="3 + 3", expected_output="6")],
    task=run_subject,
    score=lambda ctx: float(ctx.output == ctx.expected_output),
    components=[Component(name="instructions", initial_text="Answer.")],
    reflection="openai:gpt-5-mini",
    budget=50,
)
```

The stable vocabulary is `optimize`, `Optimization`, `Example`, `Component`,
`Candidate`, `Runtime`, `MetricResult`, `Plan`, `Stage`, and
`OptimizationResult`. Advanced contracts remain public from focused modules.

## Package boundary

- pydantic-gepa executes typed optimization and returns normalized evidence.
- Autobench records benchmark and optimization evidence.
- Autoptimize chooses experiments, validates candidates, and owns promotion.
