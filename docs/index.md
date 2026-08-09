# Typed evolutionary optimization for Pydantic applications

`pydantic-gepa` lets you optimize prompts, instructions, tool descriptions,
structured-output schemas, and other text components without exposing your
application to GEPA's callback plumbing. You provide typed examples, the
callable that runs your application, a scoring function, and the components
that may change. The library runs Pydantic Evals internally and returns a typed,
inspectable optimization result.

```python
from pydantic_gepa import Component, Example, optimize

instructions = Component(
    name="instructions",
    initial_text="Classify the support request.",
    kind="instructions",
)

result = optimize(
    train=[Example(inputs="Where is my order?", expected_output="shipping")],
    validation=[Example(inputs="Refund order 123", expected_output="refund")],
    task=run_classifier,
    score=lambda ctx: float(ctx.output == ctx.expected_output),
    components=[instructions],
    reflection="openai:gpt-5-mini",
    budget=50,
)

print(result.best_candidate.values)
print(result.best_score)
```

## What the package owns

The common API owns the integration work between three systems:

1. **Your application** remains an ordinary typed callable or Pydantic AI agent.
2. **Pydantic Evals** executes and evaluates examples internally. You do not
   need to construct `Dataset` or `Case` objects.
3. **GEPA** proposes and selects candidate components. Typed configuration,
   candidate normalization, reflection evidence, and results are handled by
   `pydantic-gepa`.

This boundary makes the optimizer useful outside Autobench. Autobench can
record its evidence, and Autoptimize can orchestrate promotion, but neither is
required for a standalone optimization.

## What you can optimize

- Pydantic AI agent instructions and system prompts
- tool descriptions and parameter descriptions
- Pydantic output-model and nested field descriptions
- several coupled components in one candidate
- arbitrary values derived from candidate text through a typed context
- staged component groups with shared budgets and checkpoints
- experimental Optimize Anything objectives

## Start here

- [Install the package](installation.md)
- [Run the first optimization](getting-started.md)
- [Learn the mental model](concepts.md)
- [Choose a complete example](examples.md)
- [Inspect the Python API](api-reference.md)

!!! note "Alpha status"

    The package is usable, typed, and tested, but its version is still pre-1.0.
    Experimental APIs are explicitly isolated under
    `pydantic_gepa.experimental`.
