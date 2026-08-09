# Candidate Injection

Candidate generation and application execution are separate concerns. An
injection temporarily binds candidate values while an example runs.

## Pydantic AI instructions

```python
from pydantic_gepa import AgentInstructionsInjection, Component

instructions = Component(
    name="instructions",
    initial_text="Classify the ticket.",
    kind="instructions",
)

injection = AgentInstructionsInjection(
    agent=agent,
    candidate_component=instructions,
)
```

The injection uses the agent's `override(instructions=...)` context manager. It
does not permanently mutate the agent.

## Structured output type

`ModelOutputInjection` owns both component collection and candidate-specific
Pydantic model construction:

```python
from pydantic_gepa import ModelOutputInjection

output_schema = ModelOutputInjection(ExtractionOutput)

result = agent.run_sync(
    prompt,
    output_type=output_schema.require(),
)
```

Pass `output_schema` in the optimization's `injections`, and merge
`output_schema.components` into the component catalog. While an evaluation is
active, `require()` returns the model type with candidate field descriptions.
Outside that scope it returns the configured baseline model type.

```python
components = ComponentCatalog.from_components([instructions]).merge(
    output_schema.components
)
```

No user-written `build_output_type` function is needed.

## Arbitrary typed values

Use `CandidateContext` and `DerivedValueInjection` for routing policies,
retriever configuration, or application-specific values:

```python
from pydantic_gepa import CandidateContext, DerivedValueInjection

policy_context = CandidateContext[RoutingPolicy]("routing-policy")

policy_injection = DerivedValueInjection(
    component="routing.policy",
    context=policy_context,
    required_components=("routing.policy", "routing.fallback"),
    derive_value=lambda values: RoutingPolicy(
        primary=values["routing.policy"],
        fallback=values["routing.fallback"],
    ),
)
```

The task reads `policy_context.require()`. Context variables keep concurrent
evaluations isolated.

## Validation-only injection

`NoopInjection(component="prompt")` verifies that a candidate has the named
component without changing application state. It is useful when the task reads
the candidate through an external mechanism.

## Lifecycle

For every evaluation, the runtime:

1. validates required components;
2. enters all injection context managers;
3. runs the task or evaluator-controlled calls;
4. records outputs, metrics, feedback, and traces;
5. exits contexts in reverse order, including failures.

Do not implement injection by assigning global mutable state. Use the provided
contexts or an application SDK's scoped override API.
