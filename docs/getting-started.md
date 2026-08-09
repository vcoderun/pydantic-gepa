# First Optimization

This example optimizes the instructions of a small support-routing subject. It
uses a deterministic reflection callable so it can be understood without a
second model provider.

## 1. Define examples

```python
from pydantic_gepa import Example

train = [
    Example(name="refund", inputs="Refund order 123", expected_output="refund"),
    Example(name="shipping", inputs="Where is my parcel?", expected_output="shipping"),
]

validation = [
    Example(name="refund-validation", inputs="I need my money back", expected_output="refund"),
    Example(name="shipping-validation", inputs="Tracking has not updated", expected_output="shipping"),
]
```

Training and validation are explicit. Validation should represent the behavior
you want to generalize to, not simply repeat training examples.

## 2. Define the subject

```python
active_instructions = "Route every request as other."

def run_subject(text: str) -> str:
    lowered = text.lower()
    if "refund" in active_instructions.lower() and (
        "refund" in lowered or "money back" in lowered
    ):
        return "refund"
    if "shipping" in active_instructions.lower() and (
        "parcel" in lowered or "tracking" in lowered
    ):
        return "shipping"
    return "other"
```

In a real application, this function usually calls a Pydantic AI agent. The
optimizer only requires an input-to-output callable.

## 3. Bind a candidate component

Pydantic AI agents can use `AgentInstructionsInjection`. For this plain callable
we use a typed candidate context:

```python
from pydantic_gepa import CandidateContext, Component, DerivedValueInjection

instructions = Component(
    name="instructions",
    initial_text="Route every request as other.",
    kind="instructions",
)
instruction_context = CandidateContext[str]("instructions", instructions.initial_text)

def subject(text: str) -> str:
    global active_instructions
    active_instructions = instruction_context.require()
    return run_subject(text)

injection = DerivedValueInjection(
    component="instructions",
    context=instruction_context,
    required_components=("instructions",),
    derive_value=lambda candidate: candidate["instructions"],
)
```

An injection is active only while an example is evaluated. Context-local values
prevent one candidate from becoming permanent application state.

## 4. Optimize

```python
from pydantic_gepa import CallableReflectionModel, optimize
from pydantic_gepa.configuration import BudgetConfig, GEPAConfig, ReflectionConfig

config = GEPAConfig(
    budget=BudgetConfig(max_metric_calls=12),
    reflection=ReflectionConfig(
        model=CallableReflectionModel(
            lambda _prompt: "Route refund requests as refund and delivery questions as shipping."
        )
    ),
)

result = optimize(
    train=train,
    validation=validation,
    task=subject,
    score=lambda ctx: float(ctx.output == ctx.expected_output),
    components=[instructions],
    injections=[injection],
    config=config,
)
```

## 5. Inspect the evidence

```python
print(result.best_candidate.values["instructions"])
print(result.best_score)
print(result.validation_scores)

for candidate in result.candidate_history:
    print(candidate.candidate_id, candidate.parent_ids, candidate.score)
```

The result preserves candidate values, parent-child lineage, objective and
validation scores, budget use, Pareto information when available, and artifact
references. See [Results and lineage](results.md).

The repository's runnable version is [`examples/basic.py`](examples.md#basic-instruction-optimization).
