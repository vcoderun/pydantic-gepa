# Reflection And Feedback

Reflection is the step that turns evaluation evidence into proposed component
text. Better evidence is usually more valuable than a larger mutation budget.

## What reaches reflection

The adapter can include:

- example identity and inputs
- expected and actual outputs
- objective and secondary metrics
- evaluator feedback and structured side information
- assertions and normalized failures
- captured runtime attempts and artifacts
- current candidate values and selected components

The normalized failure category is `None` for a non-failure. Failures may be
classified as task, evaluator, assertion, infrastructure, or low-score evidence.

## Metric feedback

Return a `MetricResult` when a scalar alone would be ambiguous:

```python
return MetricResult(
    score=0.5,
    feedback="The route was correct, but the required account identifier was omitted.",
    side_info={
        "correct_route": True,
        "missing_fields": ["account_id"],
    },
)
```

Feedback should identify observed behavior, not prescribe a brittle exact
prompt. Side information should be serializable and focused on mutation-relevant
evidence.

## Reflection models

### Model identifier

```python
ReflectionConfig(model="openai:gpt-5-mini")
```

### Callable

```python
CallableReflectionModel(lambda prompt: propose(prompt), retries=2)
```

The callable may be synchronous or asynchronous; the adapter presents GEPA's
synchronous reflection contract.

### Pydantic AI

```python
from pydantic_gepa.integrations.pydantic_ai import PydanticAIReflectionModel

reflection = PydanticAIReflectionModel.from_model(
    "openai:gpt-5-mini",
    max_output_tokens=2_000,
    timeout=30,
    retries=2,
)
```

Pydantic AI remains an optional import. The integration records reflection
usage and cost when the provider exposes them.

## Perfect scores

`skip_perfect_score=True` avoids reflecting on examples that already satisfy
the configured perfect score. This reduces cost and keeps reflection focused on
useful errors.

## Custom proposers

A custom proposer is an advanced seam for systems that already own a mutation
engine. It should consume normalized evidence and return component text using
the component's serialization contract. Prefer the standard reflection model
until evidence proves a custom proposer is necessary.
