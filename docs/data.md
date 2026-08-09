# Examples And Data

## Typed examples

`Example[InputsT, OutputT, MetadataT]` is the common data contract:

```python
from pydantic import BaseModel
from pydantic_gepa import Example

class Ticket(BaseModel):
    text: str
    customer_tier: str

examples = [
    Example(
        id="ticket-001",
        name="enterprise-refund",
        inputs=Ticket(text="Refund invoice 42", customer_tier="enterprise"),
        expected_output="refund",
        metadata={"source": "held-out", "locale": "en"},
    )
]
```

The optimizer preserves the types of `inputs`, `expected_output`, and metadata
through task and scorer callables. Pydantic Evals conversion happens inside the
common API.

## Identity

Provide a stable `id` when the sample has a domain identity. Otherwise the
library can derive a deterministic fingerprint from normalized content. Stable
identity matters for caches, reflection evidence, reproducible splits, and
cross-run recording.

## Attachments

Attachments let examples carry binary evidence without forcing the payload into
JSON-like metadata:

```python
from pydantic_gepa import Attachment, Example

image = Attachment.from_bytes(
    b"...",
    kind="image",
    media_type="image/png",
    reference="receipt.png",
)

example = Example(
    inputs="Extract the receipt total.",
    expected_output="19.95",
    attachments=(image,),
)
```

An attachment records its kind, reference, media type, byte size, and digest.
The task remains responsible for translating a reference into the SDK-specific
binary or URL type it consumes.

## Train, validation, and test

The one-shot API requires separate `train` and `validation` sequences. For a
reusable deterministic split, use `DataSplit`:

```python
from pydantic_gepa import DataSplit

split = DataSplit.partition(
    examples,
    validation_fraction=0.2,
    test_fraction=0.1,
    seed=17,
)
```

Do not choose the best candidate on the same examples used to mutate it. Keep
the test set untouched until final rescoring or promotion.

## Scorer context

A common score callable receives `EvaluationContext`:

```python
def score(ctx):
    exact = float(ctx.output == ctx.expected_output)
    return exact
```

The context exposes the example name, inputs, output, expected output, metadata,
and execution duration. Return a float, a named metric mapping, or
`MetricResult` for feedback and side information.

## Advanced evaluator escape hatch

Custom Pydantic Evals evaluators may be supplied through `evaluators=...` when
their lifecycle is required. This is an advanced extension point; ordinary
users should prefer score callables and built-in helpers such as
`model_field_accuracy`.
