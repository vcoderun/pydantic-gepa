# Multimodal Evidence

`Example` attachments and typed inputs let the optimizer evaluate image, audio,
document, and other binary-backed tasks without reducing them to text.

## Referenced media

Keep large media outside the candidate and reference it from typed input:

```python
from pydantic import BaseModel
from pydantic_ai import ImageUrl
from pydantic_gepa import Example

class ExtractionInput(BaseModel):
    image: ImageUrl

example = Example(
    id="receipt-42",
    inputs=ExtractionInput(image=ImageUrl(url="https://example.test/receipt.png")),
    expected_output=Receipt(total=19.95, currency="USD"),
)
```

## Binary attachment metadata

Use `Attachment.from_bytes` when the optimization dataset owns binary content:

```python
attachment = Attachment.from_bytes(
    image_bytes,
    kind="image",
    media_type="image/png",
    reference="receipt-42.png",
)
```

The digest and size participate in stable evidence identity. The integration's
encoder decides how content is represented to reflection; large binary payloads
should not be embedded directly in reflection prompts.

## Structured extraction

Combine multimodal inputs with `ModelOutputInjection` and
`model_field_accuracy`:

```python
score = model_field_accuracy("customer_name", "account_id", "document_id")
```

This measures field-level output quality while GEPA may optimize both agent
instructions and field descriptions.

## Pydantic AI evidence encoder

`pydantic_gepa.integrations.pydantic_ai.evidence_encoder` understands supported
Pydantic AI content parts. `register_evidence` extends an encoder explicitly for
integration-specific payloads.

## Operational limits

- Prefer references or artifact stores for large files.
- Never place secrets in attachment metadata or reflection feedback.
- Keep provider-specific image conversion in the application adapter.
- Include media identity in example ids so caches do not confuse changed files.

See the full [`dot_optimization.py`](https://github.com/vcoderun/pydantic-gepa/blob/main/examples/dot_optimization.py)
example.
