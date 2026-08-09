# Troubleshooting

## GEPA or Pydantic Evals cannot be imported

Install the runtime extra:

```bash
uv add "pydantic-gepa[integrations]"
```

The base package intentionally does not require every integration dependency.

## Candidate is missing a component

Ensure the component exists in both the initial candidate and component catalog.
When using schema injection, merge `output_schema.components` into the catalog.

## Candidate context has no active value

Read `CandidateContext.require()` only while the task is running under its
injection. Add the injection to the optimization and avoid reading it at module
import time.

## Structured output descriptions do not change

Pass `output_type=output_schema.require()` to the Pydantic AI call. Supplying the
original model class bypasses the candidate-specific type.

## Scores are invalid

Verify that the selected objective key exists, is numeric, is finite, and lies
within configured bounds. Return `MetricResult` for diagnostic metadata rather
than placing nonnumeric values in the objective.

## Training and validation are rejected

Provide distinct validation examples. Same-set validation requires an explicit
compatibility setting and does not demonstrate generalization.

## Resume is incompatible

The target, callable identities, components, configuration, or stage graph
changed. Start a fresh run directory unless compatibility can be restored
honestly. Do not disable checks to reuse stale state.

## Run appears stuck

Enable `ProgressConfig(display_bar=True)` and a Rich or Logfire observer. Check
provider timeouts, concurrency, retry behavior, and reflection model logs.
Evaluation and reflection are separate model-call sources.

## Event logs appear but GEPA progress does not

Package events and backend progress are separate. Enable both the package Rich
observer and `display_bar` when both views are desired.

## CLI cannot import target

Run from an environment where the application package is installed or on
`sys.path`. Use `module:attribute`, not a filesystem path, and keep module import
free of required network calls.

## Type checker loses callable types

Keep task, scorer, and example types explicit. Avoid replacing typed values with
`Any` or generic dictionaries solely to satisfy an adapter. The common API is
generic across input, output, and metadata types.
