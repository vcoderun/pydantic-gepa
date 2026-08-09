# Components And Candidates

## Component

A `Component` describes text GEPA may edit:

```python
from pydantic_gepa import Component

instructions = Component(
    name="agent.instructions",
    initial_text="Answer the user's question.",
    kind="instructions",
    semantic_type="agent.instructions",
    asset_ref="prompt://support-agent/main",
    source="prompts/support.md",
    optimizable=True,
)
```

Important fields:

| Field | Meaning |
| --- | --- |
| `name` | Stable key used in candidates and GEPA |
| `initial_text` | Baseline human-readable value |
| `kind` | Instructions, system prompt, schema description, or custom text |
| `semantic_type` | Optional domain classification |
| `asset_ref` | Optional external tracked-asset identity |
| `injection_target` | Optional application binding hint |
| `serialization` | Raw text by default; explicit JSON string when required |
| `coupled_components` | Components that should be considered together |

## Candidate

A `Candidate` is one concrete assignment:

```python
from pydantic_gepa import Candidate

candidate = Candidate(
    id="candidate-7",
    parent_id="candidate-3",
    generation=2,
    values={
        "agent.instructions": "Classify first, then answer concisely.",
        "tool:search:description": "Search verified documents only.",
    },
    metadata={"proposer": "reflection"},
)
```

`fingerprint()` hashes normalized values. `save_yaml()` and `load_yaml()` provide
a portable snapshot. Candidate models are frozen; create a new candidate rather
than mutating evidence in place.

## Catalog

`ComponentCatalog` deduplicates by component name, produces the initial
candidate, selects component groups, and merges schema-derived components:

```python
from pydantic_gepa import ComponentCatalog

catalog = ComponentCatalog.from_components([instructions, search_description])
prompt_catalog = catalog.select(include=["agent"], mode="prefix")
initial = catalog.to_candidate(candidate_id="baseline")
```

Prefix selection treats `agent`, `agent.*`, and `agent:*` as one namespace.
Use `mode="exact"` when only exact component names should match.

## Component naming

Use stable names that communicate ownership and target:

```text
agent.instructions
agent.system_prompt
tool:search:description
tool:search:param:query
output:ExtractionResult:param:customer_name
```

Do not put candidate versions into component names. Version and lineage belong
to `Candidate.id`, `parent_id`, `generation`, and external asset references.

## Raw and JSON-string serialization

Raw text is the default and preferred representation. Use
`serialization="json_string"` only when a backend contract explicitly expects
a JSON-encoded string. Components encode before entering the candidate and
decode at the injection boundary.
