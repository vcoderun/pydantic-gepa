# Schema Optimization

Descriptions in tool and output schemas affect how models choose tools and
produce structured data. pydantic-gepa can expose those descriptions as normal
candidate components.

## Pydantic models

```python
from pydantic import BaseModel, Field
from pydantic_gepa import collect_model_components

class Address(BaseModel):
    city: str = Field(description="City")
    postal_code: str = Field(description="Postal code")

class Order(BaseModel):
    destination: Address
    status: str = Field(description="Order status")

catalog = collect_model_components(Order, model_name="Order")
print(catalog.names())
```

Nested paths remain stable, including arrays and definitions. Typical keys are:

```text
output:Order:description
output:Order:param:destination.city
output:Order:param:destination.postal_code
output:Order:param:status
```

## First-class output injection

Prefer `ModelOutputInjection(Order)` in an optimization. It collects components,
applies candidate descriptions, and exposes the active model type. You do not
need to write a custom model subclass factory.

## Tool definitions

The generic `ToolDefinitionView` contract requires a name, optional description,
and parameters JSON schema:

```python
catalog = collect_tool_components(search_tool)
candidate = catalog.to_candidate()
updated = apply_tool_schema_candidate(search_tool, candidate.values)
```

`collect_toolset_components` merges several tools. Component keys distinguish
the tool description from individual parameters:

```text
tool:search:description
tool:search:param:query
tool:search:param:limit
```

## Applying candidates

`apply_model_schema_candidate` and `apply_tool_schema_candidate` return candidate
views and copied schemas. They preserve original definitions and report the
description overrides that were actually applied.

## Field-name fallback

Fields without descriptions may use their field name as the initial component
text. Disable `include_field_name_fallback` when only deliberate human-authored
descriptions should be optimized.

## Scope selection

Schema catalogs are ordinary `ComponentCatalog` values. Select only tool
descriptions, only output fields, or a specific prefix before a staged run:

```python
tool_components = catalog.select(include=["tool:"], mode="prefix")
```

## Validation

After applying a candidate, validate both schema construction and real model
outputs. A syntactically valid description candidate can still reduce tool
choice or extraction quality.
