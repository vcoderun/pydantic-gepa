from __future__ import annotations as _annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict, Field

import pydantic_gepa.injections as injections
import pydantic_gepa.schema_components as schema_components
from pydantic_gepa import (
    Candidate,
    CandidateComponent,
    CandidateComponentError,
    ComponentCatalog,
    ComponentSelector,
    ModelOutputInjection,
    apply_model_schema_candidate,
    apply_tool_schema_candidate,
    collect_model_components,
    collect_tool_components,
    collect_toolset_components,
    description_key,
    format_schema_path,
    iter_model_descriptions,
    iter_schema_descriptions,
    parameter_key,
    parse_schema_path,
    set_schema_description,
)
from pydantic_gepa.schema_components import copy_json, copy_json_object
from pydantic_gepa.values import JsonValue


def test_component_catalog_builds_candidates_and_selects_by_prefix_or_exact_match() -> None:
    catalog = ComponentCatalog.from_components(
        [
            CandidateComponent(name="instructions", initial_text="base", kind="instructions"),
            CandidateComponent(
                name="tool:search:param:query",
                initial_text="search query",
                kind="schema_description",
                semantic_type="tool.parameter.description",
                source="search",
                path="query",
            ),
            CandidateComponent(
                name="tool:search:param:limit",
                initial_text="limit",
                optimizable=False,
            ),
            CandidateComponent(name="instructions", initial_text="deduped"),
        ]
    )

    assert catalog.names() == [
        "instructions",
        "tool:search:param:query",
        "tool:search:param:limit",
    ]
    assert catalog.values() == {
        "instructions": "deduped",
        "tool:search:param:query": "search query",
    }
    assert catalog.to_candidate(candidate_id="seed").id == "seed"
    assert catalog.select(include=["tool:search"]).names() == [
        "tool:search:param:query",
        "tool:search:param:limit",
    ]
    assert catalog.select(exclude=["tool:search:param:limit"], mode="exact").names() == [
        "instructions",
        "tool:search:param:query",
    ]
    assert ComponentSelector(include=("tool:search",)).accepts("tool:search:param:query")
    assert not ComponentSelector(include=("tool:search",)).accepts("instructions")


def test_component_catalog_merges_with_last_component_winning() -> None:
    first = ComponentCatalog.from_components(
        [CandidateComponent(name="instructions", initial_text="old")]
    )
    second = ComponentCatalog.from_components(
        [CandidateComponent(name="instructions", initial_text="new")]
    )

    assert first.merge(second).values() == {"instructions": "new"}


def test_candidate_yaml_roundtrip_and_invalid_payloads(tmp_path: Path) -> None:
    path = tmp_path / "candidate.yaml"
    candidate = Candidate.from_gepa_dict(
        {"instructions": "Do it."},
        candidate_id="candidate_1",
        metadata={"score": 1.0},
    )

    candidate.save_yaml(path)
    loaded = Candidate.load_yaml(path)
    empty_path = tmp_path / "empty.yaml"
    invalid_path = tmp_path / "invalid.yaml"
    empty_path.write_text("", encoding="utf-8")
    invalid_path.write_text("- nope\n", encoding="utf-8")

    assert loaded == candidate
    assert Candidate.load_yaml(empty_path) == Candidate()
    with pytest.raises(CandidateComponentError, match="mapping"):
        Candidate.load_yaml(invalid_path)


def test_tool_schema_component_collection_uses_tool_and_parameter_descriptions() -> None:
    tool = _ToolDefinition(
        name="search",
        description="Search the index.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "field": {"type": "string", "description": "Filter field"},
                        },
                    },
                },
            },
        },
    )

    descriptions = iter_schema_descriptions(
        tool.parameters_json_schema,
        tool_name=tool.name,
        target="tool",
    )
    catalog = collect_tool_components(tool)
    output_catalog = collect_toolset_components([tool], target="output", serialization="raw")

    assert [item.key for item in descriptions] == [
        "tool:search:param:query",
        "tool:search:param:filters[].field",
    ]
    assert descriptions[1].formatted_path == "filters[].field"
    assert catalog.names() == [
        "tool:search:description",
        "tool:search:param:query",
        "tool:search:param:filters[].field",
    ]
    assert output_catalog.values()["output:search:description"] == "Search the index."
    assert description_key("search", target="tool") == "tool:search:description"
    assert parameter_key("search", ("filters", "[]", "field"), target="tool") == (
        "tool:search:param:filters[].field"
    )
    assert format_schema_path(("items", "[]")) == "items[]"
    assert parse_schema_path("filters[].field") == ("filters", "[]", "field")


def test_apply_tool_schema_candidate_returns_updated_schema_without_mutating_source() -> None:
    original_schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "ignored": {"type": "string"},
        },
    }
    tool = _ToolDefinition(
        name="search",
        description="Search the index.",
        parameters_json_schema=original_schema,
    )
    catalog = collect_tool_components(tool)

    updated = apply_tool_schema_candidate(
        tool,
        {
            "tool:search:description": "Search documents precisely.",
            "tool:search:param:query": "Natural language query.",
        },
        components=catalog,
    )

    assert updated.description == "Search documents precisely."
    assert updated.parameters_json_schema["properties"] == {
        "query": {"type": "string", "description": "Natural language query."},
        "ignored": {"type": "string"},
    }
    assert original_schema["properties"] == {
        "query": {"type": "string", "description": "Search query"},
        "ignored": {"type": "string"},
    }
    assert set_schema_description(updated.parameters_json_schema, ("missing",), "x") is False
    assert (
        set_schema_description(
            updated.parameters_json_schema, ("query",), "Natural language query."
        )
        is False
    )
    assert (
        set_schema_description(updated.parameters_json_schema, ("query",), "Better query.") is True
    )


def test_apply_tool_schema_candidate_handles_missing_updates_and_sparse_schemas() -> None:
    tool = _ToolDefinition(
        name="plain",
        description=None,
        parameters_json_schema={
            "type": "object",
            "properties": {
                "bad": "not a schema",
                "array": {"type": "array", "items": "not a schema"},
                "empty": {"type": "object", "properties": []},
            },
        },
    )
    catalog = ComponentCatalog.from_components(
        [
            CandidateComponent(name="manual", initial_text="manual", path=None),
            CandidateComponent(name="missing", initial_text="missing", path="missing"),
        ]
    )

    updated = apply_tool_schema_candidate(tool, {"other": "value"}, components=catalog)

    assert collect_tool_components(tool).names() == []
    assert updated.description is None
    assert updated.parameters_json_schema == tool.parameters_json_schema
    assert parse_schema_path("") == ()
    assert format_schema_path(("[]", "name")) == "[].name"
    assert set_schema_description(
        {"type": "array", "items": {"type": "object"}},
        ("[]",),
        "Array item",
    )
    assert not set_schema_description({"type": "array", "items": []}, ("[]",), "x")
    assert not set_schema_description({"type": "object"}, ("missing",), "x")


def test_model_schema_component_collection_uses_field_descriptions_and_fallbacks() -> None:
    catalog = collect_model_components(_OrderSummary, model_name="OrderSummary")
    descriptions = iter_model_descriptions(
        _OrderSummary.model_json_schema(),
        model_name="OrderSummary",
        target="output",
        include_field_name_fallback=True,
    )

    assert [item.key for item in descriptions] == [
        "output:OrderSummary:param:status",
        "output:OrderSummary:param:shipping.city",
        "output:OrderSummary:param:shipping.postal_code",
        "output:OrderSummary:param:tags[]",
    ]
    assert catalog.names() == [item.key for item in descriptions]
    assert catalog.values()["output:OrderSummary:param:shipping.postal_code"] == "postal_code"


def test_apply_model_schema_candidate_updates_nested_schema_descriptions() -> None:
    catalog = collect_model_components(_OrderSummary, model_name="OrderSummary")

    updated = apply_model_schema_candidate(
        _OrderSummary,
        {
            "output:OrderSummary:param:status": "Current fulfillment state.",
            "output:OrderSummary:param:shipping.city": "Destination city name.",
            "output:OrderSummary:param:shipping.postal_code": "Destination postal code.",
        },
        model_name="OrderSummary",
        components=catalog,
    )

    properties = updated.json_schema["properties"]
    assert isinstance(properties, dict)
    status_schema = properties["status"]
    assert isinstance(status_schema, dict)
    assert status_schema["description"] == "Current fulfillment state."
    shipping_schema = properties["shipping"]
    assert isinstance(shipping_schema, dict)
    definitions = updated.json_schema["$defs"]
    assert isinstance(definitions, dict)
    shipping_definition = definitions["_ShippingAddress"]
    assert isinstance(shipping_definition, dict)
    shipping_properties = shipping_definition["properties"]
    assert isinstance(shipping_properties, dict)
    city_schema = shipping_properties["city"]
    assert isinstance(city_schema, dict)
    assert city_schema["description"] == "Destination city name."
    postal_code_schema = shipping_properties["postal_code"]
    assert isinstance(postal_code_schema, dict)
    assert postal_code_schema["description"] == "Destination postal code."
    assert updated.description_overrides == {
        "status": "Current fulfillment state.",
        "shipping.city": "Destination city name.",
        "shipping.postal_code": "Destination postal code.",
    }


def test_model_schema_candidate_handles_top_level_description_and_nested_arrays() -> None:
    catalog = collect_model_components(_InventorySummary, model_name="InventorySummary")

    updated = apply_model_schema_candidate(
        _InventorySummary,
        {
            "output:InventorySummary:description": "Optimized inventory schema.",
            "output:InventorySummary:param:warehouses[].city": "Warehouse city name.",
        },
        model_name="InventorySummary",
        components=catalog,
    )

    assert description_key("InventorySummary", target="output") in catalog.names()
    assert updated.description == "Optimized inventory schema."
    definitions = updated.json_schema["$defs"]
    assert isinstance(definitions, dict)
    warehouse_definition = definitions["_ShippingAddress"]
    assert isinstance(warehouse_definition, dict)
    warehouse_properties = warehouse_definition["properties"]
    assert isinstance(warehouse_properties, dict)
    city_schema = warehouse_properties["city"]
    assert isinstance(city_schema, dict)
    assert city_schema["description"] == "Warehouse city name."


def test_model_output_injection_builds_candidate_output_type_and_resets_context() -> None:
    injection = ModelOutputInjection(_InventorySummary, model_name="InventorySummary")

    assert injection.require() is _InventorySummary
    assert injection.get() is _InventorySummary

    with injection.apply({}):
        assert injection.require() is _InventorySummary

    with injection.apply(
        {
            "output:InventorySummary:description": "Optimized inventory output schema.",
            "output:InventorySummary:param:warehouses[].city": (
                "Warehouse city name for the shipment."
            ),
        }
    ):
        output_type = injection.require()
        output = output_type.model_validate(
            {
                "warehouses": [
                    {
                        "city": "Istanbul",
                        "postal_code": "34000",
                    }
                ]
            }
        )
        schema = output_type.model_json_schema()

    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    warehouse_definition = definitions["_ShippingAddressCandidate"]
    assert isinstance(warehouse_definition, dict)
    warehouse_properties = warehouse_definition["properties"]
    assert isinstance(warehouse_properties, dict)
    city_schema = warehouse_properties["city"]
    assert isinstance(city_schema, dict)
    assert output_type is not _InventorySummary
    assert isinstance(output, _InventorySummary)
    assert schema["description"] == "Optimized inventory output schema."
    assert city_schema["description"] == "Warehouse city name for the shipment."
    assert injection.require() is _InventorySummary


def test_model_output_injection_updates_direct_and_nested_model_fields() -> None:
    order_injection = ModelOutputInjection(_OrderSummary, model_name="OrderSummary")

    with order_injection.apply(
        {
            "output:OrderSummary:param:status": "Current fulfillment state.",
            "output:OrderSummary:param:shipping.city": "Destination city name.",
        }
    ):
        output_type = order_injection.require()
        output = output_type.model_validate(
            {
                "status": "shipped",
                "shipping": {
                    "city": "Istanbul",
                    "postal_code": "34000",
                },
                "tags": ["priority"],
            }
        )
        schema = output_type.model_json_schema()

    properties = schema["properties"]
    assert isinstance(properties, dict)
    status_schema = properties["status"]
    assert isinstance(status_schema, dict)
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    shipping_definition = definitions["_ShippingAddressCandidate"]
    assert isinstance(shipping_definition, dict)
    shipping_properties = shipping_definition["properties"]
    assert isinstance(shipping_properties, dict)
    city_schema = shipping_properties["city"]
    assert isinstance(city_schema, dict)
    assert isinstance(output, _OrderSummary)
    assert status_schema["description"] == "Current fulfillment state."
    assert city_schema["description"] == "Destination city name."


def test_model_output_injection_ignores_nested_overrides_for_unsupported_annotations() -> None:
    components = ComponentCatalog.from_components(
        [
            CandidateComponent(
                name="output:MapOutput:param:payload.key",
                initial_text="Payload entry.",
                path="payload.key",
            )
        ]
    )
    injection = ModelOutputInjection(
        _MapOutput,
        model_name="MapOutput",
        components=components,
    )

    with injection.apply({"output:MapOutput:param:payload.key": "Specific payload key."}):
        assert injection.require() is _MapOutput
    assert (
        injections._candidate_annotation(dict[str, str], {"key": "Specific payload key."})
        == (dict[str, str])
    )
    assert injections._candidate_annotation(list[str], {"item": "Tag item."}) == list[str]
    assert injections._nested_description_overrides({"tags[]": "Tag item."}, "tags") == {}


def test_model_schema_helpers_tolerate_sparse_invalid_and_missing_refs() -> None:
    malformed_schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {
            "broken": "not-a-schema",
            "missing_ref": {"$ref": "#/$defs/Unknown"},
            "nested_array": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string", "description": "Name"}},
                },
            },
        },
    }

    descriptions = iter_model_descriptions(
        malformed_schema,
        model_name="Broken",
        target="output",
        include_field_name_fallback=True,
    )
    empty_descriptions = iter_model_descriptions(
        {"type": "object", "properties": []},
        model_name="Empty",
        target="output",
        include_field_name_fallback=True,
    )

    assert descriptions == [
        schema_components.SchemaDescription(
            key="output:Broken:param:missing_ref",
            path=("missing_ref",),
            text="missing_ref",
        ),
        schema_components.SchemaDescription(
            key="output:Broken:param:nested_array[].name",
            path=("nested_array", "[]", "name"),
            text="Name",
        ),
    ]
    assert empty_descriptions == []
    assert schema_components._schema_has_properties(None, {}) is False
    assert schema_components._resolved_schema({"$ref": "#/$defs/Missing"}, {}) is None


def test_apply_model_schema_candidate_skips_unchanged_descriptions() -> None:
    catalog = collect_model_components(_OrderSummary, model_name="OrderSummary")
    unchanged = apply_model_schema_candidate(
        _OrderSummary,
        {"output:OrderSummary:param:status": "Order status"},
        model_name="OrderSummary",
        components=catalog,
    )

    assert unchanged.description_overrides == {}


def test_copy_json_helpers_preserve_value_shape_without_aliasing() -> None:
    source: dict[str, JsonValue] = {"nested": {"items": [1, {"name": "value"}]}}
    copied = copy_json_object(source)
    copied_value = copy_json(source)

    assert copied == source
    assert copied_value == source
    assert copied is not source
    assert copied["nested"] is not source["nested"]


@dataclass(frozen=True)
class _ToolDefinition:
    name: str
    description: str | None
    parameters_json_schema: dict[str, JsonValue]


class _ShippingAddress(BaseModel):
    city: str = Field(description="City to ship to")
    postal_code: str


class _OrderSummary(BaseModel):
    status: str = Field(description="Order status")
    shipping: _ShippingAddress
    tags: list[str]


class _InventorySummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={"description": "Inventory summary schema"})

    warehouses: list[_ShippingAddress]


class _MapOutput(BaseModel):
    payload: dict[str, str]
