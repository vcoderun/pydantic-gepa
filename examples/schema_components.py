from __future__ import annotations as _annotations

from dataclasses import dataclass

from pydantic_gepa import (
    apply_tool_schema_candidate,
    collect_tool_components,
)
from pydantic_gepa.values import JsonValue


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str | None
    parameters_json_schema: dict[str, JsonValue]


def main() -> None:
    search_tool = ToolDefinition(
        name="search",
        description="Search the document index.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Maximum result count"},
            },
        },
    )

    catalog = collect_tool_components(search_tool)
    candidate = catalog.to_candidate()
    candidate = candidate.model_copy(
        update={
            "values": {
                **candidate.values,
                "tool:search:param:query": "Precise natural-language search query.",
            }
        }
    )

    updated = apply_tool_schema_candidate(search_tool, candidate.to_gepa_dict())
    properties = updated.parameters_json_schema["properties"]
    assert isinstance(properties, dict)
    query_schema = properties["query"]
    assert isinstance(query_schema, dict)
    description = query_schema["description"]
    assert isinstance(description, str)

    print(catalog.names())
    print(description)


if __name__ == "__main__":
    main()
