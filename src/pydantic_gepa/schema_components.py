from __future__ import annotations as _annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .candidates import CandidateComponent
from .components import ComponentCatalog
from .values import JsonValue

SchemaComponentTarget = Literal["tool", "output"]


@runtime_checkable
class ToolDefinitionView(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str | None: ...

    @property
    def parameters_json_schema(self) -> Mapping[str, JsonValue]: ...


class SchemaDescription(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    path: tuple[str, ...]
    text: str

    @property
    def formatted_path(self) -> str:
        return format_schema_path(self.path)


@dataclass(frozen=True)
class ToolSchemaCandidate:
    name: str
    description: str | None = None
    parameters_json_schema: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSchemaCandidate:
    name: str
    description: str | None = None
    json_schema: dict[str, JsonValue] = field(default_factory=dict)
    description_overrides: dict[str, str] = field(default_factory=dict)


def collect_tool_components(
    tool_def: ToolDefinitionView,
    *,
    target: SchemaComponentTarget = "tool",
    serialization: Literal["raw", "json_string"] = "raw",
) -> ComponentCatalog:
    components: list[CandidateComponent] = []
    if tool_def.description is not None and tool_def.description.strip():
        components.append(
            CandidateComponent(
                name=description_key(tool_def.name, target=target),
                initial_text=tool_def.description,
                kind="tool_schema" if target == "tool" else "output_schema",
                semantic_type=f"{target}.description",
                source=tool_def.name,
                path="description",
                serialization=serialization,
            )
        )
    for description in iter_schema_descriptions(
        tool_def.parameters_json_schema,
        tool_name=tool_def.name,
        target=target,
    ):
        components.append(
            CandidateComponent(
                name=description.key,
                initial_text=description.text,
                kind="schema_description",
                semantic_type=f"{target}.parameter.description",
                source=tool_def.name,
                path=description.formatted_path,
                serialization=serialization,
            )
        )
    return ComponentCatalog.from_components(components)


def collect_toolset_components(
    tool_defs: Iterable[ToolDefinitionView],
    *,
    target: SchemaComponentTarget = "tool",
    serialization: Literal["raw", "json_string"] = "raw",
) -> ComponentCatalog:
    catalog = ComponentCatalog()
    for tool_def in tool_defs:
        catalog = catalog.merge(
            collect_tool_components(
                tool_def,
                target=target,
                serialization=serialization,
            )
        )
    return catalog


def collect_model_components(
    model_type: type[BaseModel],
    *,
    model_name: str | None = None,
    target: SchemaComponentTarget = "output",
    include_field_name_fallback: bool = True,
    serialization: Literal["raw", "json_string"] = "raw",
) -> ComponentCatalog:
    active_model_name = model_name or model_type.__name__
    schema = copy_json_object(model_type.model_json_schema())
    components: list[CandidateComponent] = []
    description = schema.get("description")
    if isinstance(description, str) and description.strip():
        components.append(
            CandidateComponent(
                name=description_key(active_model_name, target=target),
                initial_text=description,
                kind="output_schema" if target == "output" else "tool_schema",
                semantic_type=f"{target}.description",
                source=active_model_name,
                path="description",
                serialization=serialization,
            )
        )
    for field_description in iter_model_descriptions(
        schema,
        model_name=active_model_name,
        target=target,
        include_field_name_fallback=include_field_name_fallback,
    ):
        components.append(
            CandidateComponent(
                name=field_description.key,
                initial_text=field_description.text,
                kind="field_description",
                semantic_type=f"{target}.field.description",
                source=active_model_name,
                path=field_description.formatted_path,
                serialization=serialization,
            )
        )
    return ComponentCatalog.from_components(components)


def apply_tool_schema_candidate(
    tool_def: ToolDefinitionView,
    candidate: Mapping[str, str],
    *,
    target: SchemaComponentTarget = "tool",
    components: ComponentCatalog | None = None,
) -> ToolSchemaCandidate:
    active_components = components or collect_tool_components(tool_def, target=target)
    component_by_name = {component.name: component for component in active_components.components}

    description = tool_def.description
    description_component = component_by_name.get(description_key(tool_def.name, target=target))
    if description_component is not None and description_component.name in candidate:
        description = description_component.decode(candidate[description_component.name])

    schema = copy_json_object(tool_def.parameters_json_schema)
    for component in active_components.components:
        if component.path is None or component.name not in candidate:
            continue
        if component.path == "description":
            continue
        set_schema_description(
            schema,
            parse_schema_path(component.path),
            component.decode(candidate[component.name]),
        )

    return ToolSchemaCandidate(
        name=tool_def.name,
        description=description,
        parameters_json_schema=schema,
    )


def apply_model_schema_candidate(
    model_type: type[BaseModel],
    candidate: Mapping[str, str],
    *,
    model_name: str | None = None,
    target: SchemaComponentTarget = "output",
    include_field_name_fallback: bool = True,
    components: ComponentCatalog | None = None,
) -> ModelSchemaCandidate:
    active_model_name = model_name or model_type.__name__
    active_components = components or collect_model_components(
        model_type,
        model_name=active_model_name,
        target=target,
        include_field_name_fallback=include_field_name_fallback,
    )
    component_by_name = {component.name: component for component in active_components.components}
    schema = copy_json_object(model_type.model_json_schema())

    description = schema.get("description")
    if not isinstance(description, str):
        description = None
    description_component = component_by_name.get(description_key(active_model_name, target=target))
    if description_component is not None and description_component.name in candidate:
        description = description_component.decode(candidate[description_component.name])
        schema["description"] = description

    description_overrides: dict[str, str] = {}
    for component in active_components.components:
        if component.path is None or component.name not in candidate:
            continue
        if component.path == "description":
            continue
        override = component.decode(candidate[component.name])
        if set_schema_description(
            schema,
            parse_schema_path(component.path),
            override,
        ):
            description_overrides[component.path] = override

    return ModelSchemaCandidate(
        name=active_model_name,
        description=description,
        json_schema=schema,
        description_overrides=description_overrides,
    )


def iter_schema_descriptions(
    schema: Mapping[str, JsonValue],
    *,
    tool_name: str,
    target: SchemaComponentTarget,
) -> list[SchemaDescription]:
    return [
        SchemaDescription(
            key=parameter_key(tool_name, path, target=target),
            path=path,
            text=text,
        )
        for path, text in _iter_schema_descriptions(schema)
    ]


def iter_model_descriptions(
    schema: Mapping[str, JsonValue],
    *,
    model_name: str,
    target: SchemaComponentTarget,
    include_field_name_fallback: bool,
) -> list[SchemaDescription]:
    return [
        SchemaDescription(
            key=parameter_key(model_name, path, target=target),
            path=path,
            text=text,
        )
        for path, text in _iter_model_descriptions(
            schema,
            include_field_name_fallback=include_field_name_fallback,
        )
    ]


def description_key(tool_name: str, *, target: SchemaComponentTarget) -> str:
    return f"{target}:{tool_name}:description"


def parameter_key(
    tool_name: str,
    path: tuple[str, ...],
    *,
    target: SchemaComponentTarget,
) -> str:
    return f"{target}:{tool_name}:param:{format_schema_path(path)}"


def format_schema_path(path: tuple[str, ...]) -> str:
    formatted: list[str] = []
    for segment in path:
        if segment == "[]":
            if formatted:
                formatted[-1] = f"{formatted[-1]}[]"
            else:
                formatted.append("[]")
        else:
            formatted.append(segment)
    return ".".join(formatted)


def parse_schema_path(path: str) -> tuple[str, ...]:
    if path == "":
        return ()
    parsed: list[str] = []
    for raw_segment in path.split("."):
        if raw_segment.endswith("[]") and raw_segment != "[]":
            parsed.append(raw_segment[:-2])
            parsed.append("[]")
        else:
            parsed.append(raw_segment)
    return tuple(parsed)


def set_schema_description(
    schema: dict[str, JsonValue],
    path: tuple[str, ...],
    value: str,
) -> bool:
    definitions = _schema_definitions(schema)
    target: dict[str, JsonValue] = schema
    for segment in path:
        next_target = _next_schema_object(target, segment, definitions)
        if next_target is None:
            return False
        target = next_target
    if target.get("description") == value:
        return False
    target["description"] = value
    return True


def copy_json_object(value: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {str(key): copy_json(item) for key, item in value.items()}


def copy_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {str(key): copy_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_json(item) for item in value]
    return value


def _iter_schema_descriptions(
    schema: Mapping[str, JsonValue],
    path: tuple[str, ...] = (),
) -> list[tuple[tuple[str, ...], str]]:
    descriptions: list[tuple[tuple[str, ...], str]] = []
    description = schema.get("description")
    if path and isinstance(description, str) and description.strip():
        descriptions.append((path, description))

    schema_type = schema.get("type")
    if schema_type == "object":
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for name, subschema in properties.items():
                if isinstance(subschema, dict):
                    descriptions.extend(_iter_schema_descriptions(subschema, (*path, name)))
    if schema_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            descriptions.extend(_iter_schema_descriptions(items, (*path, "[]")))
    return descriptions


def _iter_model_descriptions(
    schema: Mapping[str, JsonValue],
    *,
    include_field_name_fallback: bool,
    path: tuple[str, ...] = (),
    definitions: Mapping[str, JsonValue] | None = None,
) -> list[tuple[tuple[str, ...], str]]:
    definitions_map = definitions or _schema_definitions(schema)
    descriptions: list[tuple[tuple[str, ...], str]] = []
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, subschema in properties.items():
            if not isinstance(subschema, dict):
                continue
            field_path = (*path, str(name))
            descriptions.extend(
                _model_description_entries(
                    subschema,
                    field_name=str(name),
                    path=field_path,
                    include_field_name_fallback=include_field_name_fallback,
                    definitions=definitions_map,
                )
            )
    return descriptions


def _model_description_entries(
    schema: Mapping[str, JsonValue],
    *,
    field_name: str,
    path: tuple[str, ...],
    include_field_name_fallback: bool,
    definitions: Mapping[str, JsonValue],
) -> list[tuple[tuple[str, ...], str]]:
    descriptions: list[tuple[tuple[str, ...], str]] = []
    description = schema.get("description")
    resolved_schema = _resolved_schema(schema, definitions) or schema
    has_nested_properties = _schema_has_properties(resolved_schema, definitions)
    items = resolved_schema.get("items")
    item_object = _as_json_mapping(items)
    resolved_items = _resolved_schema(item_object, definitions) if item_object is not None else None
    item_schema = resolved_items if resolved_items is not None else item_object
    item_has_nested_properties = (
        _schema_has_properties(item_schema, definitions)
        if isinstance(item_schema, Mapping)
        else False
    )

    if isinstance(description, str) and description.strip():
        descriptions.append((path, description))
    elif (
        include_field_name_fallback
        and resolved_schema.get("type") == "array"
        and not item_has_nested_properties
    ):
        descriptions.append(((*path, "[]"), field_name))
    elif (
        include_field_name_fallback and not has_nested_properties and not item_has_nested_properties
    ):
        descriptions.append((path, field_name))

    if has_nested_properties:
        descriptions.extend(
            _iter_model_descriptions(
                resolved_schema,
                include_field_name_fallback=include_field_name_fallback,
                path=path,
                definitions=definitions,
            )
        )

    if item_schema is not None and item_has_nested_properties:
        item_path = (*path, "[]")
        descriptions.extend(
            _iter_model_descriptions(
                item_schema,
                include_field_name_fallback=include_field_name_fallback,
                path=item_path,
                definitions=definitions,
            )
        )
    return descriptions


def _next_schema_object(
    schema: Mapping[str, JsonValue],
    segment: str,
    definitions: Mapping[str, JsonValue],
) -> dict[str, JsonValue] | None:
    resolved_schema = _resolved_schema(schema, definitions) or schema
    if segment == "[]":
        items = resolved_schema.get("items")
        item_object = _as_json_object(items)
        if item_object is None:
            return None
        resolved_items = _resolved_schema(item_object, definitions)
        return _as_json_object(resolved_items) if resolved_items is not None else item_object

    properties = resolved_schema.get("properties")
    if not isinstance(properties, dict):
        return None
    subschema = properties.get(segment)
    subschema_object = _as_json_object(subschema)
    if subschema_object is None:
        return None
    resolved_subschema = _resolved_schema(subschema_object, definitions)
    return (
        _as_json_object(resolved_subschema) if resolved_subschema is not None else subschema_object
    )


def _schema_definitions(schema: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    definitions = schema.get("$defs")
    if isinstance(definitions, dict):
        return definitions
    return {}


def _schema_has_properties(
    schema: Mapping[str, JsonValue] | None,
    definitions: Mapping[str, JsonValue],
) -> bool:
    if schema is None:
        return False
    resolved_schema = _resolved_schema(schema, definitions) or schema
    properties = resolved_schema.get("properties")
    return isinstance(properties, dict)


def _as_json_mapping(value: Any) -> Mapping[str, JsonValue] | None:
    if isinstance(value, Mapping):
        return cast("Mapping[str, JsonValue]", value)
    return None


def _as_json_object(value: Any) -> dict[str, JsonValue] | None:
    if isinstance(value, dict):
        return cast("dict[str, JsonValue]", value)
    return None


def _resolved_schema(
    schema: Mapping[str, JsonValue],
    definitions: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue] | None:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/$defs/"):
        return None
    definition_name = reference.removeprefix("#/$defs/")
    definition = definitions.get(definition_name)
    if isinstance(definition, dict):
        return definition
    return None


__all__ = (
    "SchemaComponentTarget",
    "SchemaDescription",
    "ModelSchemaCandidate",
    "ToolDefinitionView",
    "ToolSchemaCandidate",
    "apply_model_schema_candidate",
    "apply_tool_schema_candidate",
    "collect_model_components",
    "collect_tool_components",
    "collect_toolset_components",
    "copy_json",
    "copy_json_object",
    "description_key",
    "format_schema_path",
    "iter_model_descriptions",
    "iter_schema_descriptions",
    "parameter_key",
    "parse_schema_path",
    "set_schema_description",
)
