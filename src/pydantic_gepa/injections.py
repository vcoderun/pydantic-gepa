from __future__ import annotations as _annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import GenericAlias
from typing import Annotated, Generic, Protocol, TypeVar, get_args, get_origin, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, create_model

from .candidates import CandidateComponent
from .components import ComponentCatalog
from .errors import CandidateInjectionError
from .schema_components import (
    SchemaComponentTarget,
    apply_model_schema_candidate,
    collect_model_components,
    format_schema_path,
    parse_schema_path,
)

ValueT = TypeVar("ValueT")
ModelT = TypeVar("ModelT", bound=BaseModel)


@runtime_checkable
class CandidateInjection(Protocol):
    component: str

    def apply(self, candidate: Mapping[str, str]) -> AbstractContextManager[None]: ...


@runtime_checkable
class InstructionOverrideAgent(Protocol):
    def override(self, *, instructions: str) -> AbstractContextManager[None]: ...


class CandidateContext(Generic[ValueT]):
    def __init__(self, name: str, default: ValueT | None = None) -> None:
        self.name = name
        self.default = default
        self._context_var: ContextVar[ValueT | None] = ContextVar(name, default=default)

    def get(self) -> ValueT | None:
        return self._context_var.get()

    def require(self) -> ValueT:
        value = self.get()
        if value is None:
            raise CandidateInjectionError(f"Candidate context '{self.name}' has no active value.")
        return value

    @contextmanager
    def use(self, value: ValueT) -> Iterator[None]:
        token = self._context_var.set(value)
        try:
            yield
        finally:
            self._context_var.reset(token)


class AgentInstructionsInjection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent: InstructionOverrideAgent
    component: str = "instructions"
    candidate_component: CandidateComponent = Field(
        default_factory=lambda: CandidateComponent(name="instructions", initial_text="")
    )

    def model_post_init(self, __context: dict[str, str] | None) -> None:
        if self.candidate_component.name != self.component:
            self.candidate_component = self.candidate_component.model_copy(
                update={"name": self.component}
            )

    def apply(self, candidate: Mapping[str, str]) -> AbstractContextManager[None]:
        if self.component not in candidate:
            raise CandidateInjectionError(f"Candidate is missing component '{self.component}'.")
        instructions = self.candidate_component.decode(candidate[self.component])
        if not isinstance(self.agent, InstructionOverrideAgent):
            raise CandidateInjectionError("Agent does not expose an override() method.")
        context = self.agent.override(instructions=instructions)
        if not isinstance(context, AbstractContextManager):
            raise CandidateInjectionError("Agent override() must return a context manager.")
        return context


@dataclass(slots=True)
class DerivedValueInjection(Generic[ValueT]):
    component: str
    context: CandidateContext[ValueT]
    derive_value: Callable[[Mapping[str, str]], ValueT]
    required_components: tuple[str, ...] = field(default_factory=tuple)

    def apply(self, candidate: Mapping[str, str]) -> AbstractContextManager[None]:
        missing_components = [
            component for component in self.required_components if component not in candidate
        ]
        if missing_components:
            raise CandidateInjectionError(
                "Candidate is missing required components: "
                + ", ".join(sorted(missing_components))
                + "."
            )
        return self.context.use(self.derive_value(candidate))


class ModelOutputInjection(Generic[ModelT]):
    component: str
    model_type: type[ModelT]
    model_name: str
    target: SchemaComponentTarget
    include_field_name_fallback: bool
    components: ComponentCatalog
    context: CandidateContext[type[ModelT]]

    def __init__(
        self,
        model_type: type[ModelT],
        *,
        model_name: str | None = None,
        component: str = "output_schema",
        target: SchemaComponentTarget = "output",
        include_field_name_fallback: bool = True,
        components: ComponentCatalog | None = None,
        context: CandidateContext[type[ModelT]] | None = None,
    ) -> None:
        self.component = component
        self.model_type = model_type
        self.model_name = model_name or model_type.__name__
        self.target = target
        self.include_field_name_fallback = include_field_name_fallback
        self.components = components or collect_model_components(
            model_type,
            model_name=self.model_name,
            target=target,
            include_field_name_fallback=include_field_name_fallback,
        )
        self.context = context or CandidateContext[type[ModelT]](
            name=f"{component}.{self.model_name}",
            default=model_type,
        )

    def get(self) -> type[ModelT]:
        return self.context.require()

    def require(self) -> type[ModelT]:
        return self.context.require()

    def apply(self, candidate: Mapping[str, str]) -> AbstractContextManager[None]:
        return self.context.use(
            _candidate_model_output_type(
                self.model_type,
                candidate,
                model_name=self.model_name,
                target=self.target,
                include_field_name_fallback=self.include_field_name_fallback,
                components=self.components,
            )
        )


class NoopInjection(BaseModel):
    component: str

    def apply(self, candidate: Mapping[str, str]) -> AbstractContextManager[None]:
        if self.component not in candidate:
            raise CandidateInjectionError(f"Candidate is missing component '{self.component}'.")
        return nullcontext()


def _candidate_model_output_type(
    model_type: type[ModelT],
    candidate: Mapping[str, str],
    *,
    model_name: str,
    target: SchemaComponentTarget,
    include_field_name_fallback: bool,
    components: ComponentCatalog,
) -> type[ModelT]:
    schema_candidate = apply_model_schema_candidate(
        model_type,
        candidate,
        model_name=model_name,
        target=target,
        include_field_name_fallback=include_field_name_fallback,
        components=components,
    )
    field_definitions = _candidate_field_definitions(
        model_type,
        schema_candidate.description_overrides,
    )
    base_description = model_type.model_json_schema().get("description")
    active_base_description = base_description if isinstance(base_description, str) else None
    if not field_definitions and schema_candidate.description == active_base_description:
        return model_type

    if schema_candidate.description is not None:
        output_type = create_model(
            schema_candidate.name,
            __base__=model_type,
            __config__=ConfigDict(json_schema_extra={"description": schema_candidate.description}),
            __doc__=schema_candidate.description,
            __module__=model_type.__module__,
            **field_definitions,
        )
    else:
        output_type = create_model(
            schema_candidate.name,
            __base__=model_type,
            __module__=model_type.__module__,
            **field_definitions,
        )
    return output_type


def _candidate_field_definitions(
    model_type: type[BaseModel],
    description_overrides: Mapping[str, str],
):
    field_definitions = {}
    for field_name, field_info in model_type.model_fields.items():
        direct_description = description_overrides.get(field_name)
        nested_overrides = _nested_description_overrides(description_overrides, field_name)
        annotation = field_info.annotation
        candidate_annotation = _candidate_annotation(annotation, nested_overrides)
        if direct_description is None and candidate_annotation is annotation:
            continue
        field_definitions[field_name] = _candidate_field_definition(
            candidate_annotation,
            field_info,
            direct_description=direct_description,
        )
    return field_definitions


def _candidate_field_definition(
    annotation,
    field_info,
    *,
    direct_description: str | None,
):
    field_data = field_info.asdict()
    field_attributes = dict(field_data["attributes"])
    if direct_description is not None:
        field_attributes["description"] = direct_description
    return Annotated[
        annotation,
        *field_data["metadata"],
        Field(**field_attributes),
    ]


def _candidate_annotation(annotation, description_overrides: Mapping[str, str]):
    if not description_overrides:
        return annotation

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _nested_candidate_model_type(annotation, description_overrides)

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is list and len(args) == 1:
        item_annotation = args[0]
        if isinstance(item_annotation, type) and issubclass(item_annotation, BaseModel):
            nested_model = _nested_candidate_model_type(item_annotation, description_overrides)
            return GenericAlias(list, (nested_model,))
    return annotation


def _nested_candidate_model_type(
    model_type: type[BaseModel],
    description_overrides: Mapping[str, str],
) -> type[BaseModel]:
    field_definitions = _candidate_field_definitions(model_type, description_overrides)
    return create_model(
        f"{model_type.__name__}Candidate",
        __base__=model_type,
        __module__=model_type.__module__,
        **field_definitions,
    )


def _nested_description_overrides(
    description_overrides: Mapping[str, str],
    field_name: str,
) -> dict[str, str]:
    nested: dict[str, str] = {}
    for path, description in description_overrides.items():
        parsed_path = parse_schema_path(path)
        if len(parsed_path) < 2 or parsed_path[0] != field_name:
            continue
        nested_path = parsed_path[1:]
        if nested_path[0] == "[]":
            nested_path = nested_path[1:]
        if nested_path:
            nested[format_schema_path(nested_path)] = description
    return nested


__all__ = (
    "AgentInstructionsInjection",
    "CandidateContext",
    "CandidateInjection",
    "DerivedValueInjection",
    "ModelOutputInjection",
    "NoopInjection",
)
