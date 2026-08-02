from __future__ import annotations as _annotations

from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .candidates import Candidate, CandidateComponent

SelectorMode = Literal["exact", "prefix"]


class ComponentSelector(BaseModel):
    model_config = ConfigDict(frozen=True)

    include: tuple[str, ...] | None = None
    exclude: tuple[str, ...] = ()
    mode: SelectorMode = "prefix"

    def accepts(self, component_name: str) -> bool:
        if self.include is not None and not self._matches_any(component_name, self.include):
            return False
        return not self._matches_any(component_name, self.exclude)

    def _matches_any(self, component_name: str, patterns: tuple[str, ...]) -> bool:
        return any(self._matches(component_name, pattern) for pattern in patterns)

    def _matches(self, component_name: str, pattern: str) -> bool:
        if self.mode == "exact":
            return component_name == pattern
        return (
            component_name == pattern
            or component_name.startswith(f"{pattern}.")
            or component_name.startswith(f"{pattern}:")
        )


class ComponentCatalog(BaseModel):
    model_config = ConfigDict(frozen=True)

    components: tuple[CandidateComponent, ...] = Field(default_factory=tuple)

    @classmethod
    def from_components(cls, components: Iterable[CandidateComponent]) -> ComponentCatalog:
        by_name: dict[str, CandidateComponent] = {}
        for component in components:
            by_name[component.name] = component
        return cls(components=tuple(by_name.values()))

    def names(self) -> list[str]:
        return [component.name for component in self.components]

    def values(self) -> dict[str, str]:
        return {
            component.name: component.initial_value
            for component in self.components
            if component.optimizable
        }

    def to_candidate(
        self,
        *,
        candidate_id: str | None = None,
        metadata: Mapping[str, str | int | float | bool | None] | None = None,
    ) -> Candidate:
        return Candidate.from_gepa_dict(
            self.values(),
            candidate_id=candidate_id,
            metadata=dict(metadata or {}),
        )

    def select(
        self,
        *,
        include: Iterable[str] | None = None,
        exclude: Iterable[str] = (),
        mode: SelectorMode = "prefix",
    ) -> ComponentCatalog:
        selector = ComponentSelector(
            include=tuple(include) if include is not None else None,
            exclude=tuple(exclude),
            mode=mode,
        )
        return ComponentCatalog(
            components=tuple(
                component for component in self.components if selector.accepts(component.name)
            )
        )

    def merge(self, other: ComponentCatalog) -> ComponentCatalog:
        return ComponentCatalog.from_components((*self.components, *other.components))


__all__ = (
    "ComponentCatalog",
    "ComponentSelector",
    "SelectorMode",
)
