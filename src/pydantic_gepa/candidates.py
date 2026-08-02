from __future__ import annotations as _annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .errors import CandidateComponentError

MetadataValue = str | int | float | bool | None

SerializationMode = Literal["raw", "json_string"]
ComponentKind = Literal[
    "instructions",
    "system_prompt",
    "input_schema",
    "output_schema",
    "tool_schema",
    "field_description",
    "schema_description",
    "custom",
]


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: dict[str, str] = Field(default_factory=dict)
    id: str | None = None
    parent_id: str | None = None
    generation: int | None = None
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)

    def to_gepa_dict(self) -> dict[str, str]:
        return dict(self.values)

    def fingerprint(self) -> str:
        payload = json.dumps(
            self.values,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(payload.encode()).hexdigest()

    def save_yaml(self, path: str | Path) -> None:
        destination = Path(path)
        payload = self.model_dump(mode="json")
        with destination.open("w", encoding="utf-8") as file:
            yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, path: str | Path) -> Self:
        source = Path(path)
        with source.open(encoding="utf-8") as file:
            loaded = yaml.safe_load(file)
        if loaded is None:
            return cls()
        if not isinstance(loaded, dict):
            raise CandidateComponentError("Candidate YAML must contain a mapping.")
        return cls.model_validate(loaded)

    @classmethod
    def from_gepa_dict(
        cls,
        values: dict[str, str],
        *,
        candidate_id: str | None = None,
        parent_id: str | None = None,
        generation: int | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> Candidate:
        return cls(
            values=dict(values),
            id=candidate_id,
            parent_id=parent_id,
            generation=generation,
            metadata=metadata or {},
        )


class InstructionsCandidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    instructions: str

    def to_candidate(self, *, component: str = "instructions") -> Candidate:
        candidate_component = CandidateComponent(name=component, initial_text=self.instructions)
        return Candidate(values={component: candidate_component.initial_value})


class CandidateComponent(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    initial_text: str
    kind: ComponentKind = "custom"
    semantic_type: str | None = None
    asset_ref: str | None = None
    injection_target: str | None = None
    source: str | None = None
    path: str | None = None
    optimizable: bool = True
    serialization: SerializationMode = "raw"
    coupled_components: tuple[str, ...] = ()

    @property
    def initial_value(self) -> str:
        return self.encode(self.initial_text)

    def encode(self, value: str) -> str:
        if self.serialization == "raw":
            return value
        return json.dumps(value)

    def decode(self, value: str) -> str:
        if self.serialization == "raw":
            return value
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CandidateComponentError(
                f"Candidate component '{self.name}' is not valid JSON."
            ) from exc
        if not isinstance(decoded, str):
            raise CandidateComponentError(
                f"Candidate component '{self.name}' must decode to a string."
            )
        return decoded


Component = CandidateComponent


__all__ = (
    "Candidate",
    "CandidateComponent",
    "Component",
    "ComponentKind",
    "InstructionsCandidate",
    "MetadataValue",
    "SerializationMode",
)
