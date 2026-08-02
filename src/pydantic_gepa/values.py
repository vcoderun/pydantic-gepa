from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeAlias

from pydantic import BaseModel
from typing_extensions import TypeAliasType


class ReprSerializable(Protocol):
    def __repr__(self) -> str: ...


JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue = TypeAliasType(
    "JsonValue",
    JsonScalar | list["JsonValue"] | dict[str, "JsonValue"],
)
SerializableValue: TypeAlias = (
    JsonScalar
    | BaseModel
    | Mapping[str, "SerializableValue"]
    | Sequence["SerializableValue"]
    | ReprSerializable
)
Metadata: TypeAlias = Mapping[str, JsonScalar]


__all__ = (
    "JsonScalar",
    "JsonValue",
    "Metadata",
    "ReprSerializable",
    "SerializableValue",
)
