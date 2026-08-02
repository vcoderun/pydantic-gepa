from __future__ import annotations as _annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field

from ..errors import EvidenceEncodingError
from ..values import JsonValue

ValueT = TypeVar("ValueT")
AttachmentKind = Literal["binary", "image", "audio", "video", "document"]
EvidenceValue: TypeAlias = JsonValue


class Attachment(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: AttachmentKind = "binary"
    reference: str | None = None
    media_type: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    digest: str | None = None

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        *,
        kind: AttachmentKind = "binary",
        reference: str | None = None,
        media_type: str | None = None,
    ) -> Attachment:
        return cls(
            kind=kind,
            reference=reference,
            media_type=media_type,
            size_bytes=len(content),
            digest=sha256(content).hexdigest(),
        )

    def as_evidence(self) -> dict[str, EvidenceValue]:
        return {
            "type": "attachment",
            "kind": self.kind,
            "reference": self.reference,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
        }


class Encoder:
    def __init__(
        self,
        *,
        max_depth: int = 8,
        max_items: int = 100,
        max_string_length: int = 4_000,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if max_items < 1:
            raise ValueError("max_items must be at least 1.")
        if max_string_length < 1:
            raise ValueError("max_string_length must be at least 1.")
        self.max_depth = max_depth
        self.max_items = max_items
        self.max_string_length = max_string_length
        self._encoders: list[_RegisteredEncoder[Any]] = []

    def register(
        self,
        value_type: type[ValueT],
        encode: Callable[[ValueT], EvidenceValue],
    ) -> None:
        self._encoders.append(
            _RegisteredEncoder(
                value_type=value_type,
                encode=cast("Callable[[Any], EvidenceValue]", encode),
            )
        )

    def encode(self, value: Any) -> EvidenceValue:
        return self._encode(value, depth=0, active=set())

    def mapping(self, values: Mapping[str, ValueT]) -> dict[str, EvidenceValue]:
        encoded = self._encode(values, depth=0, active=set())
        if not isinstance(encoded, dict):
            raise EvidenceEncodingError("Evidence mapping did not encode to a mapping.")
        return encoded

    def _encode(self, value: Any, *, depth: int, active: set[int]) -> EvidenceValue:
        for registered in self._encoders:
            if isinstance(value, registered.value_type):
                return registered.encode(value)

        if isinstance(value, Enum):
            if isinstance(value.value, str):
                return value.value
            return self._encode(value.value, depth=depth, active=active)
        if value is None or isinstance(value, bool | int):
            return value
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return {"type": "float", "value": str(value)}
        if isinstance(value, str):
            return self._string(value)
        if isinstance(value, Attachment):
            return value.as_evidence()
        if isinstance(value, bytes | bytearray):
            return Attachment.from_bytes(bytes(value)).as_evidence()
        if isinstance(value, Path):
            return {"type": "path", "path": str(value)}
        if isinstance(value, datetime | date):
            return value.isoformat()
        identity = id(value)
        if identity in active:
            return {"type": "cycle", "python_type": type(value).__qualname__}
        if depth >= self.max_depth:
            return {"type": "truncated", "reason": "max_depth"}

        if isinstance(value, BaseModel):
            active.add(identity)
            try:
                return self._encode(value.model_dump(mode="python"), depth=depth + 1, active=active)
            finally:
                active.remove(identity)
        if is_dataclass(value) and not isinstance(value, type):
            active.add(identity)
            try:
                return self._encode(
                    {field.name: getattr(value, field.name) for field in fields(value)},
                    depth=depth + 1,
                    active=active,
                )
            finally:
                active.remove(identity)
        if isinstance(value, Mapping):
            active.add(identity)
            try:
                return self._mapping(value, depth=depth + 1, active=active)
            finally:
                active.remove(identity)
        if isinstance(value, Sequence):
            active.add(identity)
            try:
                return self._sequence(value, depth=depth + 1, active=active)
            finally:
                active.remove(identity)

        raise EvidenceEncodingError(
            f"No evidence encoder is registered for '{type(value).__qualname__}'."
        )

    def _mapping(
        self,
        value: Mapping[Any, Any],
        *,
        depth: int,
        active: set[int],
    ) -> dict[str, EvidenceValue]:
        if any(not isinstance(key, str) for key in value):
            raise EvidenceEncodingError("Evidence mappings require string keys.")
        keys = sorted(key for key in value if isinstance(key, str))
        selected = keys[: self.max_items]
        encoded = {key: self._encode(value[key], depth=depth, active=active) for key in selected}
        if len(keys) > self.max_items:
            return {
                "type": "truncated_mapping",
                "items": encoded,
                "omitted": len(keys) - self.max_items,
            }
        return encoded

    def _sequence(
        self,
        value: Sequence[Any],
        *,
        depth: int,
        active: set[int],
    ) -> list[EvidenceValue] | dict[str, EvidenceValue]:
        selected = value[: self.max_items]
        encoded = [self._encode(item, depth=depth, active=active) for item in selected]
        if len(value) > self.max_items:
            return {
                "type": "truncated_sequence",
                "items": encoded,
                "omitted": len(value) - self.max_items,
            }
        return encoded

    def _string(self, value: str) -> str:
        if len(value) <= self.max_string_length:
            return value
        omitted = len(value) - self.max_string_length
        return f"{value[: self.max_string_length]}...[{omitted} chars omitted]"


class _RegisteredEncoder(BaseModel, Generic[ValueT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    value_type: type[ValueT]
    encode: Callable[[ValueT], EvidenceValue]


__all__ = (
    "Attachment",
    "AttachmentKind",
    "Encoder",
    "EvidenceValue",
)
