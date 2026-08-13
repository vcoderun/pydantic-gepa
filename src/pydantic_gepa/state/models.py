from __future__ import annotations as _annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ..candidates import Candidate
from ..values import JsonValue

RunStatus = Literal["running", "completed", "failed"]
ResultT = TypeVar("ResultT", bound=BaseModel)


class CompatibilityFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    digest: str
    dimensions: dict[str, str]

    @classmethod
    def from_dimensions(cls, dimensions: Mapping[str, str]) -> CompatibilityFingerprint:
        normalized = dict(sorted(dimensions.items()))
        payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        return cls(digest=sha256(payload.encode()).hexdigest(), dimensions=normalized)


def content_fingerprint(value: Any) -> str:
    payload = json.dumps(
        _json_content(value),
        default=repr,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(payload.encode()).hexdigest()


def _json_content(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _json_content(value.model_dump(mode="python"))
    if is_dataclass(value) and not isinstance(value, type):
        return _json_content(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_content(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_content(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_json_content(item) for item in value), key=repr)
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return repr(value)


class RunState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    fingerprint: CompatibilityFingerprint
    status: RunStatus = "running"
    next_stage: int = Field(default=0, ge=0)
    accepted_candidate: Candidate
    stages: tuple[JsonValue, ...] = ()
    metric_calls: int = Field(default=0, ge=0)
    backend_checkpoint: str | None = None
    error: str | None = None
    resumed: bool = Field(default=False, exclude=True)
    reset: bool = Field(default=False, exclude=True)


class RunStore(Protocol):
    @property
    def backend_directory(self) -> Path: ...

    def prepare(
        self,
        *,
        fingerprint: CompatibilityFingerprint,
        initial_candidate: Candidate,
    ) -> RunState: ...

    def checkpoint(self, state: RunState) -> None: ...

    def write_candidate(self, candidate: Candidate) -> Path: ...

    def write_stage(self, result: BaseModel) -> Path: ...

    def write_result(self, result: BaseModel) -> Path: ...

    def load_result(self, model: type[ResultT]) -> ResultT | None: ...

    def load_state(self) -> RunState | None: ...

    def reset(self) -> None: ...


__all__ = (
    "CompatibilityFingerprint",
    "ResultT",
    "RunState",
    "RunStatus",
    "RunStore",
    "content_fingerprint",
)
