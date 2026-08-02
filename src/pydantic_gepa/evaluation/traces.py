from __future__ import annotations as _annotations

from pydantic import BaseModel, ConfigDict, Field

from ..values import JsonValue


class ErrorInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    message: str

    @classmethod
    def from_exception(cls, error: BaseException) -> ErrorInfo:
        return cls(kind=type(error).__qualname__, message=str(error))


class ComponentTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    component: str = Field(min_length=1)
    kind: str | None = None
    parent_id: str | None = None
    input: JsonValue = None
    output: JsonValue = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    duration_seconds: float | None = Field(default=None, ge=0)
    error: ErrorInfo | None = None


__all__ = (
    "ComponentTrace",
    "ErrorInfo",
)
