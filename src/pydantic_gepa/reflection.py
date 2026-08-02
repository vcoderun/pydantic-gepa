from __future__ import annotations as _annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from typing import Literal, Protocol, TypeAlias, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .harness import run_awaitable_sync
from .values import JsonValue

ReflectionMessage = Mapping[str, JsonValue]
ReflectionPrompt: TypeAlias = str | list[ReflectionMessage]


class ReflectionUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: int = Field(default=1, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0)


class ReflectionResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    usage: ReflectionUsage = Field(default_factory=ReflectionUsage)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ReflectionError(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    message: str


class ReflectionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt: ReflectionPrompt
    response: ReflectionResponse | None = None
    error: ReflectionError | None = None
    attempts: int = Field(ge=1)
    duration: float = Field(ge=0)


ReflectionOutput: TypeAlias = str | ReflectionResponse
ReflectionCall: TypeAlias = Callable[
    [ReflectionPrompt], ReflectionOutput | Awaitable[ReflectionOutput]
]
ReflectionFunction: TypeAlias = Callable[[ReflectionPrompt], str]
ReflectionFailure = Literal["raise", "empty"]


@runtime_checkable
class ReflectionModel(Protocol):
    def __call__(self, prompt: ReflectionPrompt) -> str: ...

    @property
    def total_cost(self) -> float: ...

    @property
    def total_tokens_in(self) -> int: ...

    @property
    def total_tokens_out(self) -> int: ...


@dataclass(slots=True)
class CallableReflectionModel:
    call: ReflectionCall
    retries: int = 0
    on_error: ReflectionFailure = "raise"
    _records: list[ReflectionRecord] = field(default_factory=list, init=False)
    _usage: ReflectionUsage = field(
        default_factory=lambda: ReflectionUsage(requests=0),
        init=False,
    )
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.retries < 0:
            raise ValueError("retries cannot be negative.")

    @property
    def records(self) -> tuple[ReflectionRecord, ...]:
        with self._lock:
            return tuple(self._records)

    @property
    def total_cost(self) -> float:
        with self._lock:
            return self._usage.cost

    @property
    def total_tokens_in(self) -> int:
        with self._lock:
            return self._usage.input_tokens

    @property
    def total_tokens_out(self) -> int:
        with self._lock:
            return self._usage.output_tokens

    def __call__(self, prompt: ReflectionPrompt) -> str:
        started = perf_counter()
        attempt = 0
        while True:
            attempt += 1
            try:
                output = self.call(prompt)
                if inspect.isawaitable(output):
                    output = run_awaitable_sync(cast("Awaitable[ReflectionOutput]", output))
                response = _response(output, prompt)
                self._record(
                    ReflectionRecord(
                        prompt=prompt,
                        response=response,
                        attempts=attempt,
                        duration=perf_counter() - started,
                    )
                )
                return response.text
            except Exception as exc:
                if attempt <= self.retries:
                    continue
                self._record(
                    ReflectionRecord(
                        prompt=prompt,
                        error=ReflectionError(type=type(exc).__name__, message=str(exc)),
                        attempts=attempt,
                        duration=perf_counter() - started,
                    )
                )
                if self.on_error == "empty":
                    return ""
                raise

    def _record(self, record: ReflectionRecord) -> None:
        with self._lock:
            self._records.append(record)
            if record.response is not None:
                usage = record.response.usage
                self._usage = ReflectionUsage(
                    requests=self._usage.requests + usage.requests,
                    input_tokens=self._usage.input_tokens + usage.input_tokens,
                    output_tokens=self._usage.output_tokens + usage.output_tokens,
                    cost=self._usage.cost + usage.cost,
                )


def _response(output: ReflectionOutput, prompt: ReflectionPrompt) -> ReflectionResponse:
    if isinstance(output, ReflectionResponse):
        return output
    input_size = len(prompt) if isinstance(prompt, str) else sum(len(str(item)) for item in prompt)
    return ReflectionResponse(
        text=output,
        usage=ReflectionUsage(
            input_tokens=max(1, input_size // 4),
            output_tokens=max(1, len(output) // 4),
        ),
    )


__all__ = (
    "CallableReflectionModel",
    "ReflectionCall",
    "ReflectionError",
    "ReflectionFailure",
    "ReflectionFunction",
    "ReflectionMessage",
    "ReflectionModel",
    "ReflectionOutput",
    "ReflectionPrompt",
    "ReflectionRecord",
    "ReflectionResponse",
    "ReflectionUsage",
)
