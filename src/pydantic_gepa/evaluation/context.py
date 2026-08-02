from __future__ import annotations as _annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Generic, TypeVar
from uuid import uuid4

from ..candidates import Candidate
from ..harness import run_awaitable_sync
from ..runtime import Runtime
from .evidence import Attachment, Encoder
from .models import Example
from .traces import ComponentTrace, ErrorInfo

InputsT = TypeVar("InputsT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")
CapturedT = TypeVar("CapturedT")


@dataclass(frozen=True, slots=True)
class Invocation(Generic[InputsT, OutputT]):
    id: str
    inputs: InputsT
    output: OutputT | None
    duration_seconds: float
    error: ErrorInfo | None = None


@dataclass(slots=True)
class Context(Generic[InputsT, OutputT, MetadataT]):
    runtime: Runtime[InputsT, OutputT]
    example: Example[InputsT, OutputT, MetadataT]
    candidate: Candidate
    run_id: str = field(default_factory=lambda: uuid4().hex)
    stage_id: str | None = None
    case_id: str | None = None
    attempt: int = 1
    active_components: tuple[str, ...] = ()
    frozen_components: tuple[str, ...] = ()
    encoder: Encoder = field(default_factory=Encoder)
    _invocations: list[Invocation[InputsT, OutputT]] = field(default_factory=list, init=False)
    _traces: list[ComponentTrace] = field(default_factory=list, init=False)
    _artifacts: list[Attachment] = field(default_factory=list, init=False)
    _task_errors: list[BaseException] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.attempt < 1:
            raise ValueError("attempt must be at least 1.")
        if self.case_id is None:
            self.case_id = self.example.name or uuid4().hex

    @property
    def invocations(self) -> tuple[Invocation[InputsT, OutputT], ...]:
        return tuple(self._invocations)

    @property
    def traces(self) -> tuple[ComponentTrace, ...]:
        return tuple(self._traces)

    @property
    def artifacts(self) -> tuple[Attachment, ...]:
        return tuple(self._artifacts)

    @property
    def task_errors(self) -> tuple[ErrorInfo, ...]:
        return tuple(ErrorInfo.from_exception(error) for error in self._task_errors)

    @property
    def output(self) -> OutputT | None:
        return self._invocations[-1].output if self._invocations else None

    def task_failed_with(self, error: BaseException) -> bool:
        return any(task_error is error for task_error in self._task_errors)

    async def arun(self) -> OutputT:
        return await self.arun_with(self.example.inputs)

    async def arun_with(self, inputs: InputsT) -> OutputT:
        invocation_id = uuid4().hex
        started = perf_counter()
        try:
            output = await self.runtime.arun(self.candidate, inputs)
        except asyncio.CancelledError as error:
            self._task_errors.append(error)
            self._record_invocation(invocation_id, inputs, started, error=error)
            raise
        except Exception as error:
            self._task_errors.append(error)
            self._record_invocation(invocation_id, inputs, started, error=error)
            raise
        self._record_invocation(invocation_id, inputs, started, output=output)
        return output

    def run(self) -> OutputT:
        return run_awaitable_sync(self.arun())

    def run_with(self, inputs: InputsT) -> OutputT:
        return run_awaitable_sync(self.arun_with(inputs))

    def artifact(self, attachment: Attachment) -> Attachment:
        self._artifacts.append(attachment)
        return attachment

    def capture(
        self,
        component: str,
        call: Callable[[], CapturedT],
        *,
        kind: str | None = None,
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> CapturedT:
        started = perf_counter()
        try:
            output = call()
        except Exception as error:
            self.record_trace(
                component,
                kind=kind,
                input=input,
                metadata=metadata,
                duration_seconds=perf_counter() - started,
                error=error,
                parent_id=parent_id,
            )
            raise
        self.record_trace(
            component,
            kind=kind,
            input=input,
            output=output,
            metadata=metadata,
            duration_seconds=perf_counter() - started,
            parent_id=parent_id,
        )
        return output

    async def acapture(
        self,
        component: str,
        call: Callable[[], Awaitable[CapturedT]],
        *,
        kind: str | None = None,
        input: Any = None,
        metadata: Mapping[str, Any] | None = None,
        parent_id: str | None = None,
    ) -> CapturedT:
        started = perf_counter()
        try:
            output = await call()
        except asyncio.CancelledError as error:
            self.record_trace(
                component,
                kind=kind,
                input=input,
                metadata=metadata,
                duration_seconds=perf_counter() - started,
                error=error,
                parent_id=parent_id,
            )
            raise
        except Exception as error:
            self.record_trace(
                component,
                kind=kind,
                input=input,
                metadata=metadata,
                duration_seconds=perf_counter() - started,
                error=error,
                parent_id=parent_id,
            )
            raise
        self.record_trace(
            component,
            kind=kind,
            input=input,
            output=output,
            metadata=metadata,
            duration_seconds=perf_counter() - started,
            parent_id=parent_id,
        )
        return output

    def record_trace(
        self,
        component: str,
        *,
        kind: str | None = None,
        input: Any = None,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        duration_seconds: float | None = None,
        error: BaseException | None = None,
        parent_id: str | None = None,
        trace_id: str | None = None,
    ) -> ComponentTrace:
        trace = ComponentTrace(
            id=trace_id or uuid4().hex,
            component=component,
            kind=kind,
            parent_id=parent_id,
            input=self.encoder.encode(input),
            output=self.encoder.encode(output),
            metadata=self.encoder.mapping(metadata or {}),
            duration_seconds=duration_seconds,
            error=ErrorInfo.from_exception(error) if error is not None else None,
        )
        self._traces.append(trace)
        return trace

    def _record_invocation(
        self,
        invocation_id: str,
        inputs: InputsT,
        started: float,
        *,
        output: OutputT | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._invocations.append(
            Invocation(
                id=invocation_id,
                inputs=inputs,
                output=output,
                duration_seconds=perf_counter() - started,
                error=ErrorInfo.from_exception(error) if error is not None else None,
            )
        )


__all__ = (
    "Context",
    "Invocation",
)
