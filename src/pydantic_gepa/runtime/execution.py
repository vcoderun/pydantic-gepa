from __future__ import annotations as _annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    ExitStack,
    asynccontextmanager,
)
from threading import BoundedSemaphore
from typing import Generic, Literal, TypeAlias, TypeVar, cast, overload

from ..candidates import Candidate
from ..errors import CandidateInjectionError
from ..harness import run_awaitable_sync
from ..injections import CandidateInjection

InputsT = TypeVar("InputsT")
OutputT = TypeVar("OutputT")
SubjectT = TypeVar("SubjectT")
ResolvedT = TypeVar("ResolvedT")

Isolation = Literal["context_local", "serialized", "factory"]
RuntimeTask: TypeAlias = Callable[[InputsT], OutputT | Awaitable[OutputT]]
CandidateScope: TypeAlias = Callable[
    [Candidate],
    AbstractContextManager[None] | AbstractAsyncContextManager[None],
]
NormalizeCandidate: TypeAlias = Callable[[Candidate], Candidate]
Factory: TypeAlias = Callable[[Candidate], SubjectT | Awaitable[SubjectT]]
FactoryTask: TypeAlias = Callable[[SubjectT, InputsT], OutputT | Awaitable[OutputT]]
SubjectCleanup: TypeAlias = Callable[[SubjectT], None | Awaitable[None]]
Invocation: TypeAlias = Callable[[Candidate, InputsT], Awaitable[OutputT]]


class Runtime(Generic[InputsT, OutputT]):
    @overload
    def __init__(
        self,
        run: Callable[[InputsT], Awaitable[OutputT]],
        *,
        scope: CandidateScope | None = None,
        required_components: Sequence[str] = (),
        isolation: Isolation | None = None,
        max_concurrency: int | None = None,
        normalize: NormalizeCandidate | None = None,
        identity: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        run: Callable[[InputsT], OutputT],
        *,
        scope: CandidateScope | None = None,
        required_components: Sequence[str] = (),
        isolation: Isolation | None = None,
        max_concurrency: int | None = None,
        normalize: NormalizeCandidate | None = None,
        identity: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        run: RuntimeTask[InputsT, OutputT],
        *,
        scope: CandidateScope | None = None,
        required_components: Sequence[str] = (),
        isolation: Isolation | None = None,
        max_concurrency: int | None = None,
        normalize: NormalizeCandidate | None = None,
        identity: str | None = None,
    ) -> None:
        if isolation == "factory":
            raise ValueError("Use Runtime.from_factory() for factory isolation.")
        active_isolation: Isolation = (
            ("serialized" if scope is not None else "context_local")
            if isolation is None
            else isolation
        )

        async def invoke(candidate: Candidate, inputs: InputsT) -> OutputT:
            async with _candidate_scope(scope, candidate):
                return await _resolve(run(inputs))

        self._configure(
            invoke,
            required_components=required_components,
            isolation=active_isolation,
            max_concurrency=max_concurrency,
            normalize=normalize,
            identity=identity or f"{type(run).__module__}.{type(run).__qualname__}:{id(run)}",
        )

    @classmethod
    @overload
    def from_factory(
        cls,
        create: Callable[[Candidate], SubjectT],
        run: FactoryTask[SubjectT, InputsT, OutputT],
        *,
        close: SubjectCleanup[SubjectT] | None = None,
        required_components: Sequence[str] = (),
        max_concurrency: int | None = None,
        normalize: NormalizeCandidate | None = None,
        identity: str | None = None,
    ) -> Runtime[InputsT, OutputT]: ...

    @classmethod
    @overload
    def from_factory(
        cls,
        create: Callable[[Candidate], Awaitable[SubjectT]],
        run: FactoryTask[SubjectT, InputsT, OutputT],
        *,
        close: SubjectCleanup[SubjectT] | None = None,
        required_components: Sequence[str] = (),
        max_concurrency: int | None = None,
        normalize: NormalizeCandidate | None = None,
        identity: str | None = None,
    ) -> Runtime[InputsT, OutputT]: ...

    @classmethod
    def from_factory(
        cls,
        create: Factory[SubjectT],
        run: FactoryTask[SubjectT, InputsT, OutputT],
        *,
        close: SubjectCleanup[SubjectT] | None = None,
        required_components: Sequence[str] = (),
        max_concurrency: int | None = None,
        normalize: NormalizeCandidate | None = None,
        identity: str | None = None,
    ) -> Runtime[InputsT, OutputT]:
        async def invoke(candidate: Candidate, inputs: InputsT) -> OutputT:
            subject = await _resolve(create(candidate))
            try:
                return await _resolve(run(subject, inputs))
            finally:
                if close is not None:
                    await _resolve(close(subject))

        runtime = cls.__new__(cls)
        runtime._configure(
            invoke,
            required_components=required_components,
            isolation="factory",
            max_concurrency=max_concurrency,
            normalize=normalize,
            identity=identity
            or (
                f"{type(create).__module__}.{type(create).__qualname__}:{id(create)}:"
                f"{type(run).__module__}.{type(run).__qualname__}:{id(run)}"
            ),
        )
        return runtime

    @classmethod
    def from_injections(
        cls,
        run: RuntimeTask[InputsT, OutputT],
        injections: Sequence[CandidateInjection],
        *,
        max_concurrency: int | None = None,
        normalize: NormalizeCandidate | None = None,
        identity: str | None = None,
    ) -> Runtime[InputsT, OutputT]:
        active_injections = tuple(injections)

        async def invoke(candidate: Candidate, inputs: InputsT) -> OutputT:
            with ExitStack() as stack:
                for injection in active_injections:
                    stack.enter_context(injection.apply(candidate.values))
                return await _resolve(run(inputs))

        runtime = cls.__new__(cls)
        runtime._configure(
            invoke,
            required_components=tuple(
                dict.fromkeys(injection.component for injection in active_injections)
            ),
            isolation="serialized",
            max_concurrency=max_concurrency,
            normalize=normalize,
            identity=identity or f"{type(run).__module__}.{type(run).__qualname__}:{id(run)}",
        )
        return runtime

    @property
    def isolation(self) -> Isolation:
        return self._isolation

    @property
    def max_concurrency(self) -> int | None:
        return self._max_concurrency

    @property
    def required_components(self) -> tuple[str, ...]:
        return self._required_components

    @property
    def identity(self) -> str:
        return self._identity

    def normalize_candidate(self, candidate: Candidate) -> Candidate:
        return self._prepare_candidate(candidate)

    async def arun(self, candidate: Candidate, inputs: InputsT) -> OutputT:
        active_candidate = self.normalize_candidate(candidate)
        await self._acquire()
        try:
            return await self._invoke(active_candidate, inputs)
        finally:
            if self._capacity is not None:
                self._capacity.release()

    def run(self, candidate: Candidate, inputs: InputsT) -> OutputT:
        return run_awaitable_sync(self.arun(candidate, inputs))

    def _configure(
        self,
        invoke: Invocation[InputsT, OutputT],
        *,
        required_components: Sequence[str],
        isolation: Isolation,
        max_concurrency: int | None,
        normalize: NormalizeCandidate | None,
        identity: str,
    ) -> None:
        normalized_components = tuple(dict.fromkeys(required_components))
        if any(not component for component in normalized_components):
            raise ValueError("Runtime component names cannot be empty.")
        if max_concurrency is not None and max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1.")
        if isolation == "serialized" and max_concurrency not in (None, 1):
            raise ValueError("Serialized runtimes cannot use max_concurrency greater than 1.")

        active_concurrency = 1 if isolation == "serialized" else max_concurrency
        self._invoke = invoke
        self._required_components = normalized_components
        self._isolation: Isolation = isolation
        self._max_concurrency = active_concurrency
        self._normalize = normalize
        self._identity = identity
        self._capacity = (
            BoundedSemaphore(active_concurrency) if active_concurrency is not None else None
        )

    def _prepare_candidate(self, candidate: Candidate) -> Candidate:
        active_candidate = self._normalize(candidate) if self._normalize is not None else candidate
        missing = [
            component
            for component in self._required_components
            if component not in active_candidate.values
        ]
        if missing:
            raise CandidateInjectionError(
                "Candidate is missing required components: " + ", ".join(missing) + "."
            )
        return active_candidate

    async def _acquire(self) -> None:
        if self._capacity is None:
            return
        while not self._capacity.acquire(blocking=False):
            await asyncio.sleep(0.001)


@asynccontextmanager
async def _candidate_scope(
    scope: CandidateScope | None,
    candidate: Candidate,
):
    if scope is None:
        yield
        return

    context = scope(candidate)
    if isinstance(context, AbstractAsyncContextManager):
        async with context:
            yield
        return
    with context:
        yield


async def _resolve(value: ResolvedT | Awaitable[ResolvedT]) -> ResolvedT:
    if inspect.isawaitable(value):
        return await cast("Awaitable[ResolvedT]", value)
    return value


__all__ = (
    "CandidateScope",
    "Factory",
    "FactoryTask",
    "Isolation",
    "NormalizeCandidate",
    "Runtime",
    "RuntimeTask",
    "SubjectCleanup",
)
