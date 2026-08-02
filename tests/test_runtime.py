from __future__ import annotations as _annotations

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass

import pytest

from pydantic_gepa import Candidate, CandidateInjectionError, Runtime
from pydantic_gepa.injections import CandidateContext, DerivedValueInjection


def test_runtime_runs_sync_task_and_normalizes_candidate() -> None:
    seen: list[str] = []

    def run(value: str) -> str:
        return value.upper()

    def normalize(candidate: Candidate) -> Candidate:
        return candidate.model_copy(
            update={"values": {**candidate.values, "prompt": candidate.values["prompt"].strip()}}
        )

    @contextmanager
    def scope(candidate: Candidate) -> Iterator[None]:
        seen.append(candidate.values["prompt"])
        yield

    runtime = Runtime(
        run,
        scope=scope,
        required_components=("prompt", "prompt"),
        isolation="serialized",
        normalize=normalize,
    )

    assert runtime.run(Candidate(values={"prompt": " ready ", "frozen": "kept"}), "ok") == "OK"
    assert seen == ["ready"]
    assert runtime.required_components == ("prompt",)
    assert runtime.isolation == "serialized"
    assert runtime.max_concurrency == 1


def test_runtime_uses_safe_isolation_defaults() -> None:
    @contextmanager
    def scope(_: Candidate) -> Iterator[None]:
        yield

    pure_runtime = Runtime(lambda value: value)
    scoped_runtime = Runtime(lambda value: value, scope=scope)

    assert scoped_runtime.run(Candidate(), "value") == "value"
    assert pure_runtime.isolation == "context_local"
    assert pure_runtime.max_concurrency is None
    assert scoped_runtime.isolation == "serialized"
    assert scoped_runtime.max_concurrency == 1


@pytest.mark.asyncio
async def test_runtime_supports_async_scope_and_task_cleanup() -> None:
    active: list[str] = []

    @asynccontextmanager
    async def scope(candidate: Candidate):
        active.append(candidate.values["prompt"])
        try:
            yield
        finally:
            active.pop()

    async def run(value: str) -> str:
        assert active == ["candidate"]
        await asyncio.sleep(0)
        return value.upper()

    runtime = Runtime(run, scope=scope, required_components=("prompt",))

    assert await runtime.arun(Candidate(values={"prompt": "candidate"}), "ok") == "OK"
    assert active == []


@pytest.mark.asyncio
async def test_runtime_restores_scope_after_failure_and_cancellation() -> None:
    active: list[str] = []
    entered = asyncio.Event()
    release = asyncio.Event()

    @contextmanager
    def scope(candidate: Candidate) -> Iterator[None]:
        active.append(candidate.values["prompt"])
        try:
            yield
        finally:
            active.pop()

    async def fail(_: str) -> str:
        raise RuntimeError("task failed")

    with pytest.raises(RuntimeError, match="task failed"):
        await Runtime(fail, scope=scope).arun(Candidate(values={"prompt": "failure"}), "x")
    assert active == []

    async def wait(_: str) -> str:
        entered.set()
        await release.wait()
        return "done"

    runtime = Runtime(wait, scope=scope, isolation="serialized")
    task = asyncio.create_task(runtime.arun(Candidate(values={"prompt": "cancel"}), "x"))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert active == []
    release.set()
    assert await runtime.arun(Candidate(values={"prompt": "after-cancel"}), "x") == "done"
    assert active == []


@pytest.mark.asyncio
async def test_serialized_runtime_prevents_overlapping_invocations() -> None:
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    active = 0
    maximum_active = 0

    async def run(value: str) -> str:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if value == "first":
            first_entered.set()
            await release_first.wait()
        active -= 1
        return value

    runtime = Runtime(run, isolation="serialized")
    candidate = Candidate()
    first = asyncio.create_task(runtime.arun(candidate, "first"))
    await first_entered.wait()
    second = asyncio.create_task(runtime.arun(candidate, "second"))
    await asyncio.sleep(0.005)
    assert not second.done()
    release_first.set()

    assert await asyncio.gather(first, second) == ["first", "second"]
    assert maximum_active == 1


@pytest.mark.asyncio
async def test_context_local_runtime_respects_configured_concurrency() -> None:
    release = asyncio.Event()
    two_entered = asyncio.Event()
    active = 0
    maximum_active = 0

    async def run(value: int) -> int:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        if active == 2:
            two_entered.set()
        await release.wait()
        active -= 1
        return value

    runtime = Runtime(run, max_concurrency=2)
    candidate = Candidate()
    tasks = [asyncio.create_task(runtime.arun(candidate, value)) for value in range(3)]
    await two_entered.wait()
    assert maximum_active == 2
    release.set()

    assert await asyncio.gather(*tasks) == [0, 1, 2]
    assert runtime.max_concurrency == 2


@pytest.mark.asyncio
async def test_factory_runtime_creates_and_closes_isolated_subjects() -> None:
    created: list[str] = []
    closed: list[str] = []

    async def create(candidate: Candidate) -> _Subject:
        subject = _Subject(prompt=candidate.values["prompt"])
        created.append(subject.prompt)
        return subject

    async def run(subject: _Subject, value: str) -> str:
        return f"{subject.prompt}:{value}"

    async def close(subject: _Subject) -> None:
        closed.append(subject.prompt)

    runtime = Runtime.from_factory(
        create,
        run,
        close=close,
        required_components=("prompt",),
        max_concurrency=2,
    )

    results = await asyncio.gather(
        runtime.arun(Candidate(values={"prompt": "one"}), "a"),
        runtime.arun(Candidate(values={"prompt": "two"}), "b"),
    )

    assert results == ["one:a", "two:b"]
    assert created == ["one", "two"]
    assert closed == ["one", "two"]
    assert runtime.isolation == "factory"


@pytest.mark.asyncio
async def test_factory_runtime_closes_subject_after_task_failure() -> None:
    closed: list[str] = []

    def create(candidate: Candidate) -> _Subject:
        return _Subject(prompt=candidate.values["prompt"])

    def fail(_: _Subject, __: str) -> str:
        raise RuntimeError("factory task failed")

    def close(subject: _Subject) -> None:
        closed.append(subject.prompt)

    runtime = Runtime.from_factory(create, fail, close=close)
    with pytest.raises(RuntimeError, match="factory task failed"):
        await runtime.arun(Candidate(values={"prompt": "candidate"}), "x")
    assert closed == ["candidate"]


def test_runtime_adapts_existing_injections_and_restores_context() -> None:
    context = CandidateContext[str]("prompt")
    injection = DerivedValueInjection(
        component="prompt",
        context=context,
        derive_value=lambda candidate: candidate["prompt"],
        required_components=("prompt",),
    )
    runtime = Runtime.from_injections(lambda _: context.require(), [injection])

    assert runtime.run(Candidate(values={"prompt": "active"}), None) == "active"
    assert context.get() is None


def test_runtime_closes_entered_injections_when_later_injection_fails() -> None:
    entered: list[str] = []
    runtime = Runtime.from_injections(
        lambda _: "unused",
        [_TrackingInjection("first", entered), _FailingInjection("second")],
    )

    with pytest.raises(RuntimeError, match="cannot enter"):
        runtime.run(Candidate(values={"first": "1", "second": "2"}), None)
    assert entered == []


def test_runtime_validates_configuration_and_required_components() -> None:
    with pytest.raises(ValueError, match="Runtime.from_factory"):
        Runtime(lambda value: value, isolation="factory")
    with pytest.raises(ValueError, match="at least 1"):
        Runtime(lambda value: value, max_concurrency=0)
    with pytest.raises(ValueError, match="greater than 1"):
        Runtime(lambda value: value, isolation="serialized", max_concurrency=2)
    with pytest.raises(ValueError, match="cannot be empty"):
        Runtime(lambda value: value, required_components=("",))

    runtime = Runtime(lambda value: value, required_components=("prompt",))
    with pytest.raises(CandidateInjectionError, match="prompt"):
        runtime.run(Candidate(values={"other": "preserved"}), "value")


@pytest.mark.asyncio
async def test_sync_runtime_entrypoint_works_inside_event_loop() -> None:
    async def run(value: str) -> str:
        await asyncio.sleep(0)
        return value.upper()

    runtime = Runtime(run)
    assert runtime.run(Candidate(), "ok") == "OK"


@dataclass(frozen=True, slots=True)
class _Subject:
    prompt: str


@dataclass(slots=True)
class _TrackingInjection:
    component: str
    entered: list[str]

    @contextmanager
    def apply(self, candidate: Mapping[str, str]) -> Iterator[None]:
        self.entered.append(candidate[self.component])
        try:
            yield
        finally:
            self.entered.pop()


@dataclass(slots=True)
class _FailingInjection:
    component: str

    def apply(self, candidate: Mapping[str, str]) -> AbstractContextManager[None]:
        del candidate
        raise RuntimeError("cannot enter")
