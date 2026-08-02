from __future__ import annotations as _annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from threading import Thread
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from .errors import EvaluationHarnessError

CaseT = TypeVar("CaseT")
OutputT = TypeVar("OutputT")
ReportT = TypeVar("ReportT")
EvaluatorT = TypeVar("EvaluatorT")
CaseT_co = TypeVar("CaseT_co", covariant=True)
OutputT_contra = TypeVar("OutputT_contra", contravariant=True)
ReportT_co = TypeVar("ReportT_co", covariant=True)
EvaluatorT_co = TypeVar("EvaluatorT_co", covariant=True)
EvaluatorT_contra = TypeVar("EvaluatorT_contra", contravariant=True)


@runtime_checkable
class EvalDataset(Protocol[EvaluatorT_co]):
    @property
    def evaluators(self) -> Sequence[EvaluatorT_co]: ...


@runtime_checkable
class NamedEvalDataset(Protocol):
    @property
    def name(self) -> str | None: ...


class DatasetConstructor(Protocol[CaseT, OutputT_contra, ReportT_co, EvaluatorT_contra]):
    def __call__(
        self, *, cases: Sequence[CaseT], evaluators: Sequence[EvaluatorT_contra]
    ) -> EvaluatableDataset[CaseT, OutputT_contra, ReportT_co]: ...


class NamedDatasetConstructor(Protocol[CaseT, OutputT_contra, ReportT_co, EvaluatorT_contra]):
    def __call__(
        self,
        *,
        name: str | None,
        cases: Sequence[CaseT],
        evaluators: Sequence[EvaluatorT_contra],
    ) -> EvaluatableDataset[CaseT, OutputT_contra, ReportT_co]: ...


@runtime_checkable
class EvaluatableDataset(Protocol[CaseT_co, OutputT_contra, ReportT_co]):
    def evaluate(
        self,
        task: Callable[[CaseT_co], OutputT_contra],
        *,
        max_concurrency: int,
        progress: bool,
    ) -> ReportT_co | Awaitable[ReportT_co]: ...


class PydanticEvalsHarness(BaseModel, Generic[CaseT, OutputT, ReportT, EvaluatorT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: EvalDataset[EvaluatorT]
    task: Callable[[CaseT], OutputT]
    max_concurrency: int = 5

    def evaluate(self, batch: Sequence[CaseT]) -> ReportT:
        dataset = self._batch_dataset(batch)
        result = dataset.evaluate(
            self.task,
            max_concurrency=self.max_concurrency,
            progress=False,
        )
        if inspect.isawaitable(result):
            return run_awaitable_sync(cast("Awaitable[ReportT]", result))
        return result

    def _batch_dataset(self, batch: Sequence[CaseT]) -> EvaluatableDataset[CaseT, OutputT, ReportT]:
        dataset_type = cast(
            "DatasetConstructor[CaseT, OutputT, ReportT, EvaluatorT]",
            type(self.dataset),
        )
        named_dataset_type = cast(
            "NamedDatasetConstructor[CaseT, OutputT, ReportT, EvaluatorT]",
            type(self.dataset),
        )
        evaluators = list(self.dataset.evaluators)
        dataset_name = self.dataset.name if isinstance(self.dataset, NamedEvalDataset) else None
        try:
            if dataset_name is not None:
                dataset = named_dataset_type(
                    name=dataset_name,
                    cases=list(batch),
                    evaluators=evaluators,
                )
            else:
                dataset = dataset_type(cases=list(batch), evaluators=evaluators)
        except TypeError as exc:
            if dataset_name is not None:
                try:
                    dataset = dataset_type(cases=list(batch), evaluators=evaluators)
                except TypeError as inner_exc:
                    raise EvaluationHarnessError(
                        "Dataset must be constructible with cases= and evaluators=."
                    ) from inner_exc
            else:
                raise EvaluationHarnessError(
                    "Dataset must be constructible with cases= and evaluators=."
                ) from exc
        if not isinstance(dataset, EvaluatableDataset):
            raise EvaluationHarnessError("Dataset does not expose evaluate().")
        return dataset


def run_awaitable_sync(awaitable: Awaitable[OutputT]) -> OutputT:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_await_result(awaitable))
    return _run_awaitable_in_thread(awaitable)


async def _await_result(awaitable: Awaitable[OutputT]) -> OutputT:
    return await awaitable


def _run_awaitable_in_thread(awaitable: Awaitable[OutputT]) -> OutputT:
    result: list[OutputT] = []
    errors: list[BaseException] = []

    def runner() -> None:
        try:
            result.append(asyncio.run(_await_result(awaitable)))
        except BaseException as exc:  # pragma: no cover - thread boundary
            errors.append(exc)

    thread = Thread(target=runner)
    thread.start()
    thread.join()

    if errors:
        raise errors[0]
    return result[0]


__all__ = (
    "PydanticEvalsHarness",
    "run_awaitable_sync",
)
