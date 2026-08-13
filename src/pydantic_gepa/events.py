from __future__ import annotations as _annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock
from time import monotonic_ns
from typing import Annotated, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .candidates import Candidate, CandidateComponent
from .evaluation.models import CaseResult
from .objectives import ObjectiveDirection
from .values import JsonValue

ObserverPolicy = Literal["raise", "warn", "ignore"]
EventBackend = Literal["gepa", "optimize_anything", "plan"]
StageKind = Literal["component", "composition", "engine", "rescore"]
EvaluationSplit = Literal["train", "validation", "test", "unknown"]
SelectionMethod = Literal["best_score", "vote", "adaptive", "pipeline"]


class DatasetDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    train_count: int = Field(ge=0)
    validation_count: int = Field(ge=0)
    test_count: int = Field(ge=0)
    train_fingerprint: str | None = None
    validation_fingerprint: str | None = None
    test_fingerprint: str | None = None


class MetricDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    role: Literal["objective", "constraint", "diagnostic"] = "diagnostic"
    direction: ObjectiveDirection | None = None
    semantic_type: str | None = None
    unit: str | None = None


class RunDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    configuration_fingerprint: str
    composition_fingerprint: str | None = None
    objective: MetricDeclaration
    datasets: DatasetDeclaration
    evaluation_call_limit: int | None = Field(default=None, ge=0)
    optimizer_cost_limit: float | None = Field(default=None, ge=0)
    checkpoint_path: str | None = None
    engine_declaration: JsonValue = None


class BudgetSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evaluation_calls: int | None = Field(default=None, ge=0)
    evaluation_call_limit: int | None = Field(default=None, ge=0)
    optimizer_cost: float | None = Field(default=None, ge=0)
    optimizer_cost_limit: float | None = Field(default=None, ge=0)
    evaluation_cost: float | None = Field(default=None, ge=0)
    total_cost: float | None = Field(default=None, ge=0)


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_version: Literal["1"] = "1"
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    monotonic_ns: int = Field(default_factory=monotonic_ns, ge=0)
    run_id: str
    sequence: int = Field(default=0, ge=0)
    execution_id: str | None = None
    parent_execution_id: str | None = None
    backend: EventBackend | None = None
    engine: str | None = None
    composition: str | None = None
    pipeline_id: str | None = None
    step_id: str | None = None
    branch_id: str | None = None
    engine_execution_id: str | None = None
    stage_id: str | None = None
    stage_kind: StageKind | None = None
    iteration: int | None = Field(default=None, ge=0)
    candidate_id: str | None = None
    parent_ids: tuple[str, ...] = ()
    component: str | None = None
    case_id: str | None = None
    metric: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RunStarted(EventBase):
    kind: Literal["run.started"] = "run.started"
    seed: Candidate | None = None
    declaration: RunDeclaration | None = None


class RunCompleted(EventBase):
    kind: Literal["run.completed"] = "run.completed"
    score: float | None = None
    total_metric_calls: int | None = Field(default=None, ge=0)
    budget: BudgetSnapshot | None = None


class RunFailed(EventBase):
    kind: Literal["run.failed"] = "run.failed"
    error_type: str
    message: str


class RunCancelled(EventBase):
    kind: Literal["run.cancelled"] = "run.cancelled"
    error_type: str
    message: str


class StageStarted(EventBase):
    kind: Literal["stage.started"] = "stage.started"


class StageCompleted(EventBase):
    kind: Literal["stage.completed"] = "stage.completed"
    score: float | None = None
    budget: BudgetSnapshot | None = None


class StageFailed(EventBase):
    kind: Literal["stage.failed"] = "stage.failed"
    error_type: str
    message: str


class IterationStarted(EventBase):
    kind: Literal["iteration.started"] = "iteration.started"


class IterationCompleted(EventBase):
    kind: Literal["iteration.completed"] = "iteration.completed"
    score: float | None = None


class EvaluationStarted(EventBase):
    kind: Literal["evaluation.started"] = "evaluation.started"
    evaluation_id: str
    split: EvaluationSplit = "unknown"
    case_count: int = Field(default=1, ge=1)
    candidate: Candidate | None = None


class EvaluationCompleted(EventBase):
    kind: Literal["evaluation.completed"] = "evaluation.completed"
    evaluation_id: str
    split: EvaluationSplit = "unknown"
    case_count: int = Field(default=1, ge=1)
    scores: tuple[float, ...] = ()


class EvaluationSkipped(EventBase):
    kind: Literal["evaluation.skipped"] = "evaluation.skipped"
    evaluation_id: str
    split: EvaluationSplit = "unknown"
    reason: str


class CaseEvaluated(EventBase):
    kind: Literal["case.evaluated"] = "case.evaluated"
    evaluation_id: str
    split: EvaluationSplit = "unknown"
    result: CaseResult[JsonValue]
    transformed_score: float | None = None


class ReflectionStarted(EventBase):
    kind: Literal["reflection.started"] = "reflection.started"


class ReflectionCompleted(EventBase):
    kind: Literal["reflection.completed"] = "reflection.completed"


class BackendError(EventBase):
    kind: Literal["backend.error"] = "backend.error"
    error_type: str
    message: str
    will_continue: bool


class ParetoFrontUpdated(EventBase):
    kind: Literal["pareto_front.updated"] = "pareto_front.updated"
    candidate_ids: tuple[str, ...] = ()


class SelectionCompleted(EventBase):
    kind: Literal["selection.completed"] = "selection.completed"
    method: SelectionMethod
    selected_execution_id: str
    contender_execution_ids: tuple[str, ...]
    contender_scores: tuple[float, ...]
    score: float
    reason: str | None = None


class ComponentsRegistered(EventBase):
    kind: Literal["components.registered"] = "components.registered"
    components: tuple[CandidateComponent, ...]


class CandidateProposed(EventBase):
    kind: Literal["candidate.proposed"] = "candidate.proposed"
    candidate: Candidate | None = None


class CandidateNormalized(EventBase):
    kind: Literal["candidate.normalized"] = "candidate.normalized"
    candidate: Candidate


class CandidateEvaluated(EventBase):
    kind: Literal["candidate.evaluated"] = "candidate.evaluated"
    score: float | None = None
    scores: tuple[float, ...] = ()


class CandidateAccepted(EventBase):
    kind: Literal["candidate.accepted"] = "candidate.accepted"
    score: float | None = None


class CandidateRejected(EventBase):
    kind: Literal["candidate.rejected"] = "candidate.rejected"
    reason: str
    score: float | None = None


class MetricStarted(EventBase):
    kind: Literal["metric.started"] = "metric.started"


class MetricCompleted(EventBase):
    kind: Literal["metric.completed"] = "metric.completed"
    value: float
    role: Literal["objective", "constraint", "diagnostic"] = "diagnostic"
    semantic_type: str | None = None
    unit: str | None = None
    direction: ObjectiveDirection | None = None
    transformed_value: float | None = None
    evaluation_id: str | None = None


class MetricFailed(EventBase):
    kind: Literal["metric.failed"] = "metric.failed"
    error_type: str
    message: str


class BudgetUpdated(EventBase):
    kind: Literal["budget.updated"] = "budget.updated"
    used: int = Field(ge=0)
    remaining: int | None = Field(default=None, ge=0)
    optimizer_cost: float | None = Field(default=None, ge=0)
    optimizer_cost_remaining: float | None = Field(default=None, ge=0)
    evaluation_cost: float | None = Field(default=None, ge=0)
    total_cost: float | None = Field(default=None, ge=0)


class BudgetExhausted(EventBase):
    kind: Literal["budget.exhausted"] = "budget.exhausted"
    used: int = Field(ge=0)
    resource: Literal["evaluation_calls", "optimizer_cost"] = "evaluation_calls"


class CheckpointWritten(EventBase):
    kind: Literal["checkpoint.written"] = "checkpoint.written"
    path: str


class CheckpointResumed(EventBase):
    kind: Literal["checkpoint.resumed"] = "checkpoint.resumed"
    path: str


class CheckpointRejected(EventBase):
    kind: Literal["checkpoint.rejected"] = "checkpoint.rejected"
    path: str
    reason: str


class CheckpointReset(EventBase):
    kind: Literal["checkpoint.reset"] = "checkpoint.reset"
    path: str


class FinalRescoreStarted(EventBase):
    kind: Literal["final_rescore.started"] = "final_rescore.started"


class FinalRescoreCompleted(EventBase):
    kind: Literal["final_rescore.completed"] = "final_rescore.completed"
    score: float


class BackendProgress(EventBase):
    kind: Literal["backend.progress"] = "backend.progress"
    name: str


Event: TypeAlias = Annotated[
    RunStarted
    | RunCompleted
    | RunFailed
    | RunCancelled
    | StageStarted
    | StageCompleted
    | StageFailed
    | IterationStarted
    | IterationCompleted
    | EvaluationStarted
    | EvaluationCompleted
    | EvaluationSkipped
    | CaseEvaluated
    | ReflectionStarted
    | ReflectionCompleted
    | BackendError
    | ParetoFrontUpdated
    | SelectionCompleted
    | ComponentsRegistered
    | CandidateProposed
    | CandidateNormalized
    | CandidateEvaluated
    | CandidateAccepted
    | CandidateRejected
    | MetricStarted
    | MetricCompleted
    | MetricFailed
    | BudgetUpdated
    | BudgetExhausted
    | CheckpointWritten
    | CheckpointResumed
    | CheckpointRejected
    | CheckpointReset
    | FinalRescoreStarted
    | FinalRescoreCompleted
    | BackendProgress,
    Field(discriminator="kind"),
]
Observer: TypeAlias = Callable[[Event], None]


@dataclass(frozen=True, slots=True)
class _Subscriber:
    identifier: str
    observer: Observer
    on_error: ObserverPolicy


@dataclass(frozen=True, slots=True)
class _ExecutionScope:
    execution_id: str
    stage_id: str | None = None
    parent_execution_id: str | None = None
    engine: str | None = None
    composition: str | None = None
    pipeline_id: str | None = None
    step_id: str | None = None
    branch_id: str | None = None
    engine_execution_id: str | None = None
    stage_kind: StageKind | None = None


_SUBSCRIBERS: ContextVar[tuple[_Subscriber, ...]] = ContextVar(
    "pydantic_gepa_event_subscribers",
    default=(),
)
_SCOPE: ContextVar[_ExecutionScope | None] = ContextVar(
    "pydantic_gepa_event_scope",
    default=None,
)


class ObserverHandle:
    def __init__(self, identifier: str) -> None:
        self._identifier = identifier
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        _SUBSCRIBERS.set(
            tuple(
                subscriber
                for subscriber in _SUBSCRIBERS.get()
                if subscriber.identifier != self._identifier
            )
        )
        self._closed = True

    def __enter__(self) -> ObserverHandle:
        return self

    def __exit__(self, *_error: type[BaseException] | BaseException | None) -> None:
        self.close()


def subscribe(
    observer: Observer,
    *,
    on_error: ObserverPolicy = "ignore",
) -> ObserverHandle:
    identifier = uuid4().hex
    _SUBSCRIBERS.set((*_SUBSCRIBERS.get(), _Subscriber(identifier, observer, on_error)))
    return ObserverHandle(identifier)


class _EventDispatcher:
    def __init__(
        self,
        *,
        run_id: str,
        execution_id: str,
        backend: EventBackend,
        local_observers: Sequence[Observer] = (),
        local_error_policy: ObserverPolicy = "raise",
        parent_execution_id: str | None = None,
        stage_id: str | None = None,
        subscribers: Sequence[_Subscriber] | None = None,
    ) -> None:
        self.run_id = run_id
        self.execution_id = execution_id
        self.backend = backend
        self.parent_execution_id = parent_execution_id
        self.stage_id = stage_id
        self._local_observers: tuple[Observer, ...] = tuple(local_observers)
        self._local_error_policy: ObserverPolicy = local_error_policy
        self._subscribers: tuple[_Subscriber, ...] = tuple(
            _SUBSCRIBERS.get() if subscribers is None else subscribers
        )
        self._sequence = 0
        self._identifier_sequence = 0
        self._lock = RLock()

    def next_id(self, category: str) -> str:
        with self._lock:
            identifier = f"{self.execution_id}:{category}:{self._identifier_sequence}"
            self._identifier_sequence += 1
            return identifier

    def emit(self, event: Event) -> None:
        with self._lock:
            updates: dict[str, JsonValue | datetime] = {
                "sequence": self._sequence,
                "occurred_at": datetime.now(UTC),
                "monotonic_ns": monotonic_ns(),
            }
            self._sequence += 1
            if event.execution_id is None:
                updates["execution_id"] = self.execution_id
            if event.backend is None:
                updates["backend"] = self.backend
            scope = _SCOPE.get()
            parent_execution_id = (
                self.parent_execution_id
                if scope is None or scope.parent_execution_id is None
                else scope.parent_execution_id
            )
            if event.parent_execution_id is None and parent_execution_id is not None:
                updates["parent_execution_id"] = parent_execution_id
            stage_id = self.stage_id if scope is None or scope.stage_id is None else scope.stage_id
            if event.stage_id is None and stage_id is not None:
                updates["stage_id"] = stage_id
            if scope is not None:
                if event.engine is None and scope.engine is not None:
                    updates["engine"] = scope.engine
                if event.composition is None and scope.composition is not None:
                    updates["composition"] = scope.composition
                if event.pipeline_id is None and scope.pipeline_id is not None:
                    updates["pipeline_id"] = scope.pipeline_id
                if event.step_id is None and scope.step_id is not None:
                    updates["step_id"] = scope.step_id
                if event.branch_id is None and scope.branch_id is not None:
                    updates["branch_id"] = scope.branch_id
                if event.engine_execution_id is None and scope.engine_execution_id is not None:
                    updates["engine_execution_id"] = scope.engine_execution_id
                if event.stage_kind is None and scope.stage_kind is not None:
                    updates["stage_kind"] = scope.stage_kind
            dispatched = event.model_copy(update=updates)
            failures: list[Exception] = []
            for observer in self._local_observers:
                failure = _deliver(observer, dispatched, self._local_error_policy)
                if failure is not None:
                    failures.append(failure)
            for subscriber in self._subscribers:
                failure = _deliver(subscriber.observer, dispatched, subscriber.on_error)
                if failure is not None:
                    failures.append(failure)
            if failures:
                raise failures[0]


def _dispatcher(
    *,
    run_id: str,
    backend: EventBackend,
    local_observers: Sequence[Observer] = (),
    local_error_policy: ObserverPolicy = "raise",
    execution_id: str | None = None,
) -> _EventDispatcher:
    scope = _SCOPE.get()
    return _EventDispatcher(
        run_id=run_id,
        execution_id=execution_id or uuid4().hex,
        backend=backend,
        local_observers=local_observers,
        local_error_policy=local_error_policy,
        parent_execution_id=None if scope is None else scope.execution_id,
        stage_id=None if scope is None else scope.stage_id,
    )


@contextmanager
def _event_scope(
    dispatcher: _EventDispatcher,
    *,
    stage_id: str | None = None,
    parent_execution_id: str | None = None,
    engine: str | None = None,
    composition: str | None = None,
    pipeline_id: str | None = None,
    step_id: str | None = None,
    branch_id: str | None = None,
    engine_execution_id: str | None = None,
    stage_kind: StageKind | None = None,
):
    current = _SCOPE.get()
    token = _SCOPE.set(
        _ExecutionScope(
            execution_id=dispatcher.execution_id,
            stage_id=(
                stage_id or (None if current is None else current.stage_id) or dispatcher.stage_id
            ),
            parent_execution_id=(
                parent_execution_id
                if parent_execution_id is not None
                else None
                if current is None
                else current.parent_execution_id
            ),
            engine=engine if engine is not None else None if current is None else current.engine,
            composition=(
                composition
                if composition is not None
                else None
                if current is None
                else current.composition
            ),
            pipeline_id=(
                pipeline_id
                if pipeline_id is not None
                else None
                if current is None
                else current.pipeline_id
            ),
            step_id=(
                step_id if step_id is not None else None if current is None else current.step_id
            ),
            branch_id=(
                branch_id
                if branch_id is not None
                else None
                if current is None
                else current.branch_id
            ),
            engine_execution_id=(
                engine_execution_id
                if engine_execution_id is not None
                else None
                if current is None
                else current.engine_execution_id
            ),
            stage_kind=(
                stage_kind
                if stage_kind is not None
                else None
                if current is None
                else current.stage_kind
            ),
        )
    )
    try:
        yield
    finally:
        _SCOPE.reset(token)


def compose_observers(
    *observers: Observer,
    on_error: ObserverPolicy = "raise",
) -> Observer:
    def notify(event: Event) -> None:
        failures: list[Exception] = []
        for observer in observers:
            failure = _deliver(observer, event, on_error)
            if failure is not None:
                failures.append(failure)
        if failures:
            raise failures[0]

    return notify


def _deliver(observer: Observer, event: Event, policy: ObserverPolicy) -> Exception | None:
    try:
        observer(event)
    except Exception as exc:
        if policy == "raise":
            return exc
        if policy == "warn":
            warnings.warn(
                f"Event observer {observer!r} failed: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
    return None


def event_payload(event: Event) -> Mapping[str, JsonValue]:
    return event.model_dump(mode="json")


__all__ = (
    "BackendError",
    "BackendProgress",
    "BudgetSnapshot",
    "BudgetExhausted",
    "BudgetUpdated",
    "CandidateAccepted",
    "CandidateEvaluated",
    "CandidateNormalized",
    "CandidateProposed",
    "CandidateRejected",
    "CaseEvaluated",
    "CheckpointRejected",
    "CheckpointReset",
    "CheckpointResumed",
    "CheckpointWritten",
    "ComponentsRegistered",
    "DatasetDeclaration",
    "EvaluationCompleted",
    "EvaluationSkipped",
    "EvaluationSplit",
    "EvaluationStarted",
    "Event",
    "EventBackend",
    "EventBase",
    "FinalRescoreCompleted",
    "FinalRescoreStarted",
    "IterationCompleted",
    "IterationStarted",
    "MetricCompleted",
    "MetricDeclaration",
    "MetricFailed",
    "MetricStarted",
    "Observer",
    "ObserverHandle",
    "ObserverPolicy",
    "ParetoFrontUpdated",
    "ReflectionCompleted",
    "ReflectionStarted",
    "RunCancelled",
    "RunCompleted",
    "RunDeclaration",
    "RunFailed",
    "RunStarted",
    "SelectionCompleted",
    "SelectionMethod",
    "StageCompleted",
    "StageFailed",
    "StageKind",
    "StageStarted",
    "compose_observers",
    "event_payload",
    "subscribe",
)
