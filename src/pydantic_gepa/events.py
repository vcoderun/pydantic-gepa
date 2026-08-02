from __future__ import annotations as _annotations

import warnings
from collections.abc import Callable, Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from .candidates import Candidate
from .values import JsonValue

ObserverPolicy = Literal["raise", "warn", "ignore"]


class EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    sequence: int = Field(default=0, ge=0)
    stage_id: str | None = None
    candidate_id: str | None = None
    parent_ids: tuple[str, ...] = ()
    component: str | None = None
    case_id: str | None = None
    metric: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class RunStarted(EventBase):
    kind: Literal["run.started"] = "run.started"
    seed: Candidate | None = None


class RunCompleted(EventBase):
    kind: Literal["run.completed"] = "run.completed"
    score: float | None = None
    total_metric_calls: int | None = Field(default=None, ge=0)


class RunFailed(EventBase):
    kind: Literal["run.failed"] = "run.failed"
    error_type: str
    message: str


class StageStarted(EventBase):
    kind: Literal["stage.started"] = "stage.started"


class StageCompleted(EventBase):
    kind: Literal["stage.completed"] = "stage.completed"
    score: float | None = None


class StageFailed(EventBase):
    kind: Literal["stage.failed"] = "stage.failed"
    error_type: str
    message: str


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


class MetricFailed(EventBase):
    kind: Literal["metric.failed"] = "metric.failed"
    error_type: str
    message: str


class BudgetUpdated(EventBase):
    kind: Literal["budget.updated"] = "budget.updated"
    used: int = Field(ge=0)
    remaining: int | None = Field(default=None, ge=0)


class BudgetExhausted(EventBase):
    kind: Literal["budget.exhausted"] = "budget.exhausted"
    used: int = Field(ge=0)


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
    | StageStarted
    | StageCompleted
    | StageFailed
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


def compose_observers(
    *observers: Observer,
    on_error: ObserverPolicy = "raise",
) -> Observer:
    def notify(event: Event) -> None:
        for observer in observers:
            try:
                observer(event)
            except Exception as exc:
                if on_error == "raise":
                    raise
                if on_error == "warn":
                    warnings.warn(
                        f"Event observer {observer!r} failed: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    return notify


def event_payload(event: Event) -> Mapping[str, JsonValue]:
    return event.model_dump(mode="json")


__all__ = (
    "BackendProgress",
    "BudgetExhausted",
    "BudgetUpdated",
    "CandidateAccepted",
    "CandidateEvaluated",
    "CandidateNormalized",
    "CandidateProposed",
    "CandidateRejected",
    "CheckpointRejected",
    "CheckpointReset",
    "CheckpointResumed",
    "CheckpointWritten",
    "Event",
    "EventBase",
    "FinalRescoreCompleted",
    "FinalRescoreStarted",
    "MetricCompleted",
    "MetricFailed",
    "MetricStarted",
    "Observer",
    "ObserverPolicy",
    "RunCompleted",
    "RunFailed",
    "RunStarted",
    "StageCompleted",
    "StageFailed",
    "StageStarted",
    "compose_observers",
    "event_payload",
)
