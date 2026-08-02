from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from .events import (
    BackendProgress,
    BudgetUpdated,
    CandidateAccepted,
    CandidateEvaluated,
    CandidateProposed,
    CandidateRejected,
    CheckpointWritten,
    Event,
    Observer,
    RunCompleted,
    RunFailed,
    RunStarted,
    event_payload,
)

BatchItemT = TypeVar("BatchItemT", contravariant=True)
ReportT = TypeVar("ReportT", contravariant=True)
TrajectoryT = TypeVar("TrajectoryT", contravariant=True)
EventValue = str | int | float | bool | None | list["EventValue"] | dict[str, "EventValue"]


@runtime_checkable
class CandidateEvaluationRecorder(Protocol[BatchItemT, ReportT, TrajectoryT]):
    def record_candidate_batch(
        self,
        *,
        candidate: Mapping[str, str],
        batch: Sequence[BatchItemT],
        report: ReportT,
        scores: Sequence[float],
        trajectories: Sequence[TrajectoryT] | None,
    ) -> None: ...


@runtime_checkable
class OptimizationEventRecorder(Protocol):
    def record_event(
        self,
        *,
        event_name: str,
        payload: Mapping[str, EventValue],
    ) -> None: ...


class GEPAEventBridge:
    def __init__(
        self,
        *,
        run_id: str = "gepa",
        on_event: Observer | None = None,
        recorder: OptimizationEventRecorder | None = None,
    ) -> None:
        if on_event is None and recorder is None:
            raise ValueError("GEPAEventBridge requires on_event or recorder.")
        self.run_id = run_id
        self.on_event = on_event
        self.recorder = recorder

    def on_optimization_start(self, event: Mapping[str, Any]) -> None:
        self._emit(RunStarted(run_id=self.run_id, metadata=_normalize_mapping(event)))

    def on_optimization_end(self, event: Mapping[str, Any]) -> None:
        self._emit(
            RunCompleted(
                run_id=self.run_id,
                total_metric_calls=_optional_int(event.get("total_metric_calls")),
                metadata=_normalize_mapping(event),
            )
        )

    def on_iteration_start(self, event: Mapping[str, Any]) -> None:
        self._progress("iteration_start", event)

    def on_iteration_end(self, event: Mapping[str, Any]) -> None:
        self._progress("iteration_end", event)

    def on_candidate_selected(self, event: Mapping[str, Any]) -> None:
        self._emit(
            CandidateProposed(
                run_id=self.run_id,
                candidate_id=_optional_text(event.get("candidate_idx")),
                metadata=_normalize_mapping(event),
            )
        )

    def on_minibatch_sampled(self, event: Mapping[str, Any]) -> None:
        self._progress("minibatch_sampled", event)

    def on_evaluation_start(self, event: Mapping[str, Any]) -> None:
        self._progress("evaluation_start", event)

    def on_evaluation_end(self, event: Mapping[str, Any]) -> None:
        scores = event.get("scores")
        self._emit(
            CandidateEvaluated(
                run_id=self.run_id,
                candidate_id=_optional_text(event.get("candidate_idx")),
                scores=_numeric_sequence(scores),
                metadata=_normalize_mapping(event),
            )
        )

    def on_evaluation_skipped(self, event: Mapping[str, Any]) -> None:
        self._progress("evaluation_skipped", event)

    def on_valset_evaluated(self, event: Mapping[str, Any]) -> None:
        self._emit(
            CandidateEvaluated(
                run_id=self.run_id,
                candidate_id=_optional_text(event.get("candidate_idx")),
                score=_optional_float(event.get("average_score")),
                metadata=_normalize_mapping(event),
            )
        )

    def on_reflective_dataset_built(self, event: Mapping[str, Any]) -> None:
        self._progress("reflective_dataset_built", event)

    def on_proposal_start(self, event: Mapping[str, Any]) -> None:
        self._progress("proposal_start", event)

    def on_proposal_end(self, event: Mapping[str, Any]) -> None:
        self._emit(CandidateProposed(run_id=self.run_id, metadata=_normalize_mapping(event)))

    def on_candidate_accepted(self, event: Mapping[str, Any]) -> None:
        self._emit(
            CandidateAccepted(
                run_id=self.run_id,
                candidate_id=_optional_text(event.get("new_candidate_idx")),
                score=_optional_float(event.get("new_score")),
                metadata=_normalize_mapping(event),
            )
        )

    def on_candidate_rejected(self, event: Mapping[str, Any]) -> None:
        self._emit(
            CandidateRejected(
                run_id=self.run_id,
                reason=str(event.get("reason", "rejected")),
                score=_optional_float(event.get("new_score")),
                metadata=_normalize_mapping(event),
            )
        )

    def on_merge_attempted(self, event: Mapping[str, Any]) -> None:
        self._progress("merge_attempted", event)

    def on_merge_accepted(self, event: Mapping[str, Any]) -> None:
        self._emit(
            CandidateAccepted(
                run_id=self.run_id,
                candidate_id=_optional_text(event.get("new_candidate_idx")),
                metadata=_normalize_mapping(event),
            )
        )

    def on_merge_rejected(self, event: Mapping[str, Any]) -> None:
        self._emit(
            CandidateRejected(
                run_id=self.run_id,
                reason=str(event.get("reason", "merge rejected")),
                metadata=_normalize_mapping(event),
            )
        )

    def on_pareto_front_updated(self, event: Mapping[str, Any]) -> None:
        self._progress("pareto_front_updated", event)

    def on_state_saved(self, event: Mapping[str, Any]) -> None:
        self._emit(
            CheckpointWritten(
                run_id=self.run_id,
                path=str(event.get("run_dir") or ""),
                metadata=_normalize_mapping(event),
            )
        )

    def on_budget_updated(self, event: Mapping[str, Any]) -> None:
        self._emit(
            BudgetUpdated(
                run_id=self.run_id,
                used=_optional_int(event.get("metric_calls_used")) or 0,
                remaining=_optional_int(event.get("metric_calls_remaining")),
                metadata=_normalize_mapping(event),
            )
        )

    def on_error(self, event: Mapping[str, Any]) -> None:
        exception = event.get("exception")
        self._emit(
            RunFailed(
                run_id=self.run_id,
                error_type=type(exception).__name__,
                message=str(exception),
                metadata=_normalize_mapping(event),
            )
        )

    def _progress(self, name: str, event: Mapping[str, Any]) -> None:
        self._emit(
            BackendProgress(run_id=self.run_id, name=name, metadata=_normalize_mapping(event))
        )

    def _emit(self, event: Event) -> None:
        if self.on_event is not None:
            self.on_event(event)
        if self.recorder is not None:
            self.recorder.record_event(
                event_name=event.kind,
                payload=dict(event_payload(event)),
            )


def _normalize_event_value(value: Any) -> EventValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseException):
        return repr(value)
    if isinstance(value, BaseModel):
        return _normalize_event_value(value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return _normalize_event_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize_event_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_normalize_event_value(item) for item in value]
    return repr(value)


def _normalize_mapping(values: Mapping[str, Any]) -> dict[str, EventValue]:
    return {str(key): _normalize_event_value(value) for key, value in values.items()}


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _numeric_sequence(value: Any) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return ()
    return tuple(
        float(item)
        for item in value
        if isinstance(item, int | float) and not isinstance(item, bool)
    )


__all__ = (
    "CandidateEvaluationRecorder",
    "EventValue",
    "GEPAEventBridge",
    "OptimizationEventRecorder",
)
