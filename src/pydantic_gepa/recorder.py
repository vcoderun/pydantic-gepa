from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

from .candidates import Candidate
from .events import (
    BackendError,
    BackendProgress,
    BudgetUpdated,
    CandidateAccepted,
    CandidateEvaluated,
    CandidateProposed,
    CandidateRejected,
    CheckpointWritten,
    EvaluationCompleted,
    EvaluationSkipped,
    EvaluationStarted,
    Event,
    IterationCompleted,
    IterationStarted,
    Observer,
    ParetoFrontUpdated,
    ReflectionCompleted,
    ReflectionStarted,
    RunCompleted,
    RunFailed,
    RunStarted,
    event_payload,
)
from .values import JsonValue

if TYPE_CHECKING:
    from gepa.core.callbacks import (
        BudgetUpdatedEvent,
        CandidateAcceptedEvent,
        CandidateRejectedEvent,
        CandidateSelectedEvent,
        ErrorEvent,
        EvaluationEndEvent,
        EvaluationSkippedEvent,
        EvaluationStartEvent,
        IterationEndEvent,
        IterationStartEvent,
        MergeAcceptedEvent,
        MergeAttemptedEvent,
        MergeRejectedEvent,
        MinibatchSampledEvent,
        OptimizationEndEvent,
        OptimizationStartEvent,
        ParetoFrontUpdatedEvent,
        ProposalEndEvent,
        ProposalStartEvent,
        ReflectiveDatasetBuiltEvent,
        StateSavedEvent,
        ValsetEvaluatedEvent,
    )

BatchItemT = TypeVar("BatchItemT", contravariant=True)
ReportT = TypeVar("ReportT", contravariant=True)
TrajectoryT = TypeVar("TrajectoryT", contravariant=True)
EventValue = str | int | float | bool | None | list["EventValue"] | dict[str, "EventValue"]
BridgeLifecycle = Literal["full", "backend_only"]


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
        lifecycle: BridgeLifecycle = "full",
        engine: str = "gepa",
        composition: str | None = None,
        pipeline_id: str | None = None,
        step_id: str | None = None,
        branch_id: str | None = None,
        engine_execution_id: str | None = None,
        parent_execution_id: str | None = None,
    ) -> None:
        if on_event is None and recorder is None:
            raise ValueError("GEPAEventBridge requires on_event or recorder.")
        self.run_id = run_id
        self.on_event = on_event
        self.recorder = recorder
        self.lifecycle = lifecycle
        self.engine = engine
        self.composition = composition
        self.pipeline_id = pipeline_id
        self.step_id = step_id
        self.branch_id = branch_id
        self.engine_execution_id = engine_execution_id
        self.parent_execution_id = parent_execution_id

    def on_optimization_start(self, event: OptimizationStartEvent) -> None:
        if self.lifecycle == "full":
            self._emit(
                RunStarted(
                    run_id=self.run_id,
                    seed=Candidate(values=dict(event["seed_candidate"])),
                    metadata={
                        "train_count": event["trainset_size"],
                        "validation_count": event["valset_size"],
                    },
                )
            )

    def on_optimization_end(self, event: OptimizationEndEvent) -> None:
        if self.lifecycle == "full":
            self._emit(
                RunCompleted(
                    run_id=self.run_id,
                    candidate_id=str(event["best_candidate_idx"]),
                    total_metric_calls=event["total_metric_calls"],
                    metadata={"total_iterations": event["total_iterations"]},
                )
            )

    def on_iteration_start(self, event: IterationStartEvent) -> None:
        self._emit(
            IterationStarted(
                run_id=self.run_id,
                iteration=event["iteration"],
            )
        )

    def on_iteration_end(self, event: IterationEndEvent) -> None:
        self._emit(
            IterationCompleted(
                run_id=self.run_id,
                iteration=event["iteration"],
                metadata={"proposal_accepted": event["proposal_accepted"]},
            )
        )

    def on_candidate_selected(self, event: CandidateSelectedEvent) -> None:
        candidate_id = str(event["candidate_idx"])
        self._emit(
            CandidateProposed(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_id=candidate_id,
                candidate=Candidate(values=dict(event["candidate"]), id=candidate_id),
                metadata={"selection_score": event["score"]},
            )
        )

    def on_minibatch_sampled(self, event: MinibatchSampledEvent) -> None:
        self._emit(
            BackendProgress(
                run_id=self.run_id,
                iteration=event["iteration"],
                name="minibatch_sampled",
                metadata={
                    "batch_size": len(event["minibatch_ids"]),
                    "train_count": event["trainset_size"],
                },
            )
        )

    def on_evaluation_start(self, event: EvaluationStartEvent) -> None:
        self._emit(
            EvaluationStarted(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_id=_optional_text(event["candidate_idx"]),
                parent_ids=tuple(str(parent) for parent in event["parent_ids"]),
                evaluation_id=self._evaluation_id(
                    event["iteration"],
                    event["candidate_idx"],
                ),
                case_count=event["batch_size"],
                metadata={
                    "capture_traces": event["capture_traces"],
                    "seed_candidate": event["is_seed_candidate"],
                },
            )
        )

    def on_evaluation_end(self, event: EvaluationEndEvent) -> None:
        scores = tuple(float(score) for score in event["scores"])
        self._emit(
            EvaluationCompleted(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_id=_optional_text(event["candidate_idx"]),
                parent_ids=tuple(str(parent) for parent in event["parent_ids"]),
                evaluation_id=self._evaluation_id(
                    event["iteration"],
                    event["candidate_idx"],
                ),
                case_count=max(1, len(scores)),
                scores=scores,
                metadata={
                    "has_trajectories": event["has_trajectories"],
                    "seed_candidate": event["is_seed_candidate"],
                },
            )
        )

    def on_evaluation_skipped(self, event: EvaluationSkippedEvent) -> None:
        self._emit(
            EvaluationSkipped(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_id=str(event["candidate_idx"]),
                evaluation_id=self._evaluation_id(
                    event["iteration"],
                    event["candidate_idx"],
                ),
                reason=event["reason"],
                metadata={"seed_candidate": event["is_seed_candidate"]},
            )
        )

    def on_valset_evaluated(self, event: ValsetEvaluatedEvent) -> None:
        self._emit(
            CandidateEvaluated(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_id=str(event["candidate_idx"]),
                score=float(event["average_score"]),
                parent_ids=tuple(str(parent) for parent in event["parent_ids"]),
                metadata={
                    "examples_evaluated": event["num_examples_evaluated"],
                    "validation_count": event["total_valset_size"],
                    "best_program": event["is_best_program"],
                },
            )
        )

    def on_reflective_dataset_built(self, event: ReflectiveDatasetBuiltEvent) -> None:
        components: list[JsonValue] = list(event["components"])
        self._emit(
            BackendProgress(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_id=str(event["candidate_idx"]),
                name="reflective_dataset_built",
                metadata={
                    "iteration_id": event["iteration_id"],
                    "components": components,
                    "record_count": sum(len(records) for records in event["dataset"].values()),
                },
            )
        )

    def on_proposal_start(self, event: ProposalStartEvent) -> None:
        components: list[JsonValue] = []
        for component in event["components"]:
            components.append(component)
        self._emit(
            ReflectionStarted(
                run_id=self.run_id,
                iteration=event["iteration"],
                metadata={"components": components},
            )
        )

    def on_proposal_end(self, event: ProposalEndEvent) -> None:
        components: list[JsonValue] = []
        for component in sorted(event["new_instructions"]):
            components.append(component)
        self._emit(
            ReflectionCompleted(
                run_id=self.run_id,
                iteration=event["iteration"],
                metadata={"components": components},
            )
        )
        self._emit(
            CandidateProposed(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate=Candidate(values=dict(event["new_instructions"])),
            )
        )

    def on_candidate_accepted(self, event: CandidateAcceptedEvent) -> None:
        self._emit(
            CandidateAccepted(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_id=str(event["new_candidate_idx"]),
                parent_ids=tuple(str(parent) for parent in event["parent_ids"]),
                score=float(event["new_score"]),
            )
        )

    def on_candidate_rejected(self, event: CandidateRejectedEvent) -> None:
        self._emit(
            CandidateRejected(
                run_id=self.run_id,
                iteration=event["iteration"],
                reason=event["reason"],
                score=float(event["new_score"]),
                metadata={"old_score": event["old_score"]},
            )
        )

    def on_merge_attempted(self, event: MergeAttemptedEvent) -> None:
        components: list[JsonValue] = []
        for component in sorted(event["merged_candidate"]):
            components.append(component)
        self._emit(
            BackendProgress(
                run_id=self.run_id,
                iteration=event["iteration"],
                parent_ids=tuple(str(parent) for parent in event["parent_ids"]),
                name="merge_attempted",
                metadata={"components": components},
            )
        )

    def on_merge_accepted(self, event: MergeAcceptedEvent) -> None:
        self._emit(
            CandidateAccepted(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_id=str(event["new_candidate_idx"]),
                parent_ids=tuple(str(parent) for parent in event["parent_ids"]),
            )
        )

    def on_merge_rejected(self, event: MergeRejectedEvent) -> None:
        self._emit(
            CandidateRejected(
                run_id=self.run_id,
                iteration=event["iteration"],
                parent_ids=tuple(str(parent) for parent in event["parent_ids"]),
                reason=event["reason"],
            )
        )

    def on_pareto_front_updated(self, event: ParetoFrontUpdatedEvent) -> None:
        displaced: list[JsonValue] = [str(candidate) for candidate in event["displaced_candidates"]]
        self._emit(
            ParetoFrontUpdated(
                run_id=self.run_id,
                iteration=event["iteration"],
                candidate_ids=tuple(str(candidate) for candidate in event["new_front"]),
                metadata={"displaced_candidate_ids": displaced},
            )
        )

    def on_state_saved(self, event: StateSavedEvent) -> None:
        self._emit(
            CheckpointWritten(
                run_id=self.run_id,
                iteration=event["iteration"],
                path=event["run_dir"] or "",
            )
        )

    def on_budget_updated(self, event: BudgetUpdatedEvent) -> None:
        self._emit(
            BudgetUpdated(
                run_id=self.run_id,
                iteration=event["iteration"],
                used=event["metric_calls_used"],
                remaining=event["metric_calls_remaining"],
                metadata={"evaluation_calls_delta": event["metric_calls_delta"]},
            )
        )

    def on_error(self, event: ErrorEvent) -> None:
        exception = event["exception"]
        if self.lifecycle == "full" and not event["will_continue"]:
            self._emit(
                RunFailed(
                    run_id=self.run_id,
                    iteration=event["iteration"],
                    error_type=type(exception).__name__,
                    message=str(exception),
                )
            )
            return
        self._emit(
            BackendError(
                run_id=self.run_id,
                iteration=event["iteration"],
                error_type=type(exception).__name__,
                message=str(exception),
                will_continue=event["will_continue"],
            )
        )

    def _evaluation_id(self, iteration: int, candidate_index: int | None) -> str:
        candidate = "seed" if candidate_index is None else str(candidate_index)
        execution = self.engine_execution_id or self.run_id
        return f"{execution}:evaluation:{iteration}:{candidate}"

    def _emit(self, event: Event) -> None:
        correlated = event.model_copy(
            update={
                "engine": self.engine,
                "composition": self.composition,
                "pipeline_id": self.pipeline_id,
                "step_id": self.step_id,
                "branch_id": self.branch_id,
                "engine_execution_id": self.engine_execution_id,
                "parent_execution_id": self.parent_execution_id,
            }
        )
        if self.on_event is not None:
            self.on_event(correlated)
        if self.recorder is not None:
            self.recorder.record_event(
                event_name=correlated.kind,
                payload=dict(event_payload(correlated)),
            )


def _optional_text(value: int | None) -> str | None:
    return None if value is None else str(value)


__all__ = (
    "BridgeLifecycle",
    "CandidateEvaluationRecorder",
    "EventValue",
    "GEPAEventBridge",
    "OptimizationEventRecorder",
)
