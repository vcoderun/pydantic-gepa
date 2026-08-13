from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from typing import Generic, Protocol, TypeVar, runtime_checkable

from .adapter import ReportCases, ReportEnvelope, ReportFailures
from .asi import PydanticEvalTrajectory, ReportCaseName, report_case_result
from .candidates import Candidate
from .compat import EvaluationBatch
from .evaluation.evidence import Encoder
from .events import (
    BackendError,
    BudgetSnapshot,
    CandidateEvaluated,
    CaseEvaluated,
    DatasetDeclaration,
    EvaluationCompleted,
    EvaluationSkipped,
    EvaluationSplit,
    EvaluationStarted,
    MetricCompleted,
    MetricDeclaration,
    RunDeclaration,
    _EventDispatcher,
)
from .objectives import ScoreObjective
from .results import BudgetSummary
from .state import content_fingerprint
from .values import JsonValue

CaseT = TypeVar("CaseT")
OutputT = TypeVar("OutputT")


def run_declaration(
    *,
    configuration_fingerprint: str,
    composition_fingerprint: str | None,
    objective: ScoreObjective,
    trainset: Sequence[CaseT],
    valset: Sequence[CaseT],
    testset: Sequence[CaseT] = (),
    evaluation_call_limit: int | None = None,
    optimizer_cost_limit: float | None = None,
    checkpoint_path: str | None = None,
    engine_declaration: JsonValue = None,
) -> RunDeclaration:
    return RunDeclaration(
        configuration_fingerprint=configuration_fingerprint,
        composition_fingerprint=composition_fingerprint,
        objective=MetricDeclaration(
            name=objective.score_key,
            role="objective",
            direction=objective.direction,
        ),
        datasets=DatasetDeclaration(
            train_count=len(trainset),
            validation_count=len(valset),
            test_count=len(testset),
            train_fingerprint=content_fingerprint(trainset),
            validation_fingerprint=content_fingerprint(valset),
            test_fingerprint=content_fingerprint(testset),
        ),
        evaluation_call_limit=evaluation_call_limit,
        optimizer_cost_limit=optimizer_cost_limit,
        checkpoint_path=checkpoint_path,
        engine_declaration=engine_declaration,
    )


def budget_snapshot(summary: BudgetSummary) -> BudgetSnapshot:
    return BudgetSnapshot(
        evaluation_calls=(
            summary.metric_calls if summary.evaluation_calls is None else summary.evaluation_calls
        ),
        evaluation_call_limit=(
            summary.metric_call_limit
            if summary.evaluation_call_limit is None
            else summary.evaluation_call_limit
        ),
        optimizer_cost=(
            summary.reflection_cost if summary.optimizer_cost is None else summary.optimizer_cost
        ),
        optimizer_cost_limit=summary.optimizer_cost_limit,
        evaluation_cost=summary.evaluation_cost,
        total_cost=summary.total_cost,
    )


@runtime_checkable
class NamedCase(Protocol):
    @property
    def name(self) -> str | None: ...


class EvaluationEventSink(Generic[CaseT, OutputT]):
    def __init__(
        self,
        dispatcher: _EventDispatcher,
        *,
        objective: ScoreObjective,
        trainset: Sequence[CaseT],
        valset: Sequence[CaseT],
        testset: Sequence[CaseT] = (),
    ) -> None:
        self.dispatcher = dispatcher
        self.objective = objective
        self.encoder = Encoder()
        self._splits: dict[str, set[EvaluationSplit]] = {}
        declared_splits: tuple[tuple[EvaluationSplit, Sequence[CaseT]], ...] = (
            ("train", trainset),
            ("validation", valset),
            ("test", testset),
        )
        for split, cases in declared_splits:
            for case in cases:
                self._splits.setdefault(content_fingerprint(case), set()).add(split)

    def started(
        self,
        *,
        candidate: Mapping[str, str],
        batch: Sequence[CaseT],
    ) -> str:
        evaluation_id = self.dispatcher.next_id("evaluation")
        active_candidate = Candidate(values=dict(candidate))
        self.dispatcher.emit(
            EvaluationStarted(
                run_id=self.dispatcher.run_id,
                candidate_id=active_candidate.fingerprint(),
                candidate=active_candidate,
                evaluation_id=evaluation_id,
                split=self._split(batch),
                case_count=len(batch),
            )
        )
        return evaluation_id

    def completed(
        self,
        *,
        evaluation_id: str,
        candidate: Mapping[str, str],
        batch: Sequence[CaseT],
        report: ReportEnvelope,
        evaluation: EvaluationBatch[PydanticEvalTrajectory, OutputT | None],
    ) -> None:
        split = self._split(batch)
        records = list(report.cases) if isinstance(report, ReportCases) else []
        if isinstance(report, ReportFailures):
            records.extend(report.failures)
        trajectories = evaluation.trajectories or []
        objective_scores = evaluation.objective_scores or [{} for _ in evaluation.scores]
        candidate_id = Candidate(values=dict(candidate)).fingerprint()
        for index, (record, transformed_score, raw_objectives) in enumerate(
            zip(records, evaluation.scores, objective_scores, strict=True)
        ):
            traces = trajectories[index].traces if index < len(trajectories) else ()
            case_result = report_case_result(
                record,
                transformed_score=transformed_score,
                objective_key=self.objective.score_key,
                objective_scores=raw_objectives,
                traces=traces,
                encoder=self.encoder,
            )
            case_id = (
                record.name
                if isinstance(record, ReportCaseName) and record.name
                else f"{evaluation_id}:case:{index}"
            )
            self.dispatcher.emit(
                CaseEvaluated(
                    run_id=self.dispatcher.run_id,
                    candidate_id=candidate_id,
                    case_id=case_id,
                    evaluation_id=evaluation_id,
                    split=split,
                    result=case_result,
                    transformed_score=transformed_score,
                )
            )
            for name, metric in case_result.metrics.items():
                self.dispatcher.emit(
                    MetricCompleted(
                        run_id=self.dispatcher.run_id,
                        candidate_id=candidate_id,
                        case_id=case_id,
                        evaluation_id=evaluation_id,
                        metric=name,
                        value=metric.score,
                        role=metric.role,
                        semantic_type=metric.semantic_type,
                        unit=metric.unit,
                        direction=metric.direction,
                        transformed_value=(
                            transformed_score if name == self.objective.score_key else None
                        ),
                    )
                )
        scores = tuple(float(score) for score in evaluation.scores)
        self.dispatcher.emit(
            EvaluationCompleted(
                run_id=self.dispatcher.run_id,
                candidate_id=candidate_id,
                evaluation_id=evaluation_id,
                split=split,
                case_count=len(batch),
                scores=scores,
            )
        )
        self.dispatcher.emit(
            CandidateEvaluated(
                run_id=self.dispatcher.run_id,
                candidate_id=candidate_id,
                score=sum(scores) / len(scores) if scores else None,
                scores=scores,
            )
        )

    def failed(
        self,
        *,
        evaluation_id: str,
        candidate: Mapping[str, str],
        batch: Sequence[CaseT],
        error: BaseException,
    ) -> None:
        candidate_id = Candidate(values=dict(candidate)).fingerprint()
        self.dispatcher.emit(
            EvaluationSkipped(
                run_id=self.dispatcher.run_id,
                candidate_id=candidate_id,
                evaluation_id=evaluation_id,
                split=self._split(batch),
                reason=f"{type(error).__name__}: {error}",
            )
        )
        self.dispatcher.emit(
            BackendError(
                run_id=self.dispatcher.run_id,
                candidate_id=candidate_id,
                error_type=type(error).__name__,
                message=str(error),
                will_continue=False,
            )
        )

    def _split(self, batch: Sequence[CaseT]) -> EvaluationSplit:
        splits: set[EvaluationSplit] = set()
        for case in batch:
            declared = self._splits.get(content_fingerprint(case))
            splits.update(declared if declared is not None else {"unknown"})
        return next(iter(splits)) if len(splits) == 1 else "unknown"


__all__ = ("EvaluationEventSink", "budget_snapshot", "run_declaration")
