from __future__ import annotations as _annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Generic, Protocol, TypeVar, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .asi import (
    PydanticEvalsASIBuilder,
    PydanticEvalTrajectory,
    ReportCaseRecord,
    SerializableValue,
)
from .candidates import CandidateComponent
from .compat import EvaluationBatch
from .components import ComponentCatalog
from .errors import CandidateComponentError
from .harness import EvalDataset, PydanticEvalsHarness
from .injections import AgentInstructionsInjection, CandidateInjection, ModelOutputInjection
from .objectives import ScoreMappingCarrier, ScoreObjective
from .recorder import CandidateEvaluationRecorder

DataInstT = TypeVar("DataInstT")
RolloutOutputT = TypeVar("RolloutOutputT")
EvaluatorT = TypeVar("EvaluatorT")
ProposalFn = Callable[
    [dict[str, str], Mapping[str, Sequence[Mapping[str, SerializableValue]]], list[str]],
    dict[str, str],
]


@runtime_checkable
class ReportCases(Protocol):
    @property
    def cases(self) -> Sequence[ReportCaseRecord]: ...


@runtime_checkable
class ReportFailures(Protocol):
    @property
    def failures(self) -> Sequence[ReportCaseRecord]: ...


@runtime_checkable
class ReportOutput(Protocol):
    @property
    def output(self) -> SerializableValue: ...


ReportEnvelope = ReportCases | ReportFailures


class PydanticGEPAAdapter(
    BaseModel,
    Generic[DataInstT, RolloutOutputT, EvaluatorT],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    harness: PydanticEvalsHarness[DataInstT, RolloutOutputT, ReportEnvelope, EvaluatorT]
    injections: list[CandidateInjection] = Field(default_factory=list)
    components: ComponentCatalog | None = None
    objective: ScoreObjective
    asi_builder: PydanticEvalsASIBuilder = Field(default_factory=PydanticEvalsASIBuilder)
    propose_new_texts: ProposalFn | None = None
    recorder: (
        CandidateEvaluationRecorder[DataInstT, ReportEnvelope, PydanticEvalTrajectory] | None
    ) = None

    @classmethod
    def from_dataset(
        cls,
        *,
        dataset: EvalDataset[EvaluatorT],
        task: Callable[[DataInstT], RolloutOutputT],
        injections: list[CandidateInjection],
        objective: ScoreObjective,
        components: ComponentCatalog | None = None,
        asi_builder: PydanticEvalsASIBuilder | None = None,
        propose_new_texts: ProposalFn | None = None,
        recorder: CandidateEvaluationRecorder[DataInstT, ReportEnvelope, PydanticEvalTrajectory]
        | None = None,
        max_concurrency: int = 5,
    ) -> PydanticGEPAAdapter[DataInstT, RolloutOutputT, EvaluatorT]:
        return cls(
            harness=PydanticEvalsHarness(
                dataset=dataset,
                task=task,
                max_concurrency=max_concurrency,
            ),
            injections=injections,
            components=components,
            objective=objective,
            asi_builder=asi_builder or PydanticEvalsASIBuilder(),
            propose_new_texts=propose_new_texts,
            recorder=recorder,
        )

    def evaluate(
        self,
        batch: list[DataInstT],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT | None]:
        active_candidate = self.normalize_candidate(candidate)
        with self.injection_scope(active_candidate):
            report = self.harness.evaluate(batch)

        outputs: list[RolloutOutputT | None] = []
        scores: list[float] = []
        objective_scores: list[dict[str, float]] = []
        trajectories: list[PydanticEvalTrajectory] | None = [] if capture_traces else None

        for report_case in _report_cases(report):
            output = (
                cast("RolloutOutputT", report_case.output)
                if isinstance(report_case, ReportOutput)
                else None
            )
            outputs.append(output)
            if isinstance(report_case, ScoreMappingCarrier):
                scores.append(self.objective.extract(report_case))
                objective_scores.append(self.objective.extract_objective_scores(report_case))
            else:
                scores.append(self.objective.failure_score)
                objective_scores.append({})
            if trajectories is not None:
                trajectories.append(PydanticEvalTrajectory(report_case=report_case))

        for failure in _report_failures(report):
            outputs.append(None)
            scores.append(self.objective.failure_score)
            objective_scores.append({})
            if trajectories is not None:
                trajectories.append(PydanticEvalTrajectory(report_case=failure))

        if self.recorder is not None:
            self.recorder.record_candidate_batch(
                candidate=active_candidate,
                batch=list(batch),
                report=report,
                scores=list(scores),
                trajectories=list(trajectories) if trajectories is not None else None,
            )

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
            num_metric_calls=len(scores),
        )

    @contextmanager
    def injection_scope(self, candidate: Mapping[str, str]) -> Iterator[None]:
        with ExitStack() as stack:
            for injection in self.injections:
                stack.enter_context(injection.apply(candidate))
            yield

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT | None],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, SerializableValue]]]:
        active_candidate = self.normalize_candidate(candidate)
        return self.asi_builder.build(
            candidate=active_candidate,
            eval_batch=eval_batch,
            components_to_update=components_to_update,
        )

    def component_names(self) -> list[str]:
        if self.components is not None:
            return self.components.names()
        return [injection.component for injection in self.injections]

    def normalize_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        components = self._candidate_components()
        normalized: dict[str, str] = {}
        for name, value in candidate.items():
            component = components.get(name)
            if component is None:
                normalized[name] = value
                continue
            try:
                normalized[name] = component.encode(component.decode(value))
            except CandidateComponentError:
                normalized[name] = component.encode(value)
        return normalized

    def _candidate_components(self) -> dict[str, CandidateComponent]:
        components = (
            {component.name: component for component in self.components.components}
            if self.components is not None
            else {}
        )
        for injection in self.injections:
            if isinstance(injection, AgentInstructionsInjection):
                component = injection.candidate_component
                components.setdefault(component.name, component)
            elif isinstance(injection, ModelOutputInjection):
                for component in injection.components.components:
                    components.setdefault(component.name, component)
        return components


def _report_cases(report: ReportCases | ReportFailures) -> list[ReportCaseRecord]:
    cases = report.cases if isinstance(report, ReportCases) else None
    if cases is not None:
        return list(cases)
    return []


def _report_failures(report: ReportCases | ReportFailures) -> list[ReportCaseRecord]:
    failures = report.failures if isinstance(report, ReportFailures) else None
    if failures is not None:
        return list(failures)
    return []


__all__ = (
    "DataInstT",
    "PydanticGEPAAdapter",
    "RolloutOutputT",
)
