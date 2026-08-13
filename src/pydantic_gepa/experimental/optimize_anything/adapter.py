from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict

from ...adapter import PydanticGEPAAdapter
from ...asi import PydanticEvalTrajectory, report_case_record
from ...candidates import Candidate
from ...compat import EvaluationBatch
from ...errors import CandidateComponentError, EvaluationHarnessError
from ...values import JsonValue, SerializableValue
from .models import CandidateMode

CaseT = TypeVar("CaseT")
RolloutOutputT = TypeVar("RolloutOutputT")
EvaluatorT = TypeVar("EvaluatorT")

SideInfo: TypeAlias = dict[str, JsonValue]
CandidateValue: TypeAlias = str | Mapping[str, str]
EvaluationPair: TypeAlias = tuple[CandidateValue, CaseT]
EvaluationOutput: TypeAlias = tuple[float, SideInfo]


class OptimizationState(Protocol):
    @property
    def best_example_evals(self) -> Sequence[Mapping[str, JsonValue]]: ...


@dataclass(slots=True)
class CandidateCodec:
    seed: Candidate
    mode: CandidateMode
    component: str | None = None

    def __post_init__(self) -> None:
        if self.mode == "text":
            active = self.component
            if active is None:
                if len(self.seed.values) != 1:
                    names = ", ".join(sorted(self.seed.values)) or "<none>"
                    raise CandidateComponentError(
                        "Text engines require exactly one component or an explicit component; "
                        f"received: {names}."
                    )
                active = next(iter(self.seed.values))
                self.component = active
            if active not in self.seed.values:
                raise CandidateComponentError(
                    f"Text engine component '{active}' is not present in the seed candidate."
                )

    def encode_seed(self) -> str | dict[str, str]:
        if self.mode == "components":
            return self.seed.to_gepa_dict()
        component = self.component
        if component is None:
            raise RuntimeError("Validated text candidate codec has no component.")
        return self.seed.values[component]

    def decode(self, value: CandidateValue) -> dict[str, str]:
        if isinstance(value, str):
            component = self.component
            if self.mode != "text" or component is None:
                raise CandidateComponentError(
                    "A text candidate was returned for a component-mode engine."
                )
            merged = dict(self.seed.values)
            merged[component] = value
            return merged
        if self.mode == "text":
            raise CandidateComponentError("A text-mode engine returned a component mapping.")
        return dict(value)


class PydanticOptimizeAnythingAdapter(
    BaseModel,
    Generic[CaseT, RolloutOutputT, EvaluatorT],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter: PydanticGEPAAdapter[CaseT, RolloutOutputT, EvaluatorT]

    def evaluate(
        self,
        batch: list[CaseT],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT | None]:
        return self.adapter.evaluate(batch, candidate, capture_traces=capture_traces)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT | None],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, SerializableValue]]]:
        return self.adapter.make_reflective_dataset(
            candidate,
            eval_batch,
            components_to_update,
        )

    def component_names(self) -> list[str]:
        return self.adapter.component_names()

    def evaluator(
        self,
        candidate: Mapping[str, str],
        example: CaseT,
        opt_state: OptimizationState | None = None,
    ) -> EvaluationOutput:
        del opt_state
        active_candidate = self.normalize_candidate(candidate)
        evaluation = self.adapter.evaluate([example], active_candidate, capture_traces=True)
        return self._result(active_candidate, evaluation, index=0, expected=1)

    def batch_evaluator(
        self,
        pairs: Sequence[tuple[Mapping[str, str], CaseT]],
        *,
        opt_states: Sequence[OptimizationState | None] | None = None,
    ) -> list[EvaluationOutput]:
        if opt_states is not None and len(opt_states) != len(pairs):
            raise EvaluationHarnessError(
                "opt_states must align one-to-one with Optimize Anything evaluation pairs."
            )
        grouped: dict[tuple[tuple[str, str], ...], list[tuple[int, CaseT]]] = {}
        candidates: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
        for index, (candidate, example) in enumerate(pairs):
            normalized = self.normalize_candidate(candidate)
            key = tuple(sorted(normalized.items()))
            grouped.setdefault(key, []).append((index, example))
            candidates[key] = normalized

        outputs: dict[int, EvaluationOutput] = {}
        for key, indexed_examples in grouped.items():
            candidate = candidates[key]
            evaluation = self.adapter.evaluate(
                [example for _, example in indexed_examples],
                candidate,
                capture_traces=True,
            )
            expected = len(indexed_examples)
            for evaluation_index, (output_index, _) in enumerate(indexed_examples):
                outputs[output_index] = self._result(
                    candidate,
                    evaluation,
                    index=evaluation_index,
                    expected=expected,
                )
        return [outputs[index] for index in range(len(pairs))]

    def normalize_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        return self.adapter.normalize_candidate(candidate)

    def _result(
        self,
        candidate: dict[str, str],
        evaluation: EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT | None],
        *,
        index: int,
        expected: int,
    ) -> EvaluationOutput:
        if len(evaluation.scores) != expected or len(evaluation.outputs) != expected:
            detail = (
                "exactly one evaluation result"
                if expected == 1
                else f"{expected} evaluation results"
            )
            raise EvaluationHarnessError(f"Optimize Anything evaluation must return {detail}.")
        score = evaluation.scores[index]
        objectives = evaluation.objective_scores
        objective_scores = (
            None if objectives is None or len(objectives) != expected else objectives[index]
        )
        return score, _side_info(
            adapter=self.adapter,
            candidate=candidate,
            evaluation=evaluation,
            index=index,
            score=score,
            objective_scores=objective_scores,
        )


def _side_info(
    *,
    adapter: PydanticGEPAAdapter[CaseT, RolloutOutputT, EvaluatorT],
    candidate: dict[str, str],
    evaluation: EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT | None],
    index: int,
    score: float,
    objective_scores: dict[str, float] | None,
) -> SideInfo:
    side_info: SideInfo = {"scores": {adapter.objective.score_key: score}}
    trajectories = evaluation.trajectories
    active_trajectory = (
        None if trajectories is None or len(trajectories) <= index else trajectories[index]
    )
    if active_trajectory is not None:
        record = report_case_record(
            active_trajectory.report_case,
            score=score,
            include_case_metadata=adapter.asi_builder.include_case_metadata,
            include_expected_output=adapter.asi_builder.include_expected_output,
            include_scores=adapter.asi_builder.include_scores,
            include_assertions=adapter.asi_builder.include_assertions,
            include_errors=adapter.asi_builder.include_errors,
        )
        for key, value in record.items():
            side_info["observed_scores" if key == "scores" else key] = cast(
                "JsonValue",
                value,
            )

    if objective_scores:
        side_info["objective_scores"] = {
            name: float(value) for name, value in objective_scores.items()
        }

    single = EvaluationBatch[
        PydanticEvalTrajectory,
        RolloutOutputT | None,
    ](
        outputs=[evaluation.outputs[index]],
        scores=[score],
        trajectories=(None if active_trajectory is None else [active_trajectory]),
        objective_scores=(None if objective_scores is None else [objective_scores]),
        num_metric_calls=evaluation.num_metric_calls,
    )
    reflective = adapter.make_reflective_dataset(
        candidate,
        single,
        adapter.component_names(),
    )
    for component, records in reflective.items():
        if records:
            side_info[f"{component}_specific_info"] = {
                "examples": [
                    {name: cast("JsonValue", value) for name, value in record.items()}
                    for record in records
                ]
            }
    return side_info


__all__ = (
    "CandidateCodec",
    "CandidateValue",
    "EvaluationOutput",
    "EvaluationPair",
    "OptimizationState",
    "PydanticOptimizeAnythingAdapter",
    "SideInfo",
)
