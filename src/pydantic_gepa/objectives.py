from __future__ import annotations as _annotations

from collections.abc import Mapping
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .values import JsonValue

ObjectiveDirection = Literal["maximize", "minimize"]
MetricRole = Literal["objective", "constraint", "diagnostic"]
ScorePrimitive = bool | int | float
MetricSideInfoValue = JsonValue


class MetricResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    score: float
    role: MetricRole = "diagnostic"
    feedback: str | None = None
    side_info: dict[str, MetricSideInfoValue] = Field(default_factory=dict)


ScoreInput = ScorePrimitive | str | MetricResult | None


@runtime_checkable
class ScoreMappingCarrier(Protocol):
    @property
    def scores(self) -> Mapping[str, ScoreInput | ScoreValueCarrier] | None: ...


@runtime_checkable
class ScoreValueCarrier(Protocol):
    @property
    def value(self) -> ScoreInput: ...


class ScoreObjective(BaseModel):
    model_config = ConfigDict(frozen=True)

    score_key: str
    direction: ObjectiveDirection = "maximize"
    failure_score: float = 0.0

    def extract(self, report_case: ScoreMappingCarrier) -> float:
        score = self._score_value(report_case)
        if score is None:
            return self.failure_score
        if self.direction == "minimize":
            return -score
        return score

    def extract_objective_scores(self, report_case: ScoreMappingCarrier) -> dict[str, float]:
        scores = _scores_mapping(report_case)
        extracted: dict[str, float] = {}
        for key, value in scores.items():
            numeric = _numeric_score_value(value)
            if numeric is not None:
                extracted[key] = numeric
        return extracted

    def _score_value(self, report_case: ScoreMappingCarrier) -> float | None:
        scores = _scores_mapping(report_case)
        if self.score_key not in scores:
            return None
        return _numeric_score_value(scores[self.score_key])


def _scores_mapping(
    report_case: ScoreMappingCarrier,
) -> Mapping[str, ScoreInput | ScoreValueCarrier]:
    return report_case.scores or {}


def _numeric_score_value(score_result: ScoreInput | ScoreValueCarrier) -> float | None:
    value = score_result.value if isinstance(score_result, ScoreValueCarrier) else score_result
    if isinstance(value, MetricResult):
        return value.score
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    return None


__all__ = (
    "MetricResult",
    "MetricRole",
    "MetricSideInfoValue",
    "ObjectiveDirection",
    "ScoreObjective",
)
