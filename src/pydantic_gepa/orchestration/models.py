from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..candidates import Candidate
from ..configuration import BudgetConfig, GEPAConfig
from ..errors import PlanError
from ..examples import PydanticGEPAOptimization
from ..objectives import ScoreObjective
from ..results import CandidateSummary, PydanticGEPAResult

InputsT = TypeVar("InputsT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")

StageStatus = Literal["completed", "failed", "skipped"]
CarryForward = Literal["accepted", "initial"]
StopPolicy = Literal["on_failure", "continue"]
AggregateName = Literal["mean", "weighted_mean", "min"]


class Budget(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_metric_calls: int = Field(gt=0)


class BudgetUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    limit: int = Field(ge=0)
    used: int = Field(ge=0)
    reported: int | None = Field(default=None, ge=0)
    exhausted: bool

    @model_validator(mode="after")
    def validate_usage(self) -> BudgetUsage:
        if self.used > self.limit:
            raise ValueError("Budget usage cannot exceed its limit.")
        if self.reported is not None and self.reported > self.limit:
            raise ValueError("Reported metric calls cannot exceed the stage limit.")
        return self


class StageError(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    message: str


class StageOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate: Candidate
    score: float
    metric_calls: int | None = Field(default=None, ge=0)
    accepted: bool = True
    history: tuple[CandidateSummary, ...] = ()
    checkpoint: str | None = None

    @classmethod
    def from_result(cls, result: PydanticGEPAResult) -> StageOutput:
        return cls(
            candidate=result.best_candidate,
            score=result.best_score,
            metric_calls=result.total_metric_calls,
            history=tuple(result.candidate_history),
            checkpoint=result.run_dir,
        )


StageExecution: TypeAlias = StageOutput | PydanticGEPAResult
StageRun: TypeAlias = Callable[[Candidate, int], StageExecution | Awaitable[StageExecution]]
StageRescore: TypeAlias = Callable[[Candidate], float | Awaitable[float]]


@dataclass(frozen=True, slots=True)
class Stage:
    id: str
    components: tuple[str, ...]
    run: StageRun
    frozen: tuple[str, ...] = ()
    budget: Budget = Budget(max_metric_calls=50)
    objective: ScoreObjective = ScoreObjective(score_key="score")
    seed: Candidate | None = None
    rescore: StageRescore | None = None
    run_id: str | None = None
    rescore_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise PlanError("Stage id cannot be empty.")
        if not self.components:
            raise PlanError(f"Stage '{self.id}' must target at least one component.")
        if len(set(self.components)) != len(self.components):
            raise PlanError(f"Stage '{self.id}' contains duplicate target components.")
        overlap = set(self.components).intersection(self.frozen)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise PlanError(f"Stage '{self.id}' targets frozen components: {names}.")
        if self.rescore_id is not None and self.rescore is None:
            raise PlanError("rescore_id requires a rescore callable.")

    @classmethod
    def from_optimization(
        cls,
        stage_id: str,
        optimization: PydanticGEPAOptimization[InputsT, OutputT, MetadataT],
        *,
        components: Sequence[str],
        frozen: Sequence[str] = (),
        budget: Budget | None = None,
        objective: ScoreObjective | None = None,
        seed: Candidate | None = None,
        rescore: StageRescore | None = None,
        run_id: str | None = None,
        rescore_id: str | None = None,
    ) -> Stage:
        def run(candidate: Candidate, max_metric_calls: int) -> PydanticGEPAResult:
            return optimization.optimize(
                initial_candidate=candidate,
                config=GEPAConfig(budget=BudgetConfig(max_metric_calls=max_metric_calls)),
            )

        return cls(
            id=stage_id,
            components=tuple(components),
            run=run,
            frozen=tuple(frozen),
            budget=budget or Budget(max_metric_calls=50),
            objective=objective or optimization.objective,
            seed=seed,
            rescore=rescore,
            run_id=run_id,
            rescore_id=rescore_id,
        )

    def snapshot(self) -> StageSpec:
        return StageSpec(
            id=self.id,
            components=self.components,
            frozen=self.frozen,
            budget=self.budget,
            objective=self.objective,
            seed=self.seed,
            run_id=self.run_id or self.id,
            rescore_id=self.rescore_id,
        )


class StageResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage_id: str
    status: StageStatus
    input_candidate: Candidate
    output_candidate: Candidate
    target_components: tuple[str, ...]
    frozen_components: tuple[str, ...]
    score: float | None = None
    final_score: float | None = None
    budget: BudgetUsage
    history: tuple[CandidateSummary, ...] = ()
    checkpoint: str | None = None
    error: StageError | None = None
    stop_reason: str | None = None

    @property
    def effective_score(self) -> float | None:
        return self.final_score if self.final_score is not None else self.score


class StageSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    components: tuple[str, ...]
    frozen: tuple[str, ...] = ()
    budget: Budget
    objective: ScoreObjective
    seed: Candidate | None = None
    run_id: str
    rescore_id: str | None = None

    def build(
        self,
        *,
        runs: Mapping[str, StageRun],
        rescores: Mapping[str, StageRescore] | None = None,
    ) -> Stage:
        try:
            run = runs[self.run_id]
        except KeyError as exc:
            raise PlanError(f"No stage runner is registered as '{self.run_id}'.") from exc
        rescore: StageRescore | None = None
        if self.rescore_id is not None:
            if rescores is None or self.rescore_id not in rescores:
                raise PlanError(f"No stage rescore callable is registered as '{self.rescore_id}'.")
            rescore = rescores[self.rescore_id]
        return Stage(
            id=self.id,
            components=self.components,
            run=run,
            frozen=self.frozen,
            budget=self.budget,
            objective=self.objective,
            seed=self.seed,
            rescore=rescore,
            run_id=self.run_id,
            rescore_id=self.rescore_id,
        )


class PlanSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    stages: tuple[StageSpec, ...]
    initial_candidate: Candidate
    budget: Budget | None = None
    carry_forward: CarryForward = "accepted"
    stop: StopPolicy = "on_failure"
    aggregate: AggregateName | None = "mean"
    aggregate_id: str | None = None
    weights: dict[str, float] = Field(default_factory=dict)
    final_rescore_id: str | None = None


Aggregate: TypeAlias = AggregateName | Callable[[Sequence[StageResult]], float]


class PlanResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    initial_candidate: Candidate
    final_candidate: Candidate
    stages: tuple[StageResult, ...]
    score: float | None = None
    final_score: float | None = None
    total_metric_calls: int = Field(ge=0)
    budget: BudgetUsage | None = None
    stop_reason: str | None = None

    @property
    def effective_score(self) -> float | None:
        return self.final_score if self.final_score is not None else self.score


__all__ = (
    "Aggregate",
    "AggregateName",
    "Budget",
    "BudgetUsage",
    "CarryForward",
    "PlanResult",
    "PlanSpec",
    "Stage",
    "StageError",
    "StageExecution",
    "StageOutput",
    "StageRescore",
    "StageResult",
    "StageRun",
    "StageSpec",
    "StageStatus",
    "StopPolicy",
)
