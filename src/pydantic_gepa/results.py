from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator

from .candidates import Candidate, MetadataValue
from .values import JsonValue, ReprSerializable

CandidateStatus = Literal["proposed", "accepted", "rejected", "best", "unknown"]
ResultBackend = Literal["gepa", "optimize_anything"]
EngineRunStatus = Literal["completed", "failed", "cancelled"]
BudgetSource = Literal["upstream", "pydantic_gepa", "mixed", "unknown"]


class CandidateDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    component: str
    before: str | None = None
    after: str | None = None


class ScoreSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    search: tuple[float, ...] = ()
    train: float | None = None
    validation: float | None = None
    baseline_test: float | None = None
    test: float | None = None
    aggregate: float | None = None


class BudgetSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_calls: int | None = Field(default=None, ge=0)
    metric_call_limit: int | None = Field(default=None, ge=0)
    reflection_cost: float | None = Field(default=None, ge=0)
    evaluation_calls: int | None = Field(default=None, ge=0)
    evaluation_call_limit: int | None = Field(default=None, ge=0)
    optimizer_cost: float | None = Field(default=None, ge=0)
    optimizer_cost_limit: float | None = Field(default=None, ge=0)
    evaluation_cost: float | None = Field(default=None, ge=0)
    total_cost: float | None = Field(default=None, ge=0)
    final_rescore_calls: int = Field(default=0, ge=0)
    final_rescore_cost: float | None = Field(default=None, ge=0)
    heldout_evaluation_calls: int = Field(default=0, ge=0)
    heldout_evaluation_cost: float | None = Field(default=None, ge=0)
    source: BudgetSource = "unknown"


class ArtifactReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    path: str


class CandidateSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    parent_ids: list[str] = Field(default_factory=list)
    generation: int | None = None
    score: float
    values: dict[str, str] = Field(default_factory=dict)
    validation_subscores: dict[str, float] = Field(default_factory=dict)
    objective_scores: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
    status: CandidateStatus = "unknown"
    feedback: tuple[str, ...] = ()
    deltas: tuple[CandidateDelta, ...] = ()


class EngineRunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_id: str
    parent_execution_id: str | None = None
    pipeline_id: str | None = None
    step_id: str | None = None
    branch_id: str | None = None
    engine: str
    family: str
    candidate_mode: Literal["components", "text"]
    input_candidate: Candidate
    output_candidate: Candidate
    search_score: float
    selection_score: float | None = None
    status: EngineRunStatus = "completed"
    stop_reason: str | None = None
    budget: BudgetSummary = Field(default_factory=BudgetSummary)
    duration_seconds: float | None = Field(default=None, ge=0)
    artifacts: tuple[ArtifactReference, ...] = ()
    reported: dict[str, JsonValue] = Field(default_factory=dict)


class SelectionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: Literal["best_score", "vote", "adaptive", "pipeline"]
    selected_execution_id: str
    selected_candidate: Candidate
    score: float
    contender_execution_ids: tuple[str, ...]
    contender_scores: tuple[float, ...]
    evaluation_calls: int = Field(default=0, ge=0)
    reason: str | None = None


class AdaptiveSliceSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine_index: int = Field(ge=0)
    engine: str
    evaluation_start: int = Field(ge=0)
    evaluation_end: int = Field(ge=0)
    evaluation_calls: int = Field(ge=0)
    score_before: float | None = None
    score_after: float | None = None
    improved: bool
    optimizer_cost: float | None = Field(default=None, ge=0)


class CompositionStepSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str
    kind: str
    input_candidate: Candidate
    output_candidate: Candidate | None = None
    engine_execution_ids: tuple[str, ...]
    selected_execution_id: str | None = None
    budget: BudgetSummary = Field(default_factory=BudgetSummary)


class CompositionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str
    pipeline_id: str
    engine_runs: tuple[EngineRunSummary, ...]
    selections: tuple[SelectionSummary, ...] = ()
    steps: tuple[CompositionStepSummary, ...] = ()
    adaptive_schedule: tuple[AdaptiveSliceSummary, ...] = ()
    stop_reason: str | None = None
    budget: BudgetSummary = Field(default_factory=BudgetSummary)


class PydanticGEPAResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_candidate: Candidate
    best_score: float
    backend: ResultBackend = "gepa"
    run_id: str | None = None
    plan_id: str | None = None
    stage_id: str | None = None
    final_candidate: Candidate | None = None
    stages: tuple[dict[str, JsonValue], ...] = ()
    composition: CompositionSummary | None = None
    scores: ScoreSummary = Field(default_factory=ScoreSummary)
    budget: BudgetSummary = Field(default_factory=BudgetSummary)
    stop_reason: str | None = None
    artifacts: tuple[ArtifactReference, ...] = ()
    checkpoints: tuple[str, ...] = ()
    reported: dict[str, JsonValue] = Field(default_factory=dict)
    derived: dict[str, JsonValue] = Field(default_factory=dict)
    best_candidate_index: int | None = None
    validation_scores: list[float] = Field(default_factory=list)
    candidate_history: list[CandidateSummary] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    parent_indices: list[list[int | None]] = Field(default_factory=list)
    objective_scores: list[dict[str, float]] | None = None
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None
    per_objective_best_candidates: dict[str, list[int]] | None = None
    objective_pareto_front: dict[str, float] | None = None
    candidate_tree_dot: str | None = None
    candidate_tree_html: str | None = None
    raw_gepa_result: SkipValidation[ReprSerializable | None] = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def normalize_final_candidate(self) -> PydanticGEPAResult:
        if self.final_candidate is None:
            self.final_candidate = self.best_candidate
        return self

    def stable_dump(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json")

    def normalize_candidates(
        self,
        normalize: Callable[[Mapping[str, str]], dict[str, str]],
    ) -> Self:
        def candidate(value: Candidate) -> Candidate:
            return value.model_copy(update={"values": normalize(value.values)})

        history = [
            item.model_copy(
                update={
                    "values": normalize(item.values),
                    "deltas": tuple(
                        CandidateDelta(
                            component=delta.component,
                            before=(
                                None
                                if delta.before is None
                                else normalize({delta.component: delta.before})[delta.component]
                            ),
                            after=(
                                None
                                if delta.after is None
                                else normalize({delta.component: delta.after})[delta.component]
                            ),
                        )
                        for delta in item.deltas
                    ),
                }
            )
            for item in self.candidate_history
        ]
        best_candidate = candidate(self.best_candidate)
        final_candidate = (
            candidate(self.final_candidate) if self.final_candidate is not None else best_candidate
        )
        derived = dict(self.derived)
        derived["best_candidate_fingerprint"] = best_candidate.fingerprint()
        return self.model_copy(
            update={
                "best_candidate": best_candidate,
                "final_candidate": final_candidate,
                "candidate_history": history,
                "candidates": [candidate(item) for item in self.candidates],
                "derived": derived,
            }
        )


OptimizationResult = PydanticGEPAResult


__all__ = (
    "AdaptiveSliceSummary",
    "ArtifactReference",
    "BudgetSummary",
    "BudgetSource",
    "CandidateDelta",
    "CandidateStatus",
    "CandidateSummary",
    "CompositionSummary",
    "CompositionStepSummary",
    "EngineRunStatus",
    "EngineRunSummary",
    "OptimizationResult",
    "PydanticGEPAResult",
    "ResultBackend",
    "ScoreSummary",
    "SelectionSummary",
)
