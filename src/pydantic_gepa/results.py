from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator

from .candidates import Candidate, MetadataValue
from .values import JsonValue, ReprSerializable

CandidateStatus = Literal["proposed", "accepted", "rejected", "best", "unknown"]
ResultBackend = Literal["gepa", "optimize_anything"]


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
    test: float | None = None
    aggregate: float | None = None


class BudgetSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    metric_calls: int | None = Field(default=None, ge=0)
    metric_call_limit: int | None = Field(default=None, ge=0)
    reflection_cost: float | None = Field(default=None, ge=0)


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


class PydanticGEPAResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    best_candidate: Candidate
    best_score: float
    backend: ResultBackend = "gepa"
    run_id: str | None = None
    plan_id: str | None = None
    final_candidate: Candidate | None = None
    stages: tuple[dict[str, JsonValue], ...] = ()
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
    "ArtifactReference",
    "BudgetSummary",
    "CandidateDelta",
    "CandidateStatus",
    "CandidateSummary",
    "OptimizationResult",
    "PydanticGEPAResult",
    "ResultBackend",
    "ScoreSummary",
)
