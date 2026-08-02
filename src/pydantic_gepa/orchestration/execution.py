from __future__ import annotations as _annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import cast

from .._version import __version__
from ..candidates import Candidate
from ..configuration import RunConfig
from ..errors import PlanError
from ..events import (
    BudgetExhausted,
    BudgetUpdated,
    CandidateAccepted,
    CandidateEvaluated,
    CandidateNormalized,
    CandidateRejected,
    CheckpointRejected,
    CheckpointReset,
    CheckpointResumed,
    CheckpointWritten,
    Event,
    FinalRescoreCompleted,
    FinalRescoreStarted,
    Observer,
    RunCompleted,
    RunFailed,
    RunStarted,
    StageCompleted,
    StageFailed,
    StageStarted,
    compose_observers,
)
from ..results import PydanticGEPAResult
from ..state import (
    CompatibilityFingerprint,
    FileRunStore,
    RunState,
    RunStore,
    content_fingerprint,
)
from .models import (
    Aggregate,
    Budget,
    BudgetUsage,
    CarryForward,
    PlanResult,
    PlanSpec,
    Stage,
    StageError,
    StageExecution,
    StageOutput,
    StageRescore,
    StageResult,
    StageRun,
    StopPolicy,
)


@dataclass(frozen=True, slots=True, init=False)
class Plan:
    stages: tuple[Stage, ...]
    initial_candidate: Candidate
    budget: Budget | None
    carry_forward: CarryForward
    stop: StopPolicy
    aggregate: Aggregate
    weights: Mapping[str, float]
    final_rescore: StageRescore | None
    aggregate_id: str | None
    final_rescore_id: str | None

    def __init__(
        self,
        *stages: Stage,
        initial_candidate: Candidate,
        budget: Budget | None = None,
        carry_forward: CarryForward = "accepted",
        stop: StopPolicy = "on_failure",
        aggregate: Aggregate = "mean",
        weights: Mapping[str, float] | None = None,
        final_rescore: StageRescore | None = None,
        aggregate_id: str | None = None,
        final_rescore_id: str | None = None,
    ) -> None:
        if not stages:
            raise PlanError("A plan must contain at least one stage.")
        stage_ids = [stage.id for stage in stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise PlanError("Stage ids must be unique within a plan.")
        if callable(aggregate) and aggregate_id is None:
            raise PlanError("Custom aggregation requires aggregate_id.")
        if aggregate == "weighted_mean" and not weights:
            raise PlanError("weighted_mean aggregation requires stage weights.")
        if final_rescore_id is not None and final_rescore is None:
            raise PlanError("final_rescore_id requires a final_rescore callable.")
        object.__setattr__(self, "stages", tuple(stages))
        object.__setattr__(self, "initial_candidate", initial_candidate)
        object.__setattr__(self, "budget", budget)
        object.__setattr__(self, "carry_forward", carry_forward)
        object.__setattr__(self, "stop", stop)
        object.__setattr__(self, "aggregate", aggregate)
        object.__setattr__(self, "weights", dict(weights or {}))
        object.__setattr__(self, "final_rescore", final_rescore)
        object.__setattr__(self, "aggregate_id", aggregate_id)
        object.__setattr__(self, "final_rescore_id", final_rescore_id)

    @classmethod
    def sequential(
        cls,
        stages: Sequence[Stage],
        *,
        initial_candidate: Candidate,
        budget: Budget | None = None,
        aggregate: Aggregate = "mean",
        weights: Mapping[str, float] | None = None,
    ) -> Plan:
        return cls(
            *stages,
            initial_candidate=initial_candidate,
            budget=budget,
            aggregate=aggregate,
            weights=weights,
        )

    async def arun(
        self,
        *,
        seed: Candidate | None = None,
        run: RunConfig | None = None,
        store: RunStore | None = None,
        on_event: Observer | Sequence[Observer] | None = None,
    ) -> PlanResult:
        initial = seed or self.initial_candidate
        current = initial
        stage_results: list[StageResult] = []
        total_calls = 0
        stop_reason: str | None = None
        active_run = run or RunConfig()
        active_store = store
        if active_store is None and active_run.directory is not None:
            active_store = FileRunStore(
                active_run.directory,
                run_id=active_run.id,
                resume=active_run.resume,
                fresh=active_run.fresh,
            )
        observers: tuple[Observer, ...]
        if on_event is None:
            observers = ()
        elif callable(on_event):
            observers = (cast("Observer", on_event),)
        else:
            observers = tuple(on_event)
        notify = compose_observers(*observers)
        sequence = 0

        def emit(event: Event) -> None:
            nonlocal sequence
            notify(event.model_copy(update={"sequence": sequence}))
            sequence += 1

        start_stage = 0
        fingerprint: CompatibilityFingerprint | None = None
        if active_store is not None:
            fingerprint = _plan_fingerprint(self, active_run)
            try:
                state = active_store.prepare(
                    fingerprint=fingerprint,
                    initial_candidate=initial,
                )
            except Exception as exc:
                emit(
                    CheckpointRejected(
                        run_id=active_run.id,
                        path=str(active_run.directory or active_store.backend_directory.parent),
                        reason=str(exc),
                    )
                )
                raise
            checkpoint_path = str(active_store.backend_directory.parent)
            if state.reset:
                emit(CheckpointReset(run_id=active_run.id, path=checkpoint_path))
            if state.resumed:
                emit(CheckpointResumed(run_id=active_run.id, path=checkpoint_path))
            current = state.accepted_candidate
            stage_results.extend(StageResult.model_validate(item) for item in state.stages)
            total_calls = state.metric_calls
            start_stage = state.next_stage
            completed = active_store.load_result(PlanResult)
            if state.status == "completed" and completed is not None:
                emit(
                    RunCompleted(
                        run_id=active_run.id,
                        score=completed.effective_score,
                        total_metric_calls=completed.total_metric_calls,
                    )
                )
                return completed

        emit(RunStarted(run_id=active_run.id, seed=current))

        for stage_index, stage in enumerate(self.stages[start_stage:], start=start_stage):
            remaining = (
                self.budget.max_metric_calls - total_calls if self.budget is not None else None
            )
            if remaining is not None and remaining <= 0:
                stop_reason = "shared_budget_exhausted"
                emit(
                    BudgetExhausted(
                        run_id=active_run.id,
                        used=total_calls,
                    )
                )
                break
            stage_limit = stage.budget.max_metric_calls
            if remaining is not None:
                stage_limit = min(stage_limit, remaining)
            stage_input = _seed_stage(current, stage)
            frozen = tuple(
                sorted(set(stage.frozen).union(set(stage_input.values) - set(stage.components)))
            )
            emit(
                StageStarted(
                    run_id=active_run.id,
                    stage_id=stage.id,
                    candidate_id=stage_input.id,
                )
            )
            try:
                execution = stage.run(stage_input, stage_limit)
                if inspect.isawaitable(execution):
                    execution = cast("StageExecution", await execution)
                output = _stage_output(execution)
                _validate_stage_output(stage, stage_input, output.candidate)
                emit(
                    CandidateNormalized(
                        run_id=active_run.id,
                        stage_id=stage.id,
                        candidate_id=output.candidate.id,
                        candidate=output.candidate,
                    )
                )
                reported = output.metric_calls
                charged = stage_limit if reported is None else reported
                merged = _merge_candidate(stage_input, output.candidate, stage.components)
                final_score = await _rescore(stage.rescore, merged)
                stage_result = StageResult(
                    stage_id=stage.id,
                    status="completed",
                    input_candidate=stage_input,
                    output_candidate=merged,
                    target_components=stage.components,
                    frozen_components=frozen,
                    score=output.score,
                    final_score=final_score,
                    budget=BudgetUsage(
                        limit=stage_limit,
                        used=charged,
                        reported=reported,
                        exhausted=charged == stage_limit,
                    ),
                    history=output.history,
                    checkpoint=output.checkpoint,
                )
                if self.carry_forward == "accepted" and output.accepted:
                    current = merged
                elif self.carry_forward == "initial":
                    current = initial
                total_calls += charged
                emit(
                    CandidateEvaluated(
                        run_id=active_run.id,
                        stage_id=stage.id,
                        candidate_id=merged.id,
                        score=stage_result.effective_score,
                    )
                )
                if output.accepted:
                    emit(
                        CandidateAccepted(
                            run_id=active_run.id,
                            stage_id=stage.id,
                            candidate_id=merged.id,
                            score=stage_result.effective_score,
                        )
                    )
                else:
                    emit(
                        CandidateRejected(
                            run_id=active_run.id,
                            stage_id=stage.id,
                            candidate_id=merged.id,
                            reason="stage output was not accepted",
                            score=stage_result.effective_score,
                        )
                    )
                emit(
                    BudgetUpdated(
                        run_id=active_run.id,
                        stage_id=stage.id,
                        used=total_calls,
                        remaining=(
                            None
                            if self.budget is None
                            else self.budget.max_metric_calls - total_calls
                        ),
                    )
                )
                emit(
                    StageCompleted(
                        run_id=active_run.id,
                        stage_id=stage.id,
                        candidate_id=merged.id,
                        score=stage_result.effective_score,
                    )
                )
            except Exception as exc:
                total_calls += stage_limit
                stage_result = StageResult(
                    stage_id=stage.id,
                    status="failed",
                    input_candidate=stage_input,
                    output_candidate=stage_input,
                    target_components=stage.components,
                    frozen_components=frozen,
                    budget=BudgetUsage(
                        limit=stage_limit,
                        used=stage_limit,
                        exhausted=True,
                    ),
                    error=StageError(type=type(exc).__name__, message=str(exc)),
                    stop_reason=f"stage_failed:{stage.id}",
                )
                if self.stop == "on_failure":
                    stop_reason = stage_result.stop_reason
                emit(
                    StageFailed(
                        run_id=active_run.id,
                        stage_id=stage.id,
                        candidate_id=stage_input.id,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
            except BaseException as exc:
                emit(
                    RunFailed(
                        run_id=active_run.id,
                        stage_id=stage.id,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                )
                raise
            stage_results.append(stage_result)
            if active_store is not None and fingerprint is not None:
                active_store.write_candidate(stage_result.output_candidate)
                active_store.write_stage(stage_result)
                active_store.checkpoint(
                    RunState(
                        run_id=active_run.id,
                        fingerprint=fingerprint,
                        accepted_candidate=current,
                        stages=tuple(result.model_dump(mode="json") for result in stage_results),
                        next_stage=stage_index + 1,
                        metric_calls=total_calls,
                        backend_checkpoint=stage_result.checkpoint,
                    )
                )
                emit(
                    CheckpointWritten(
                        run_id=active_run.id,
                        stage_id=stage.id,
                        path=str(active_store.backend_directory.parent),
                    )
                )
            if stop_reason is not None:
                break

        score = _aggregate(self.aggregate, stage_results, self.weights)
        if self.final_rescore is not None:
            emit(
                FinalRescoreStarted(
                    run_id=active_run.id,
                    candidate_id=current.id,
                )
            )
        final_score = await _rescore(self.final_rescore, current)
        if final_score is not None:
            emit(
                FinalRescoreCompleted(
                    run_id=active_run.id,
                    candidate_id=current.id,
                    score=final_score,
                )
            )
        plan_budget = None
        if self.budget is not None:
            plan_budget = BudgetUsage(
                limit=self.budget.max_metric_calls,
                used=total_calls,
                reported=total_calls,
                exhausted=total_calls == self.budget.max_metric_calls,
            )
        result = PlanResult(
            initial_candidate=initial,
            final_candidate=current,
            stages=tuple(stage_results),
            score=score,
            final_score=final_score,
            total_metric_calls=total_calls,
            budget=plan_budget,
            stop_reason=stop_reason,
        )
        if active_store is not None and fingerprint is not None:
            active_store.write_result(result)
            active_store.checkpoint(
                RunState(
                    run_id=active_run.id,
                    fingerprint=fingerprint,
                    status="completed",
                    accepted_candidate=current,
                    stages=tuple(result.model_dump(mode="json") for result in stage_results),
                    next_stage=len(self.stages),
                    metric_calls=total_calls,
                )
            )
            emit(
                CheckpointWritten(
                    run_id=active_run.id,
                    path=str(active_store.backend_directory.parent),
                )
            )
        emit(
            RunCompleted(
                run_id=active_run.id,
                score=result.effective_score,
                total_metric_calls=total_calls,
            )
        )
        return result

    def run(
        self,
        *,
        seed: Candidate | None = None,
        run: RunConfig | None = None,
        store: RunStore | None = None,
        on_event: Observer | Sequence[Observer] | None = None,
    ) -> PlanResult:
        from ..harness import run_awaitable_sync

        return run_awaitable_sync(self.arun(seed=seed, run=run, store=store, on_event=on_event))

    def snapshot(self) -> PlanSpec:
        return PlanSpec(
            stages=tuple(stage.snapshot() for stage in self.stages),
            initial_candidate=self.initial_candidate,
            budget=self.budget,
            carry_forward=self.carry_forward,
            stop=self.stop,
            aggregate=None if callable(self.aggregate) else self.aggregate,
            aggregate_id=self.aggregate_id,
            weights=dict(self.weights),
            final_rescore_id=self.final_rescore_id,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: PlanSpec,
        *,
        runs: Mapping[str, StageRun],
        rescores: Mapping[str, StageRescore] | None = None,
        aggregates: Mapping[str, Aggregate] | None = None,
    ) -> Plan:
        stages = tuple(stage.build(runs=runs, rescores=rescores) for stage in snapshot.stages)
        final_rescore: StageRescore | None = None
        if snapshot.final_rescore_id is not None:
            if rescores is None or snapshot.final_rescore_id not in rescores:
                raise PlanError(
                    f"No final rescore callable is registered as '{snapshot.final_rescore_id}'."
                )
            final_rescore = rescores[snapshot.final_rescore_id]
        aggregate: Aggregate
        if snapshot.aggregate is not None:
            aggregate = snapshot.aggregate
        elif (
            snapshot.aggregate_id is not None
            and aggregates is not None
            and snapshot.aggregate_id in aggregates
        ):
            aggregate = aggregates[snapshot.aggregate_id]
        else:
            raise PlanError("No custom aggregation callable is registered for this plan.")
        return cls(
            *stages,
            initial_candidate=snapshot.initial_candidate,
            budget=snapshot.budget,
            carry_forward=snapshot.carry_forward,
            stop=snapshot.stop,
            aggregate=aggregate,
            weights=snapshot.weights,
            final_rescore=final_rescore,
            aggregate_id=snapshot.aggregate_id,
            final_rescore_id=snapshot.final_rescore_id,
        )


def _seed_stage(candidate: Candidate, stage: Stage) -> Candidate:
    if stage.seed is None:
        seeded = candidate
    else:
        changed_frozen = {
            name
            for name, value in stage.seed.values.items()
            if name not in stage.components and candidate.values.get(name) != value
        }
        if changed_frozen:
            names = ", ".join(sorted(changed_frozen))
            raise PlanError(f"Stage '{stage.id}' seed changes frozen components: {names}.")
        values = dict(candidate.values)
        for name in stage.components:
            if name in stage.seed.values:
                values[name] = stage.seed.values[name]
        seeded = stage.seed.model_copy(update={"values": values})
    missing = set(stage.components) - set(seeded.values)
    if missing:
        names = ", ".join(sorted(missing))
        raise PlanError(f"Stage '{stage.id}' has no seed value for: {names}.")
    return seeded


def _stage_output(execution: StageExecution) -> StageOutput:
    if isinstance(execution, PydanticGEPAResult):
        return StageOutput.from_result(execution)
    return execution


def _validate_stage_output(
    stage: Stage,
    input_candidate: Candidate,
    output_candidate: Candidate,
) -> None:
    changed_frozen = {
        name
        for name in set(input_candidate.values).union(output_candidate.values)
        if name not in stage.components
        and input_candidate.values.get(name) != output_candidate.values.get(name)
    }
    if changed_frozen:
        names = ", ".join(sorted(changed_frozen))
        raise PlanError(f"Stage '{stage.id}' changed frozen components: {names}.")
    if output_candidate.values.keys() & set(stage.components) != set(stage.components):
        missing = set(stage.components) - output_candidate.values.keys()
        names = ", ".join(sorted(missing))
        raise PlanError(f"Stage '{stage.id}' did not return target components: {names}.")


def _merge_candidate(
    input_candidate: Candidate,
    output_candidate: Candidate,
    components: Sequence[str],
) -> Candidate:
    values = dict(input_candidate.values)
    for name in components:
        values[name] = output_candidate.values[name]
    return output_candidate.model_copy(update={"values": values})


async def _rescore(rescore: StageRescore | None, candidate: Candidate) -> float | None:
    if rescore is None:
        return None
    score = rescore(candidate)
    if inspect.isawaitable(score):
        score = cast("float", await score)
    return float(score)


def _aggregate(
    aggregate: Aggregate,
    stages: Sequence[StageResult],
    weights: Mapping[str, float],
) -> float | None:
    completed = [stage for stage in stages if stage.effective_score is not None]
    if not completed:
        return None
    if callable(aggregate):
        return float(aggregate(completed))
    scores = [cast("float", stage.effective_score) for stage in completed]
    if aggregate == "mean":
        return sum(scores) / len(scores)
    if aggregate == "min":
        return min(scores)
    stage_weights = [weights.get(stage.stage_id, 1.0) for stage in completed]
    total_weight = sum(stage_weights)
    if total_weight <= 0:
        raise PlanError("weighted_mean requires a positive total weight.")
    return (
        sum(score * weight for score, weight in zip(scores, stage_weights, strict=True))
        / total_weight
    )


def _plan_fingerprint(plan: Plan, run: RunConfig) -> CompatibilityFingerprint:
    try:
        backend_version = version("gepa")
    except PackageNotFoundError:
        backend_version = "not-installed"
    return CompatibilityFingerprint.from_dimensions(
        {
            "pydantic-gepa": __version__,
            "gepa": backend_version,
            "plan": content_fingerprint(plan.snapshot()),
            **run.compatibility,
        }
    )


__all__ = ("Plan",)
