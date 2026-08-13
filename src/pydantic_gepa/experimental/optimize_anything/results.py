from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

from ...candidates import Candidate
from ...results import (
    AdaptiveSliceSummary,
    ArtifactReference,
    BudgetSummary,
    CandidateSummary,
    CompositionStepSummary,
    CompositionSummary,
    EngineRunSummary,
    PydanticGEPAResult,
    ScoreSummary,
    SelectionSummary,
)
from ...values import JsonValue
from .adapter import CandidateCodec
from .models import CompositionKind, Engine, EngineResult


def engine_result(
    raw: EngineResult,
    *,
    engine: Engine,
    codec: CandidateCodec,
    input_candidate: Candidate,
    execution_id: str,
    parent_execution_id: str | None,
    pipeline_id: str,
    step_id: str,
    branch_id: str,
) -> EngineRunSummary:
    output_values = codec.decode(raw.best_candidate)
    output_candidate = Candidate(
        values=output_values,
        id=execution_id,
        parent_id=input_candidate.id or input_candidate.fingerprint(),
        generation=(input_candidate.generation or 0) + 1,
        metadata={"engine": engine.name},
    )
    metadata = raw.metadata
    adapter_cost = _finite_number(metadata.get("adapter_cost"))
    total_cost = _finite_number(metadata.get("total_cost"))
    evaluation_cost = None if total_cost is None else max(0.0, total_cost - (adapter_cost or 0.0))
    budget_status = metadata.get("budget")
    exhausted = isinstance(budget_status, Mapping) and budget_status.get("exhausted") is True
    budget = BudgetSummary(
        metric_calls=raw.total_evals,
        metric_call_limit=engine.max_evals,
        reflection_cost=adapter_cost,
        evaluation_calls=raw.total_evals,
        evaluation_call_limit=engine.max_evals,
        optimizer_cost=adapter_cost,
        optimizer_cost_limit=engine.max_token_cost,
        evaluation_cost=evaluation_cost,
        total_cost=total_cost,
        source="upstream",
    )
    duration = _finite_number(metadata.get("wall_time"))
    artifacts = tuple(
        ArtifactReference(kind=kind, path=path)
        for kind, key in (
            ("output", "output_dir"),
            ("workspace", "work_dir"),
            ("run", "run_dir"),
        )
        if isinstance((path := metadata.get(key)), str) and path
    )
    reported: dict[str, JsonValue] = {}
    for key in (
        "baseline_test_score",
        "test_score",
        "best_stage_score",
        "adaptive_stop_reason",
        "engine",
    ):
        value = metadata.get(key)
        if isinstance(value, str | int | float | bool):
            reported[key] = value
    return EngineRunSummary(
        execution_id=execution_id,
        parent_execution_id=parent_execution_id,
        pipeline_id=pipeline_id,
        step_id=step_id,
        branch_id=branch_id,
        engine=engine.name,
        family=engine.kind,
        candidate_mode=codec.mode,
        input_candidate=input_candidate,
        output_candidate=output_candidate,
        search_score=float(raw.best_score),
        stop_reason="budget_exhausted" if exhausted else None,
        budget=budget,
        duration_seconds=duration,
        artifacts=artifacts,
        reported=reported,
    )


def composition_result(
    *,
    kind: CompositionKind,
    pipeline_id: str,
    runs: Sequence[EngineRunSummary],
    selections: Sequence[SelectionSummary] = (),
    selected_execution_id: str | None = None,
    initial_candidate: Candidate,
    run_id: str,
    stage_id: str | None = None,
    budget: BudgetSummary | None = None,
    adaptive_schedule: Sequence[AdaptiveSliceSummary] = (),
    stop_reason: str | None = None,
    step_results: Sequence[PydanticGEPAResult] = (),
) -> PydanticGEPAResult:
    if not runs:
        raise ValueError("An Optimize Anything composition produced no engine runs.")
    by_id = {run.execution_id: run for run in runs}
    selected = (
        by_id[selected_execution_id]
        if selected_execution_id is not None
        else max(runs, key=lambda run: run.search_score)
    )
    active_budget = budget or (
        _pipeline_budget(step_results) if step_results else _composition_budget(runs, selections)
    )
    steps: list[CompositionStepSummary] = []
    for step_result in step_results:
        step_summary = step_result.composition
        if step_result.stage_id is None or step_summary is None:
            raise ValueError("Pipeline step results require stage and composition summaries.")
        steps.append(
            CompositionStepSummary(
                step_id=step_result.stage_id,
                kind=step_summary.kind,
                input_candidate=step_summary.engine_runs[0].input_candidate,
                output_candidate=step_result.final_candidate,
                engine_execution_ids=tuple(run.execution_id for run in step_summary.engine_runs),
                selected_execution_id=step_result.best_candidate.id,
                budget=step_result.budget,
            )
        )
    summary = CompositionSummary(
        kind=kind,
        pipeline_id=pipeline_id,
        engine_runs=tuple(runs),
        selections=tuple(selections),
        steps=tuple(steps),
        adaptive_schedule=tuple(adaptive_schedule),
        stop_reason=stop_reason,
        budget=active_budget,
    )
    history = [
        CandidateSummary(
            candidate_id=run.execution_id,
            parent_ids=[run.output_candidate.parent_id]
            if run.output_candidate.parent_id is not None
            else [],
            generation=run.output_candidate.generation,
            score=run.search_score,
            values=dict(run.output_candidate.values),
            status="best" if run.execution_id == selected.execution_id else "proposed",
            metadata={"engine": run.engine, "branch": run.branch_id},
        )
        for run in runs
    ]
    test_score = selected.reported.get("test_score")
    validation_score = selected.search_score
    artifacts = tuple(artifact for run in runs for artifact in run.artifacts)
    return PydanticGEPAResult(
        best_candidate=selected.output_candidate,
        final_candidate=(
            initial_candidate
            if kind == "parallel" and not selections
            else selected.output_candidate
        ),
        best_score=selected.search_score,
        backend="optimize_anything",
        run_id=run_id,
        stage_id=stage_id,
        composition=summary,
        scores=ScoreSummary(
            search=tuple(run.search_score for run in runs),
            validation=validation_score,
            test=float(test_score) if isinstance(test_score, int | float) else None,
            aggregate=selected.search_score,
        ),
        budget=active_budget,
        stop_reason=stop_reason or selected.stop_reason,
        artifacts=artifacts,
        reported={"composition": kind, "pipeline_id": pipeline_id},
        best_candidate_index=next(
            index for index, run in enumerate(runs) if run.execution_id == selected.execution_id
        ),
        validation_scores=[run.search_score for run in runs],
        candidate_history=history,
        candidates=[run.output_candidate for run in runs],
        total_metric_calls=active_budget.evaluation_calls,
        run_dir=next((artifact.path for artifact in artifacts if artifact.kind == "run"), None),
    )


def _composition_budget(
    runs: Sequence[EngineRunSummary],
    selections: Sequence[SelectionSummary],
) -> BudgetSummary:
    evaluation_calls = sum(run.budget.evaluation_calls or 0 for run in runs)
    evaluation_limits = [
        run.budget.evaluation_call_limit
        for run in runs
        if run.budget.evaluation_call_limit is not None
    ]
    optimizer_costs = [
        run.budget.optimizer_cost for run in runs if run.budget.optimizer_cost is not None
    ]
    optimizer_limits = [
        run.budget.optimizer_cost_limit
        for run in runs
        if run.budget.optimizer_cost_limit is not None
    ]
    evaluation_costs = [
        run.budget.evaluation_cost for run in runs if run.budget.evaluation_cost is not None
    ]
    total_costs = [run.budget.total_cost for run in runs if run.budget.total_cost is not None]
    final_rescore_calls = sum(selection.evaluation_calls for selection in selections)
    return BudgetSummary(
        metric_calls=evaluation_calls,
        metric_call_limit=sum(evaluation_limits) if evaluation_limits else None,
        reflection_cost=sum(optimizer_costs) if optimizer_costs else None,
        evaluation_calls=evaluation_calls,
        evaluation_call_limit=sum(evaluation_limits) if evaluation_limits else None,
        optimizer_cost=sum(optimizer_costs) if optimizer_costs else None,
        optimizer_cost_limit=sum(optimizer_limits) if optimizer_limits else None,
        evaluation_cost=sum(evaluation_costs) if evaluation_costs else None,
        total_cost=sum(total_costs) if total_costs else None,
        final_rescore_calls=final_rescore_calls,
        source="upstream",
    )


def _pipeline_budget(step_results: Sequence[PydanticGEPAResult]) -> BudgetSummary:
    budgets = [result.budget for result in step_results]
    evaluation_limits = [
        budget.evaluation_call_limit
        for budget in budgets
        if budget.evaluation_call_limit is not None
    ]
    optimizer_costs = [
        budget.optimizer_cost for budget in budgets if budget.optimizer_cost is not None
    ]
    optimizer_limits = [
        budget.optimizer_cost_limit for budget in budgets if budget.optimizer_cost_limit is not None
    ]
    evaluation_costs = [
        budget.evaluation_cost for budget in budgets if budget.evaluation_cost is not None
    ]
    total_costs = [budget.total_cost for budget in budgets if budget.total_cost is not None]
    final_rescore_costs = [
        budget.final_rescore_cost for budget in budgets if budget.final_rescore_cost is not None
    ]
    return BudgetSummary(
        metric_calls=sum(budget.evaluation_calls or 0 for budget in budgets),
        metric_call_limit=sum(evaluation_limits) if evaluation_limits else None,
        reflection_cost=sum(optimizer_costs) if optimizer_costs else None,
        evaluation_calls=sum(budget.evaluation_calls or 0 for budget in budgets),
        evaluation_call_limit=sum(evaluation_limits) if evaluation_limits else None,
        optimizer_cost=sum(optimizer_costs) if optimizer_costs else None,
        optimizer_cost_limit=sum(optimizer_limits) if optimizer_limits else None,
        evaluation_cost=sum(evaluation_costs) if evaluation_costs else None,
        total_cost=sum(total_costs) if total_costs else None,
        final_rescore_calls=sum(budget.final_rescore_calls for budget in budgets),
        final_rescore_cost=(sum(final_rescore_costs) if final_rescore_costs else None),
        heldout_evaluation_calls=sum(budget.heldout_evaluation_calls for budget in budgets),
        source="mixed",
    )


def _finite_number(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if isfinite(number) and number >= 0:
            return number
    return None


__all__ = (
    "composition_result",
    "engine_result",
)
