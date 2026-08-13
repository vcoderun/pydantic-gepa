from __future__ import annotations as _annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from ..._version import __version__
from ...candidates import Candidate
from ...configuration import BudgetConfig, ConfigurationError, EvaluationSetConfig, GEPAConfig
from ...errors import OptimizationDependencyError, RunStoreError
from ...eventing import EvaluationEventSink, budget_snapshot, run_declaration
from ...events import (
    BudgetExhausted,
    BudgetUpdated,
    CandidateAccepted,
    CheckpointRejected,
    CheckpointReset,
    CheckpointResumed,
    CheckpointWritten,
    ComponentsRegistered,
    FinalRescoreCompleted,
    FinalRescoreStarted,
    RunCancelled,
    RunCompleted,
    RunFailed,
    RunStarted,
    SelectionCompleted,
    StageCompleted,
    StageFailed,
    StageStarted,
    _dispatcher,
    _event_scope,
    _EventDispatcher,
)
from ...recorder import GEPAEventBridge
from ...results import (
    AdaptiveSliceSummary,
    BudgetSummary,
    EngineRunSummary,
    PydanticGEPAResult,
    SelectionSummary,
)
from ...state import CompatibilityFingerprint, FileRunStore, RunState, content_fingerprint
from ...values import JsonValue
from .adapter import (
    CandidateCodec,
    CandidateValue,
    EvaluationOutput,
    OptimizationState,
    PydanticOptimizeAnythingAdapter,
)
from .backend import (
    BatchEvaluator,
    Evaluator,
    ExecutionTracker,
    OptimizeAnythingFn,
    configure_engine,
    execute,
)
from .models import (
    AdaptiveSequential,
    BestOf,
    Composition,
    Engine,
    OptimizeAnythingConfig,
    Parallel,
    Pipeline,
    Sequential,
    Single,
    Vote,
)
from .results import composition_result, engine_result

CaseT = TypeVar("CaseT")
RolloutOutputT = TypeVar("RolloutOutputT")
EvaluatorT = TypeVar("EvaluatorT")


@dataclass(frozen=True, slots=True)
class _StepOutcome:
    runs: tuple[EngineRunSummary, ...]
    selections: tuple[SelectionSummary, ...]
    selected_execution_id: str | None
    budget: BudgetSummary | None = None
    adaptive_schedule: tuple[AdaptiveSliceSummary, ...] = ()
    stop_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _DurableRun:
    store: FileRunStore
    state: RunState


class PydanticOptimizeAnythingOptimizer(
    BaseModel,
    Generic[CaseT, RolloutOutputT, EvaluatorT],
):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter: PydanticOptimizeAnythingAdapter[CaseT, RolloutOutputT, EvaluatorT]
    initial_candidate: Candidate
    optimization_objective: str | None = None
    background: str | None = None
    optimize_fn: OptimizeAnythingFn[CaseT] | None = None

    def optimize(
        self,
        *,
        trainset: Sequence[CaseT],
        valset: Sequence[CaseT] | None = None,
        testset: Sequence[CaseT] = (),
        initial_candidate: Candidate | None = None,
        config: OptimizeAnythingConfig | GEPAConfig | None = None,
        max_metric_calls: int | None = None,
        allow_same_train_val: bool | None = None,
        objective: str | None = None,
        background: str | None = None,
    ) -> PydanticGEPAResult:
        if objective is not None and not isinstance(objective, str):
            raise TypeError("objective override must be a string.")
        if background is not None and not isinstance(background, str):
            raise TypeError("background override must be a string.")
        active_config, allow_reuse = _config(
            config,
            max_metric_calls=max_metric_calls,
            allow_same_train_val=allow_same_train_val,
        )
        active_valset = list(trainset if valset is None else valset)
        if valset is None and not allow_reuse:
            raise OptimizationDependencyError(
                "valset is required unless allow_same_train_val=True."
            )

        seed = initial_candidate or self.initial_candidate
        composition = active_config.active_composition
        engines = _engines(composition)
        mode = (
            "text" if any(engine.candidate_mode == "text" for engine in engines) else "components"
        )
        codec = CandidateCodec(seed=seed, mode=mode, component=active_config.component)
        active_objective = objective or active_config.objective or self.optimization_objective
        active_background = background or active_config.background or self.background
        dispatcher = _dispatcher(
            run_id=active_config.run.id,
            backend="optimize_anything",
            local_observers=active_config.tracking.observers,
            local_error_policy=active_config.tracking.observer_errors,
        )
        base_adapter = self.adapter.adapter.model_copy(
            update={
                "events": EvaluationEventSink(
                    dispatcher,
                    objective=self.adapter.adapter.objective,
                    trainset=trainset,
                    valset=active_valset,
                    testset=testset,
                )
            }
        )
        active_optimizer = self.model_copy(
            update={"adapter": self.adapter.model_copy(update={"adapter": base_adapter})}
        )
        declaration_values = dict(active_config.declaration())
        configuration_fingerprint = content_fingerprint(
            _compatibility_declaration(declaration_values)
        )
        composition_declaration = declaration_values.get(
            "composition",
            declaration_values.get("engine"),
        )
        declaration = run_declaration(
            configuration_fingerprint=configuration_fingerprint,
            composition_fingerprint=content_fingerprint(composition_declaration),
            objective=self.adapter.adapter.objective,
            trainset=trainset,
            valset=active_valset,
            testset=testset,
            evaluation_call_limit=_evaluation_limit(composition),
            optimizer_cost_limit=_optimizer_cost_limit(composition),
            checkpoint_path=(
                None if active_config.run.directory is None else str(active_config.run.directory)
            ),
            engine_declaration=declaration_values,
        )
        dispatcher.emit(
            RunStarted(
                run_id=active_config.run.id,
                seed=seed,
                candidate_id=seed.id or seed.fingerprint(),
                composition=composition.kind,
                pipeline_id=f"{active_config.run.id}:pipeline",
                declaration=declaration,
            )
        )
        pipeline_id = f"{active_config.run.id}:pipeline"
        durable: _DurableRun | None = None
        run_directory = active_config.run.directory
        try:
            if base_adapter.components is not None:
                dispatcher.emit(
                    ComponentsRegistered(
                        run_id=active_config.run.id,
                        components=base_adapter.components.components,
                        composition=composition.kind,
                        pipeline_id=pipeline_id,
                    )
                )
            try:
                durable = (
                    None
                    if run_directory is None
                    else _store(
                        active_config,
                        directory=run_directory,
                        seed=seed,
                        trainset=trainset,
                        valset=active_valset,
                        testset=testset,
                        objective=active_objective,
                        background=active_background,
                    )
                )
            except Exception as exc:
                dispatcher.emit(
                    CheckpointRejected(
                        run_id=active_config.run.id,
                        composition=composition.kind,
                        pipeline_id=pipeline_id,
                        path=str(run_directory),
                        reason=str(exc),
                    )
                )
                raise
            if durable is not None:
                state = durable.state
                if state.reset:
                    dispatcher.emit(
                        CheckpointReset(
                            run_id=active_config.run.id,
                            composition=composition.kind,
                            pipeline_id=pipeline_id,
                            path=str(run_directory),
                        )
                    )
                if state.resumed:
                    dispatcher.emit(
                        CheckpointResumed(
                            run_id=active_config.run.id,
                            composition=composition.kind,
                            pipeline_id=pipeline_id,
                            path=str(run_directory),
                        )
                    )
                if state.status == "completed":
                    completed = durable.store.load_result(PydanticGEPAResult)
                    if completed is None:
                        raise RunStoreError(
                            "Completed Optimize Anything checkpoint has no result artifact."
                        )
                    dispatcher.emit(
                        RunCompleted(
                            run_id=active_config.run.id,
                            candidate_id=completed.best_candidate.id
                            or completed.best_candidate.fingerprint(),
                            composition=composition.kind,
                            pipeline_id=pipeline_id,
                            score=completed.best_score,
                            total_metric_calls=completed.total_metric_calls,
                            budget=budget_snapshot(completed.budget),
                        )
                    )
                    return completed
            with _event_scope(dispatcher):
                result = active_optimizer._execute(
                    composition,
                    codec=codec,
                    trainset=list(trainset),
                    valset=active_valset,
                    objective=active_objective,
                    background=active_background,
                    pipeline_id=pipeline_id,
                    backend_directory=(
                        None if durable is None else durable.store.backend_directory
                    ),
                    run_id=active_config.run.id,
                    durable=durable,
                    dispatcher=dispatcher,
                )
                result = result.normalize_candidates(active_optimizer.adapter.normalize_candidate)
                result = active_optimizer._with_heldout_scores(
                    result,
                    seed=seed,
                    testset=list(testset),
                    composition=composition.kind,
                    pipeline_id=pipeline_id,
                    dispatcher=dispatcher,
                )
            if durable is not None:
                store = durable.store
                state = durable.state
                current_state = store.load_state() or state
                store.write_candidate(result.best_candidate)
                store.write_result(result)
                store.checkpoint(
                    current_state.model_copy(
                        update={
                            "status": "completed",
                            "accepted_candidate": result.best_candidate,
                            "metric_calls": result.total_metric_calls or 0,
                            "backend_checkpoint": str(store.backend_directory),
                        }
                    )
                )
                dispatcher.emit(
                    CheckpointWritten(
                        run_id=active_config.run.id,
                        composition=composition.kind,
                        pipeline_id=pipeline_id,
                        path=str(store.directory),
                    )
                )
        except Exception as exc:
            if durable is not None:
                store = durable.store
                state = durable.state
                current_state = store.load_state() or state
                store.checkpoint(
                    current_state.model_copy(
                        update={"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    )
                )
                dispatcher.emit(
                    CheckpointWritten(
                        run_id=active_config.run.id,
                        composition=composition.kind,
                        pipeline_id=pipeline_id,
                        path=str(store.directory),
                    )
                )
            dispatcher.emit(
                RunFailed(
                    run_id=active_config.run.id,
                    composition=composition.kind,
                    pipeline_id=pipeline_id,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            raise
        except BaseException as exc:
            if durable is not None:
                store = durable.store
                state = durable.state
                current_state = store.load_state() or state
                store.checkpoint(
                    current_state.model_copy(
                        update={"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    )
                )
                dispatcher.emit(
                    CheckpointWritten(
                        run_id=active_config.run.id,
                        composition=composition.kind,
                        pipeline_id=pipeline_id,
                        path=str(store.directory),
                    )
                )
            dispatcher.emit(
                RunCancelled(
                    run_id=active_config.run.id,
                    composition=composition.kind,
                    pipeline_id=pipeline_id,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            raise
        dispatcher.emit(
            CandidateAccepted(
                run_id=active_config.run.id,
                composition=composition.kind,
                pipeline_id=pipeline_id,
                candidate_id=result.best_candidate.id or result.best_candidate.fingerprint(),
                parent_ids=(seed.id or seed.fingerprint(),),
                score=result.best_score,
            )
        )
        dispatcher.emit(
            BudgetUpdated(
                run_id=active_config.run.id,
                composition=composition.kind,
                pipeline_id=pipeline_id,
                used=result.budget.evaluation_calls or result.total_metric_calls or 0,
                remaining=(
                    None
                    if result.budget.evaluation_call_limit is None
                    else max(
                        0,
                        result.budget.evaluation_call_limit
                        - (result.budget.evaluation_calls or result.total_metric_calls or 0),
                    )
                ),
                optimizer_cost=result.budget.optimizer_cost,
                optimizer_cost_remaining=(
                    None
                    if result.budget.optimizer_cost_limit is None
                    else max(
                        0.0,
                        result.budget.optimizer_cost_limit - (result.budget.optimizer_cost or 0.0),
                    )
                ),
                evaluation_cost=result.budget.evaluation_cost,
                total_cost=result.budget.total_cost,
            )
        )
        if result.stop_reason == "budget_exhausted":
            dispatcher.emit(
                BudgetExhausted(
                    run_id=active_config.run.id,
                    composition=composition.kind,
                    pipeline_id=pipeline_id,
                    used=result.budget.evaluation_calls or result.total_metric_calls or 0,
                )
            )
        dispatcher.emit(
            RunCompleted(
                run_id=active_config.run.id,
                composition=composition.kind,
                pipeline_id=pipeline_id,
                candidate_id=result.best_candidate.id or result.best_candidate.fingerprint(),
                score=result.best_score,
                total_metric_calls=result.total_metric_calls,
                budget=budget_snapshot(result.budget),
            )
        )
        return result

    def _with_heldout_scores(
        self,
        result: PydanticGEPAResult,
        *,
        seed: Candidate,
        testset: list[CaseT],
        composition: str,
        pipeline_id: str,
        dispatcher: _EventDispatcher,
    ) -> PydanticGEPAResult:
        if not testset:
            return result
        stage_id = f"{pipeline_id}:heldout-test"
        dispatcher.emit(
            FinalRescoreStarted(
                run_id=dispatcher.run_id,
                composition=composition,
                pipeline_id=pipeline_id,
                stage_id=stage_id,
                stage_kind="rescore",
                candidate_id=seed.id or seed.fingerprint(),
            )
        )
        with _event_scope(
            dispatcher,
            stage_id=stage_id,
            composition=composition,
            pipeline_id=pipeline_id,
            stage_kind="rescore",
        ):
            baseline_outputs = self.adapter.batch_evaluator(
                [(seed.values, example) for example in testset]
            )
            baseline_score = sum(score for score, _ in baseline_outputs) / len(baseline_outputs)
            final_candidate = result.final_candidate or result.best_candidate
            heldout_calls = len(testset)
            if final_candidate.fingerprint() == seed.fingerprint():
                final_score = baseline_score
            else:
                final_outputs = self.adapter.batch_evaluator(
                    [(final_candidate.values, example) for example in testset]
                )
                final_score = sum(score for score, _ in final_outputs) / len(final_outputs)
                heldout_calls += len(testset)
        dispatcher.emit(
            FinalRescoreCompleted(
                run_id=dispatcher.run_id,
                composition=composition,
                pipeline_id=pipeline_id,
                stage_id=stage_id,
                stage_kind="rescore",
                candidate_id=final_candidate.id or final_candidate.fingerprint(),
                score=final_score,
                metadata={"baseline_score": baseline_score, "case_count": len(testset)},
            )
        )
        scores = result.scores.model_copy(
            update={"baseline_test": baseline_score, "test": final_score}
        )
        budget = result.budget.model_copy(update={"heldout_evaluation_calls": heldout_calls})
        reported = dict(result.reported)
        reported.update(
            {
                "baseline_test_score": baseline_score,
                "test_score": final_score,
            }
        )
        return result.model_copy(update={"scores": scores, "budget": budget, "reported": reported})

    def _execute(
        self,
        composition: Composition,
        *,
        codec: CandidateCodec,
        trainset: list[CaseT],
        valset: list[CaseT],
        objective: str | None,
        background: str | None,
        pipeline_id: str,
        backend_directory: Path | None,
        run_id: str,
        durable: _DurableRun | None,
        dispatcher: _EventDispatcher,
    ) -> PydanticGEPAResult:
        if isinstance(composition, Pipeline):
            return self._pipeline(
                composition,
                codec=codec,
                trainset=trainset,
                valset=valset,
                objective=objective,
                background=background,
                pipeline_id=pipeline_id,
                backend_directory=backend_directory,
                run_id=run_id,
                durable=durable,
                dispatcher=dispatcher,
            )
        outcome = self._step(
            composition,
            codec=codec,
            trainset=trainset,
            valset=valset,
            objective=objective,
            background=background,
            pipeline_id=pipeline_id,
            step_index=0,
            backend_directory=backend_directory,
            parent_execution_id=None,
            dispatcher=dispatcher,
        )
        return composition_result(
            kind=composition.kind,
            pipeline_id=pipeline_id,
            runs=outcome.runs,
            selections=outcome.selections,
            selected_execution_id=outcome.selected_execution_id,
            initial_candidate=codec.seed,
            run_id=run_id,
            budget=outcome.budget,
            adaptive_schedule=outcome.adaptive_schedule,
            stop_reason=outcome.stop_reason,
        )

    def _pipeline(
        self,
        pipeline: Pipeline,
        *,
        codec: CandidateCodec,
        trainset: list[CaseT],
        valset: list[CaseT],
        objective: str | None,
        background: str | None,
        pipeline_id: str,
        backend_directory: Path | None,
        run_id: str,
        durable: _DurableRun | None,
        dispatcher: _EventDispatcher,
    ) -> PydanticGEPAResult:
        state = None if durable is None else durable.state
        step_results = (
            []
            if state is None
            else [PydanticGEPAResult.model_validate(item) for item in state.stages]
        )
        start_step = 0 if state is None else state.next_stage
        if start_step != len(step_results) or start_step > len(pipeline.steps):
            raise ConfigurationError("Pipeline checkpoint steps are inconsistent with run state.")
        active_candidate = codec.seed if state is None else state.accepted_candidate
        active_codec = CandidateCodec(
            seed=active_candidate,
            mode=codec.mode,
            component=codec.component,
        )
        parent_execution_id = active_candidate.id
        for step_index, step in enumerate(pipeline.steps[start_step:], start=start_step):
            outcome = self._step(
                step,
                codec=active_codec,
                trainset=trainset,
                valset=valset,
                objective=objective,
                background=background,
                pipeline_id=pipeline_id,
                step_index=step_index,
                backend_directory=backend_directory,
                parent_execution_id=parent_execution_id,
                dispatcher=dispatcher,
            )
            selected_id = outcome.selected_execution_id
            if selected_id is None:
                raise RuntimeError("A validated pipeline step produced no selected candidate.")
            selected_run = next(run for run in outcome.runs if run.execution_id == selected_id)
            step_result = composition_result(
                kind=step.kind,
                pipeline_id=pipeline_id,
                runs=outcome.runs,
                selections=outcome.selections,
                selected_execution_id=selected_id,
                initial_candidate=active_codec.seed,
                run_id=run_id,
                stage_id=f"step-{step_index}",
                budget=outcome.budget,
                adaptive_schedule=outcome.adaptive_schedule,
                stop_reason=outcome.stop_reason,
            )
            step_results.append(step_result)
            if step_index < len(pipeline.steps) - 1:
                dispatcher.emit(
                    SelectionCompleted(
                        run_id=dispatcher.run_id,
                        composition="pipeline",
                        pipeline_id=pipeline_id,
                        step_id=f"step-{step_index}",
                        stage_id=f"step-{step_index}",
                        method="pipeline",
                        selected_execution_id=selected_run.execution_id,
                        contender_execution_ids=tuple(run.execution_id for run in outcome.runs),
                        contender_scores=tuple(run.search_score for run in outcome.runs),
                        score=selected_run.search_score,
                        reason="continue_with_selected_candidate",
                        candidate_id=selected_run.output_candidate.id
                        or selected_run.output_candidate.fingerprint(),
                    )
                )
            parent_execution_id = selected_run.execution_id
            active_codec = CandidateCodec(
                seed=selected_run.output_candidate,
                mode=codec.mode,
                component=codec.component,
            )
            if durable is not None:
                store = durable.store
                state = durable.state
                store.write_candidate(selected_run.output_candidate)
                store.write_stage(step_result)
                store.checkpoint(
                    RunState(
                        run_id=state.run_id,
                        fingerprint=state.fingerprint,
                        accepted_candidate=selected_run.output_candidate,
                        stages=tuple(item.model_dump(mode="json") for item in step_results),
                        next_stage=step_index + 1,
                        metric_calls=sum(item.total_metric_calls or 0 for item in step_results),
                        backend_checkpoint=str(store.backend_directory),
                    )
                )
                dispatcher.emit(
                    CheckpointWritten(
                        run_id=dispatcher.run_id,
                        composition="pipeline",
                        pipeline_id=pipeline_id,
                        step_id=f"step-{step_index}",
                        path=str(store.directory),
                    )
                )

        all_runs: list[EngineRunSummary] = []
        all_selections: list[SelectionSummary] = []
        selected_id: str | None = None
        for step_index, step_result in enumerate(step_results):
            summary = step_result.composition
            if summary is None:
                raise RuntimeError("A pipeline step checkpoint has no composition summary.")
            all_runs.extend(summary.engine_runs)
            all_selections.extend(summary.selections)
            selected_id = step_result.best_candidate.id
            if selected_id is None:
                raise RuntimeError("A pipeline step checkpoint has no selected execution id.")
            if step_index < len(step_results) - 1:
                selected_run = next(
                    run for run in summary.engine_runs if run.execution_id == selected_id
                )
                all_selections.append(
                    SelectionSummary(
                        method="pipeline",
                        selected_execution_id=selected_run.execution_id,
                        selected_candidate=selected_run.output_candidate,
                        score=selected_run.search_score,
                        contender_execution_ids=tuple(
                            run.execution_id for run in summary.engine_runs
                        ),
                        contender_scores=tuple(run.search_score for run in summary.engine_runs),
                        reason="continue_with_selected_candidate",
                    )
                )
        return composition_result(
            kind="pipeline",
            pipeline_id=pipeline_id,
            runs=all_runs,
            selections=all_selections,
            selected_execution_id=selected_id,
            initial_candidate=codec.seed,
            run_id=run_id,
            step_results=step_results,
        )

    def _step(
        self,
        composition: Single | Sequential | Parallel | BestOf | Vote | AdaptiveSequential,
        *,
        codec: CandidateCodec,
        trainset: list[CaseT],
        valset: list[CaseT],
        objective: str | None,
        background: str | None,
        pipeline_id: str,
        step_index: int,
        backend_directory: Path | None,
        parent_execution_id: str | None,
        dispatcher: _EventDispatcher,
    ) -> _StepOutcome:
        step_id = f"step-{step_index}"
        engines = _engines(composition)
        dispatcher.emit(
            StageStarted(
                run_id=dispatcher.run_id,
                composition=composition.kind,
                pipeline_id=pipeline_id,
                step_id=step_id,
                stage_id=step_id,
                stage_kind="composition",
                candidate_id=codec.seed.id or codec.seed.fingerprint(),
                parent_execution_id=parent_execution_id,
            )
        )
        try:
            tracker = ExecutionTracker(
                dispatcher=dispatcher,
                codec=codec,
                engines=engines,
                composition=composition.kind,
                pipeline_id=pipeline_id,
                step_id=step_id,
                parent_execution_id=parent_execution_id,
            )
            runtimes = [
                configure_engine(
                    engine,
                    engine_index=index,
                    name=f"{pipeline_id}:{step_id}:{index}",
                    run_dir=(
                        None
                        if backend_directory is None
                        else backend_directory / step_id / f"branch-{index}"
                    ),
                    callback=GEPAEventBridge(
                        run_id=dispatcher.run_id,
                        on_event=dispatcher.emit,
                        lifecycle="backend_only",
                        engine=engine.name,
                        composition=composition.kind,
                        pipeline_id=pipeline_id,
                        step_id=step_id,
                    ),
                    tracker=tracker,
                    observe_engine=not isinstance(composition, Single),
                )
                for index, engine in enumerate(engines)
            ]
            evaluate, batch_evaluate = self._evaluators(codec)
            execution = execute(
                composition,
                codec=codec,
                evaluate=evaluate,
                batch_evaluate=batch_evaluate,
                trainset=trainset,
                valset=valset,
                objective=objective,
                background=background,
                runtimes=runtimes,
                optimize_fn=self.optimize_fn,
            )
        except BaseException as exc:
            dispatcher.emit(
                StageFailed(
                    run_id=dispatcher.run_id,
                    composition=composition.kind,
                    pipeline_id=pipeline_id,
                    step_id=step_id,
                    stage_id=step_id,
                    stage_kind="composition",
                    candidate_id=codec.seed.id or codec.seed.fingerprint(),
                    parent_execution_id=parent_execution_id,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            raise
        runs: list[EngineRunSummary] = []
        running_best = float("-inf")
        for index, raw in enumerate(execution.results):
            engine_index = execution.engine_indices[index]
            active_engine = engines[engine_index]
            invocation = tracker.resolve(raw, engine_index=engine_index)
            active_codec = CandidateCodec(
                seed=invocation.input_candidate,
                mode=codec.mode,
                component=codec.component,
            )
            run = engine_result(
                raw,
                engine=active_engine,
                codec=active_codec,
                input_candidate=invocation.input_candidate,
                execution_id=invocation.execution_id,
                parent_execution_id=invocation.parent_execution_id,
                pipeline_id=pipeline_id,
                step_id=step_id,
                branch_id=invocation.branch_id,
            )
            if isinstance(composition, AdaptiveSequential):
                schedule = execution.adaptive_schedule[index]
                optimizer_cost = schedule.optimizer_cost
                run = run.model_copy(
                    update={
                        "budget": BudgetSummary(
                            metric_calls=schedule.evaluation_calls,
                            metric_call_limit=composition.plateau_evals,
                            reflection_cost=optimizer_cost,
                            evaluation_calls=schedule.evaluation_calls,
                            evaluation_call_limit=composition.plateau_evals,
                            optimizer_cost=optimizer_cost,
                            optimizer_cost_limit=active_engine.max_token_cost,
                            source="mixed",
                        )
                    }
                )
            runs.append(run)
            if isinstance(composition, Sequential | AdaptiveSequential) and (
                run.search_score > running_best
            ):
                running_best = run.search_score
        if isinstance(composition, Parallel):
            dispatcher.emit(
                StageCompleted(
                    run_id=dispatcher.run_id,
                    composition=composition.kind,
                    pipeline_id=pipeline_id,
                    step_id=step_id,
                    stage_id=step_id,
                    stage_kind="composition",
                    candidate_id=codec.seed.id or codec.seed.fingerprint(),
                )
            )
            return _StepOutcome(runs=tuple(runs), selections=(), selected_execution_id=None)
        scores = (
            execution.selection_scores
            if execution.selection_scores is not None
            else tuple(run.search_score for run in runs)
        )
        selected_index = execution.selected_index
        if selected_index is None:
            selected_index = max(range(len(runs)), key=lambda index: scores[index])
        selected = runs[selected_index]
        method = (
            "vote"
            if isinstance(composition, Vote)
            else "adaptive"
            if isinstance(composition, AdaptiveSequential)
            else "best_score"
        )
        selection = SelectionSummary(
            method=method,
            selected_execution_id=selected.execution_id,
            selected_candidate=selected.output_candidate,
            score=scores[selected_index],
            contender_execution_ids=tuple(run.execution_id for run in runs),
            contender_scores=scores,
            evaluation_calls=execution.selection_evaluation_calls,
            reason=execution.stop_reason,
        )
        if isinstance(composition, Vote):
            runs = [
                run.model_copy(update={"selection_score": scores[index]})
                for index, run in enumerate(runs)
            ]
        dispatcher.emit(
            SelectionCompleted(
                run_id=dispatcher.run_id,
                composition=composition.kind,
                pipeline_id=pipeline_id,
                step_id=step_id,
                stage_id=step_id,
                method=selection.method,
                selected_execution_id=selection.selected_execution_id,
                contender_execution_ids=selection.contender_execution_ids,
                contender_scores=selection.contender_scores,
                score=selection.score,
                reason=selection.reason,
                candidate_id=selection.selected_candidate.id
                or selection.selected_candidate.fingerprint(),
            )
        )
        budget: BudgetSummary | None = None
        if isinstance(composition, AdaptiveSequential):
            total_cost = execution.total_cost
            optimizer_costs = [
                item.optimizer_cost
                for item in execution.adaptive_schedule
                if item.optimizer_cost is not None
            ]
            optimizer_cost = sum(optimizer_costs) if optimizer_costs else None
            budget = BudgetSummary(
                metric_calls=sum(execution.evaluation_calls or ()),
                metric_call_limit=composition.max_evals,
                reflection_cost=optimizer_cost,
                evaluation_calls=sum(execution.evaluation_calls or ()),
                evaluation_call_limit=composition.max_evals,
                optimizer_cost=optimizer_cost,
                optimizer_cost_limit=sum(engine.max_token_cost or 0.0 for engine in engines)
                or None,
                evaluation_cost=(
                    None if total_cost is None else max(0.0, total_cost - (optimizer_cost or 0.0))
                ),
                total_cost=total_cost,
                source="mixed",
            )
        dispatcher.emit(
            StageCompleted(
                run_id=dispatcher.run_id,
                composition=composition.kind,
                pipeline_id=pipeline_id,
                step_id=step_id,
                stage_id=step_id,
                stage_kind="composition",
                candidate_id=selected.output_candidate.id
                or selected.output_candidate.fingerprint(),
                score=selection.score,
            )
        )
        return _StepOutcome(
            runs=tuple(runs),
            selections=(selection,),
            selected_execution_id=selected.execution_id,
            budget=budget,
            adaptive_schedule=execution.adaptive_schedule,
            stop_reason=execution.stop_reason,
        )

    def _evaluators(
        self,
        codec: CandidateCodec,
    ) -> tuple[Evaluator[CaseT], BatchEvaluator[CaseT]]:
        def evaluate(
            candidate: CandidateValue,
            example: CaseT,
            opt_state: OptimizationState | None = None,
        ) -> EvaluationOutput:
            return self.adapter.evaluator(codec.decode(candidate), example, opt_state)

        def batch_evaluate(
            pairs: Sequence[tuple[CandidateValue, CaseT]],
            *,
            opt_states: Sequence[OptimizationState | None] | None = None,
        ) -> list[EvaluationOutput]:
            decoded = [(codec.decode(candidate), example) for candidate, example in pairs]
            return self.adapter.batch_evaluator(decoded, opt_states=opt_states)

        return evaluate, batch_evaluate


def _config(
    config: OptimizeAnythingConfig | GEPAConfig | None,
    *,
    max_metric_calls: int | None,
    allow_same_train_val: bool | None,
) -> tuple[OptimizeAnythingConfig, bool]:
    if isinstance(config, OptimizeAnythingConfig):
        if max_metric_calls is not None or allow_same_train_val is not None:
            raise ConfigurationError(
                "OptimizeAnythingConfig cannot be combined with legacy runtime shortcuts."
            )
        return config, False
    if isinstance(config, GEPAConfig):
        if max_metric_calls is not None or allow_same_train_val is not None:
            raise ConfigurationError(
                "GEPAConfig cannot be combined with max_metric_calls or allow_same_train_val."
            )
        warnings.warn(
            "Passing GEPAConfig to the Optimize Anything backend is deprecated; use "
            "OptimizeAnythingConfig(engine=Engine.gepa(...)).",
            DeprecationWarning,
            stacklevel=3,
        )
        return (
            OptimizeAnythingConfig(
                engine=Engine.gepa(config),
                run=config.run,
                tracking=config.tracking,
            ),
            config.evaluation_sets.allow_same_train_validation,
        )
    legacy = GEPAConfig(
        budget=BudgetConfig(max_metric_calls=max_metric_calls or 50),
        evaluation_sets=EvaluationSetConfig(allow_same_train_validation=bool(allow_same_train_val)),
    )
    return (
        OptimizeAnythingConfig(
            engine=Engine.gepa(legacy),
            run=legacy.run,
            tracking=legacy.tracking,
        ),
        legacy.evaluation_sets.allow_same_train_validation,
    )


def _engines(composition: Composition) -> tuple[Engine, ...]:
    if isinstance(composition, Single):
        return (composition.engine,)
    if isinstance(composition, Pipeline):
        return tuple(engine for step in composition.steps for engine in _engines(step))
    return composition.engines


def _evaluation_limit(composition: Composition) -> int | None:
    if isinstance(composition, AdaptiveSequential):
        return composition.max_evals
    if isinstance(composition, Pipeline):
        limits = [_evaluation_limit(step) for step in composition.steps]
    else:
        limits = [engine.max_evals for engine in _engines(composition)]
    if any(limit is None for limit in limits):
        return None
    return sum(limit for limit in limits if limit is not None)


def _optimizer_cost_limit(composition: Composition) -> float | None:
    limits = [engine.max_token_cost for engine in _engines(composition)]
    if any(limit is None for limit in limits):
        return None
    return sum(limit for limit in limits if limit is not None)


def _store(
    config: OptimizeAnythingConfig,
    *,
    directory: Path,
    seed: Candidate,
    trainset: Sequence[CaseT],
    valset: Sequence[CaseT],
    testset: Sequence[CaseT],
    objective: str | None,
    background: str | None,
) -> _DurableRun:
    store = FileRunStore(
        directory,
        run_id=config.run.id,
        resume=config.run.resume,
        fresh=config.run.fresh,
    )
    try:
        gepa_version = version("gepa")
    except PackageNotFoundError:
        gepa_version = "not-installed"
    fingerprint = CompatibilityFingerprint.from_dimensions(
        {
            "backend": "optimize_anything",
            "pydantic-gepa": __version__,
            "gepa": gepa_version,
            "candidate": seed.fingerprint(),
            "configuration": content_fingerprint(
                _compatibility_declaration(dict(config.declaration()))
            ),
            "trainset": content_fingerprint(trainset),
            "valset": content_fingerprint(valset),
            "testset": content_fingerprint(testset),
            "objective": content_fingerprint(objective),
            "background": content_fingerprint(background),
            **config.run.compatibility,
        }
    )
    return _DurableRun(
        store=store,
        state=store.prepare(fingerprint=fingerprint, initial_candidate=seed),
    )


def _compatibility_declaration(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_compatibility_declaration(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized: dict[str, JsonValue] = {}
    for key, item in value.items():
        if key == "run" and isinstance(item, dict):
            normalized[key] = {
                field: _compatibility_declaration(field_value)
                for field, field_value in item.items()
                if field not in {"directory", "resume", "fresh", "checkpoint_interval"}
            }
        else:
            normalized[key] = _compatibility_declaration(item)
    return normalized


__all__ = (
    "OptimizeAnythingFn",
    "PydanticOptimizeAnythingOptimizer",
)
