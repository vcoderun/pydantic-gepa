from __future__ import annotations as _annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable
from uuid import uuid4

from ...candidates import Candidate
from ...configuration import ConfigurationError
from ...errors import OptimizationDependencyError
from ...events import (
    BudgetSnapshot,
    CandidateNormalized,
    StageCompleted,
    StageFailed,
    StageStarted,
    _event_scope,
    _EventDispatcher,
)
from ...recorder import GEPAEventBridge
from ...results import AdaptiveSliceSummary
from .adapter import CandidateCodec, CandidateValue, EvaluationOutput, OptimizationState
from .models import (
    AdaptiveSequential,
    BestOf,
    CustomEngine,
    Engine,
    EngineResult,
    EvaluationServer,
    OptimizationTask,
    Parallel,
    Sequential,
    Single,
    Vote,
)

CaseT = TypeVar("CaseT")
CaseT_contra = TypeVar("CaseT_contra", contravariant=True)

if TYPE_CHECKING:
    from gepa.oa.config import OptimizeAnythingConfig as GEPAOptimizeAnythingConfig
    from gepa.oa.engine import Engine as GEPAEngine
    from gepa.oa.eval_server import EvalServer as GEPAEvalServer


class Evaluator(Protocol[CaseT_contra]):
    def __call__(
        self,
        candidate: CandidateValue,
        example: CaseT_contra,
        opt_state: OptimizationState | None = None,
    ) -> EvaluationOutput: ...


class BatchEvaluator(Protocol[CaseT_contra]):
    def __call__(
        self,
        pairs: Sequence[tuple[CandidateValue, CaseT_contra]],
        *,
        opt_states: Sequence[OptimizationState | None] | None = None,
    ) -> list[EvaluationOutput]: ...


class ServerSettings(Protocol):
    @property
    def max_evals(self) -> int | None: ...

    @property
    def max_concurrency(self) -> int: ...

    @property
    def output_dir(self) -> str | Path | None: ...


@dataclass(frozen=True, slots=True)
class AdaptiveServerSettings:
    max_evals: int | None
    max_concurrency: int
    output_dir: Path | None


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    results: tuple[EngineResult, ...]
    engine_indices: tuple[int, ...]
    selected_index: int | None = None
    selection_scores: tuple[float, ...] | None = None
    evaluation_calls: tuple[int, ...] | None = None
    selection_evaluation_calls: int = 0
    adaptive_schedule: tuple[AdaptiveSliceSummary, ...] = ()
    adaptive_switches: int = 0
    stop_reason: str | None = None
    total_cost: float | None = None


@dataclass(frozen=True, slots=True)
class EngineInvocation:
    engine_index: int
    invocation_index: int
    execution_id: str
    parent_execution_id: str | None
    branch_id: str
    input_candidate: Candidate
    started_at: float


@runtime_checkable
class OptimizeAnythingFn(Protocol[CaseT]):
    def __call__(
        self,
        seed_candidate: str | dict[str, str] | None = None,
        *,
        evaluator: Evaluator[CaseT] | None = None,
        batch_evaluator: BatchEvaluator[CaseT] | None = None,
        dataset: list[CaseT] | None = None,
        valset: list[CaseT] | None = None,
        objective: str | None = None,
        background: str | None = None,
        test_set: list[CaseT] | None = None,
        config: GEPAOptimizeAnythingConfig | None = None,
    ) -> EngineResult: ...


class ExecutionTracker:
    def __init__(
        self,
        *,
        dispatcher: _EventDispatcher,
        codec: CandidateCodec,
        engines: Sequence[Engine],
        composition: str,
        pipeline_id: str,
        step_id: str,
        parent_execution_id: str | None,
    ) -> None:
        self.dispatcher = dispatcher
        self.codec = codec
        self.engines = tuple(engines)
        self.composition = composition
        self.pipeline_id = pipeline_id
        self.step_id = step_id
        self.parent_execution_id = parent_execution_id
        self._lock = RLock()
        self._counts = [0] * len(engines)
        self._active: dict[int, EngineInvocation] = {}
        self._completed: dict[int, EngineInvocation] = {}
        self._completed_results: list[tuple[EngineInvocation, EngineResult]] = []
        self._pending: dict[int, list[EngineInvocation]] = {}
        self._candidates: dict[str, Candidate] = {codec.seed.fingerprint(): codec.seed}
        self._producers: dict[str, str] = {}

    def start(self, engine_index: int, seed: CandidateValue | None) -> EngineInvocation:
        with self._lock:
            input_candidate = self._input_candidate(seed)
            invocation_index = self._counts[engine_index]
            self._counts[engine_index] += 1
            branch_id = f"branch-{engine_index}"
            if invocation_index:
                branch_id = f"{branch_id}-run-{invocation_index}"
            execution_id = f"{self.pipeline_id}:{self.step_id}:{branch_id}"
            input_fingerprint = input_candidate.fingerprint()
            invocation = EngineInvocation(
                engine_index=engine_index,
                invocation_index=invocation_index,
                execution_id=execution_id,
                parent_execution_id=self._producers.get(
                    input_fingerprint,
                    self.parent_execution_id,
                ),
                branch_id=branch_id,
                input_candidate=input_candidate,
                started_at=monotonic(),
            )
            self._active[engine_index] = invocation
        self.dispatcher.emit(
            StageStarted(
                run_id=self.dispatcher.run_id,
                engine=self.engines[engine_index].name,
                composition=self.composition,
                pipeline_id=self.pipeline_id,
                step_id=self.step_id,
                branch_id=branch_id,
                engine_execution_id=execution_id,
                stage_id=execution_id,
                stage_kind="engine",
                candidate_id=input_candidate.id or input_fingerprint,
                parent_execution_id=invocation.parent_execution_id,
            )
        )
        return invocation

    def complete(self, invocation: EngineInvocation, result: EngineResult) -> EngineInvocation:
        engine = self.engines[invocation.engine_index]
        optimizer_cost = _nonnegative_float(result.metadata.get("adapter_cost"))
        total_cost = _nonnegative_float(result.metadata.get("total_cost"))
        output_candidate = Candidate(
            values=self.codec.decode(result.best_candidate),
            id=invocation.execution_id,
            parent_id=(invocation.input_candidate.id or invocation.input_candidate.fingerprint()),
            generation=(invocation.input_candidate.generation or 0) + 1,
            metadata={"engine": engine.name},
        )
        with self._lock:
            output_fingerprint = output_candidate.fingerprint()
            self._completed[id(result)] = invocation
            self._completed_results.append((invocation, result))
            self._candidates[output_fingerprint] = output_candidate
            self._producers[output_fingerprint] = invocation.execution_id
            self._active.pop(invocation.engine_index, None)
        self.dispatcher.emit(
            CandidateNormalized(
                run_id=self.dispatcher.run_id,
                engine=engine.name,
                composition=self.composition,
                pipeline_id=self.pipeline_id,
                step_id=self.step_id,
                branch_id=invocation.branch_id,
                engine_execution_id=invocation.execution_id,
                stage_id=invocation.execution_id,
                candidate_id=output_candidate.id or output_fingerprint,
                parent_ids=(
                    invocation.input_candidate.id or invocation.input_candidate.fingerprint(),
                ),
                candidate=output_candidate,
            )
        )
        self.dispatcher.emit(
            StageCompleted(
                run_id=self.dispatcher.run_id,
                engine=engine.name,
                composition=self.composition,
                pipeline_id=self.pipeline_id,
                step_id=self.step_id,
                branch_id=invocation.branch_id,
                engine_execution_id=invocation.execution_id,
                stage_id=invocation.execution_id,
                stage_kind="engine",
                candidate_id=output_candidate.id or output_fingerprint,
                parent_execution_id=invocation.parent_execution_id,
                score=float(result.best_score),
                budget=BudgetSnapshot(
                    evaluation_calls=result.total_evals,
                    evaluation_call_limit=engine.max_evals,
                    optimizer_cost=optimizer_cost,
                    optimizer_cost_limit=engine.max_token_cost,
                    evaluation_cost=(
                        None
                        if total_cost is None
                        else max(0.0, total_cost - (optimizer_cost or 0.0))
                    ),
                    total_cost=total_cost,
                ),
                metadata={"duration_seconds": max(0.0, monotonic() - invocation.started_at)},
            )
        )
        return invocation

    def fail(self, invocation: EngineInvocation, error: BaseException) -> None:
        with self._lock:
            self._active.pop(invocation.engine_index, None)
        self.dispatcher.emit(
            StageFailed(
                run_id=self.dispatcher.run_id,
                engine=self.engines[invocation.engine_index].name,
                composition=self.composition,
                pipeline_id=self.pipeline_id,
                step_id=self.step_id,
                branch_id=invocation.branch_id,
                engine_execution_id=invocation.execution_id,
                stage_id=invocation.execution_id,
                stage_kind="engine",
                candidate_id=(
                    invocation.input_candidate.id or invocation.input_candidate.fingerprint()
                ),
                parent_execution_id=invocation.parent_execution_id,
                error_type=type(error).__name__,
                message=str(error),
            )
        )

    def defer_budget_completion(self, invocation: EngineInvocation) -> None:
        with self._lock:
            self._active.pop(invocation.engine_index, None)
            self._pending.setdefault(invocation.engine_index, []).append(invocation)

    def resolve(self, result: EngineResult, *, engine_index: int) -> EngineInvocation:
        with self._lock:
            completed = self._completed.get(id(result))
            pending = self._pending.get(engine_index)
            invocation = None if not pending else pending.pop(0)
        if completed is not None:
            return completed
        if invocation is None:
            invocation = self.start(engine_index, self.codec.encode_seed())
        return self.complete(invocation, result)

    def completed_results(self) -> tuple[tuple[EngineInvocation, EngineResult], ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._completed_results,
                    key=lambda item: (item[0].engine_index, item[0].invocation_index),
                )
            )

    def is_active(self, engine_index: int) -> bool:
        with self._lock:
            return engine_index in self._active

    @contextmanager
    def evaluation_scope(self, engine_index: int) -> Iterator[None]:
        with self._lock:
            invocation = self._active.get(engine_index)
        if invocation is None:
            yield
            return
        with _event_scope(
            self.dispatcher,
            stage_id=invocation.execution_id,
            parent_execution_id=invocation.parent_execution_id,
            engine=self.engines[engine_index].name,
            composition=self.composition,
            pipeline_id=self.pipeline_id,
            step_id=self.step_id,
            branch_id=invocation.branch_id,
            engine_execution_id=invocation.execution_id,
            stage_kind="engine",
        ):
            yield

    @contextmanager
    def composition_scope(self) -> Iterator[None]:
        with _event_scope(
            self.dispatcher,
            stage_id=self.step_id,
            composition=self.composition,
            pipeline_id=self.pipeline_id,
            step_id=self.step_id,
            stage_kind="composition",
        ):
            yield

    def _input_candidate(self, seed: CandidateValue | None) -> Candidate:
        if seed is None:
            return self.codec.seed
        values = self.codec.decode(seed)
        fingerprint = Candidate(values=values).fingerprint()
        candidate = self._candidates.get(fingerprint)
        return Candidate(values=values) if candidate is None else candidate


class ObservedEngine:
    def __init__(
        self,
        engine: CustomEngine,
        *,
        tracker: ExecutionTracker,
        engine_index: int,
        name: str,
    ) -> None:
        self.name = name
        self.engine = engine
        self.tracker = tracker
        self.engine_index = engine_index

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        invocation = self.tracker.start(self.engine_index, task.seed_candidate)
        try:
            with self.tracker.evaluation_scope(self.engine_index):
                result = self.engine.run(task, server)
            return_result = result
        except BaseException as exc:
            if _is_budget_exhausted(exc):
                self.tracker.defer_budget_completion(invocation)
            else:
                self.tracker.fail(invocation, exc)
            raise
        self.tracker.complete(invocation, return_result)
        return return_result

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        self.engine.process_result(result, output_dir)


@dataclass(frozen=True, slots=True)
class BackendRuntime:
    config: GEPAOptimizeAnythingConfig
    engine_index: int
    tracker: ExecutionTracker

    @contextmanager
    def scope(self) -> Iterator[None]:
        with self.tracker.evaluation_scope(self.engine_index):
            yield


def configure_engine(
    engine: Engine,
    *,
    engine_index: int,
    name: str,
    run_dir: Path | None,
    callback: GEPAEventBridge,
    tracker: ExecutionTracker,
    observe_engine: bool,
) -> BackendRuntime:
    try:
        from gepa.optimize_anything import OptimizeAnythingConfig as GEPAOptimizeAnythingConfig
    except ImportError as exc:
        raise OptimizationDependencyError(
            "Installed GEPA does not provide the Optimize Anything engine API. "
            "Install pydantic-gepa[optimize-anything]."
        ) from exc
    engine_value: str | GEPAEngine = engine.kind
    settings: dict[str, Any]
    if engine.kind == "gepa":
        config = engine.gepa_config
        if config is None:
            raise RuntimeError("Validated GEPA engine has no GEPAConfig.")
        backend = config.to_backend_kwargs()
        settings = {
            "engine": {
                "seed": config.run.seed,
                "display_progress_bar": config.progress.display_bar,
                "raise_on_exception": config.run.raise_on_exception,
                "use_cloudpickle": config.run.use_cloudpickle,
                "track_best_outputs": config.tracking.track_best_outputs,
                "max_metric_calls": engine.max_evals,
                "max_reflection_cost": engine.max_token_cost,
                "stop_at_score": engine.stop_at_score,
                "val_evaluation_policy": config.selection.validation or "full_eval",
                "candidate_selection_strategy": config.selection.candidate,
                "frontier_type": config.selection.frontier,
                "acceptance_criterion": config.selection.acceptance,
                "parallel": engine.max_concurrency > 1,
                "max_workers": engine.max_concurrency,
                "cache_evaluation": config.run.cache_evaluations,
            },
            "reflection": {
                "skip_perfect_score": config.reflection.skip_perfect_score,
                "perfect_score": config.reflection.perfect_score,
                "batch_sampler": config.selection.batch_sampler,
                "reflection_minibatch_size": config.reflection.minibatch_size,
                "module_selector": config.selection.component,
                "reflection_lm": config.reflection.model,
                "reflection_lm_kwargs": config.reflection.model_kwargs or None,
                "reflection_prompt_template": config.reflection.prompt_template,
                "custom_candidate_proposer": config.reflection.proposer,
            },
            "merge": (
                {
                    "max_merge_invocations": config.merge.max_invocations,
                    "merge_val_overlap_floor": config.merge.validation_overlap_floor,
                }
                if config.merge.enabled
                else None
            ),
            "tracking": {
                "logger": config.tracking.logger,
                "use_wandb": config.tracking.use_wandb,
                "wandb_api_key": config.tracking.wandb_api_key,
                "wandb_init_kwargs": config.tracking.wandb_init or None,
                "wandb_attach_existing": config.tracking.wandb_attach_existing,
                "use_mlflow": config.tracking.use_mlflow,
                "mlflow_tracking_uri": config.tracking.mlflow_tracking_uri,
                "mlflow_experiment_name": config.tracking.mlflow_experiment_name,
                "mlflow_attach_existing": config.tracking.mlflow_attach_existing,
                "key_prefix": config.tracking.key_prefix,
            },
            "stop_callbacks": backend["stop_callbacks"],
            "callbacks": (*config.tracking.backend_callbacks, callback),
        }
    elif engine.kind == "autoresearch":
        options = engine.autoresearch_options
        if options is None:
            raise RuntimeError("Validated AutoResearch engine has no options.")
        settings = options.model_dump(mode="python")
        settings["handoffs"] = list(options.handoffs) or None
    elif engine.kind == "meta_harness":
        options = engine.meta_harness_options
        if options is None:
            raise RuntimeError("Validated Meta-Harness engine has no options.")
        settings = options.model_dump(mode="python")
        settings["max_candidates_per_iter"] = settings.pop("max_candidates_per_iteration")
    elif engine.kind == "best_of_n":
        options = engine.best_of_n_options
        if options is None:
            raise RuntimeError("Validated Best-of-N engine has no options.")
        settings = options.model_dump(mode="python")
        settings["max_n"] = settings.pop("max_samples")
        settings["lm_kwargs"] = settings.pop("model_options")
    else:
        if engine.custom_instance is None:
            raise RuntimeError("Validated custom engine has no instance.")
        engine_value = cast("GEPAEngine", engine.custom_instance)
        settings = {}
    config = GEPAOptimizeAnythingConfig(
        engine=engine_value,
        name=name,
        max_evals=engine.max_evals,
        max_token_cost=engine.max_token_cost,
        max_concurrency=engine.max_concurrency,
        output_dir=(
            str(engine.output_dir)
            if engine.output_dir is not None
            else None
            if run_dir is None
            else str(run_dir / "evaluations")
        ),
        run_dir=str(engine.run_dir or run_dir) if engine.run_dir or run_dir else None,
        stop_at_score=engine.stop_at_score,
        sandbox=engine.sandbox,
        engine_config=settings,
    )
    if not observe_engine:
        return BackendRuntime(config=config, engine_index=engine_index, tracker=tracker)
    if isinstance(config.engine, str):
        try:
            from gepa.oa import get_engine_cls
        except ImportError as exc:
            raise OptimizationDependencyError(
                "Installed GEPA does not provide the Optimize Anything engine API. "
                "Install pydantic-gepa[optimize-anything]."
            ) from exc
        implementation = cast("CustomEngine", get_engine_cls(config.engine)(config))
    else:
        implementation = cast("CustomEngine", config.engine)
    config = replace(
        config,
        engine=cast(
            "GEPAEngine",
            ObservedEngine(
                implementation,
                tracker=tracker,
                engine_index=engine_index,
                name=engine.name,
            ),
        ),
        engine_config={},
    )
    return BackendRuntime(config=config, engine_index=engine_index, tracker=tracker)


def execute(
    composition: Single | Sequential | Parallel | BestOf | Vote | AdaptiveSequential,
    *,
    codec: CandidateCodec,
    evaluate: Evaluator[CaseT],
    batch_evaluate: BatchEvaluator[CaseT],
    trainset: list[CaseT],
    valset: list[CaseT],
    objective: str | None,
    background: str | None,
    runtimes: list[BackendRuntime],
    optimize_fn: OptimizeAnythingFn[CaseT] | None,
) -> ExecutionOutcome:
    seed = codec.encode_seed()
    if isinstance(composition, Single):
        if optimize_fn is None:
            try:
                from gepa.optimize_anything import optimize_anything
            except ImportError as exc:
                raise OptimizationDependencyError(
                    "Installed GEPA does not provide Optimize Anything Omni. "
                    "Install pydantic-gepa[optimize-anything]."
                ) from exc
            runner = cast("OptimizeAnythingFn[CaseT]", optimize_anything)
        else:
            runner = optimize_fn
        runtime = runtimes[0]
        invocation = runtime.tracker.start(0, seed)
        try:
            with runtime.scope():
                raw = runner(
                    seed_candidate=seed,
                    evaluator=_scoped_evaluator(evaluate, runtime),
                    batch_evaluator=_scoped_batch_evaluator(batch_evaluate, runtime),
                    dataset=trainset,
                    valset=valset,
                    objective=objective,
                    background=background,
                    test_set=None,
                    config=runtime.config,
                )
        except BaseException as exc:
            runtime.tracker.fail(invocation, exc)
            raise
        runtime.tracker.complete(invocation, raw)
        return ExecutionOutcome(results=(raw,), engine_indices=(0,), selected_index=0)

    try:
        from gepa.oa.budget import BudgetTracker
        from gepa.oa.ensemble import (
            optimize_adaptive_sequential_with_server,
            optimize_best_of_with_server,
            optimize_parallel_with_server,
            optimize_sequential_with_server,
            optimize_vote_with_server,
        )
        from gepa.oa.eval_server import EvalServer
        from gepa.oa.task import Task
    except ImportError as exc:
        raise OptimizationDependencyError(
            "Installed GEPA does not provide Optimize Anything Omni. "
            "Install pydantic-gepa[optimize-anything]."
        ) from exc
    if optimize_fn is not None:
        raise ConfigurationError("A custom optimize_fn can only execute Single compositions.")
    task = Task(
        name=f"pydantic-gepa-{uuid4().hex[:8]}",
        seed_candidate=seed,
        objective=objective or "",
        background=background or "",
        train_set=trainset,
        val_set=valset,
        test_set=None,
    )

    def server(runtime: BackendRuntime, settings: ServerSettings) -> GEPAEvalServer:
        return EvalServer(
            task,
            _scoped_evaluator(evaluate, runtime),
            BudgetTracker(max_evals=settings.max_evals),
            batch_evaluate=_scoped_batch_evaluator(batch_evaluate, runtime),
            max_concurrency=settings.max_concurrency,
            output_dir=settings.output_dir,
        )

    configs = [runtime.config for runtime in runtimes]
    if isinstance(composition, AdaptiveSequential):
        settings = AdaptiveServerSettings(
            max_evals=composition.max_evals,
            max_concurrency=composition.max_concurrency,
            output_dir=composition.output_dir,
        )
        shared = EvalServer(
            task,
            _adaptive_evaluator(evaluate, runtimes),
            BudgetTracker(max_evals=settings.max_evals),
            batch_evaluate=_adaptive_batch_evaluator(batch_evaluate, runtimes),
            max_concurrency=settings.max_concurrency,
            output_dir=settings.output_dir,
        )
        shared.start()
        try:
            raw = optimize_adaptive_sequential_with_server(
                shared,
                configs,
                plateau_evals=composition.plateau_evals,
                patience=composition.patience,
                min_evals_per_stage=composition.min_evals_per_stage,
                improvement_epsilon=composition.improvement_epsilon,
                cycle=composition.cycle,
                max_switches=composition.max_switches,
            )
        finally:
            shared.stop()
        nested = _raw_results(raw.metadata.get("stage_results")) or [cast("EngineResult", raw)]
        schedule = _adaptive_schedule(raw.metadata.get("adaptive_schedule"), expected=len(nested))
        selected = max(range(len(nested)), key=lambda index: nested[index].best_score)
        switches = raw.metadata.get("adaptive_switches")
        stop_reason = raw.metadata.get("adaptive_stop_reason")
        return ExecutionOutcome(
            results=tuple(nested),
            engine_indices=tuple(item.engine_index for item in schedule),
            selected_index=selected,
            evaluation_calls=tuple(item.evaluation_calls for item in schedule),
            adaptive_schedule=schedule,
            adaptive_switches=(
                switches if isinstance(switches, int) and not isinstance(switches, bool) else 0
            ),
            stop_reason=stop_reason if isinstance(stop_reason, str) else None,
            total_cost=_nonnegative_float(raw.metadata.get("total_cost")),
        )

    servers = [server(runtime, runtime.config) for runtime in runtimes]
    for active in servers:
        active.start()
    try:
        if isinstance(composition, Sequential):
            raw = optimize_sequential_with_server(servers, configs)
        elif isinstance(composition, Parallel):
            raw_list = optimize_parallel_with_server(
                servers,
                configs,
                max_workers=composition.max_workers,
            )
            results = [cast("EngineResult", result) for result in raw_list]
            return ExecutionOutcome(
                results=tuple(results),
                engine_indices=tuple(range(len(results))),
            )
        elif isinstance(composition, BestOf):
            raw = optimize_best_of_with_server(
                servers,
                configs,
                max_workers=composition.max_workers,
            )
        else:
            if not valset:
                raise ConfigurationError("Vote requires at least one validation example.")

            def vote(candidate: CandidateValue) -> EvaluationOutput:
                pairs = [(candidate, example) for example in valset]
                with runtimes[0].tracker.composition_scope():
                    values = batch_evaluate(pairs, opt_states=None)
                score = sum(value[0] for value in values) / len(values)
                return score, {"scores": [value[0] for value in values]}

            raw = optimize_vote_with_server(
                servers,
                configs,
                vote,
                max_workers=composition.max_workers,
            )
    finally:
        for active in servers:
            active.stop()
    nested = _raw_results(raw.metadata.get("all_results")) or [cast("EngineResult", raw)]
    engine_indices = tuple(range(len(nested)))
    completed = runtimes[0].tracker.completed_results()
    if len(completed) > len(nested):
        nested = [result for _invocation, result in completed]
        engine_indices = tuple(invocation.engine_index for invocation, _result in completed)
    if isinstance(composition, Vote):
        raw_scores = raw.metadata.get("vote_scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != len(nested):
            raise ConfigurationError("Vote result omitted aligned selection scores.")
        parsed_scores = tuple(_finite_score(value) for value in raw_scores)
        if any(value is None for value in parsed_scores):
            raise ConfigurationError("Vote result contains a non-finite selection score.")
        vote_scores = tuple(value for value in parsed_scores if value is not None)
        selected = raw.metadata.get("vote_winner_idx")
        return ExecutionOutcome(
            results=tuple(nested),
            engine_indices=engine_indices,
            selected_index=(
                selected if isinstance(selected, int) and not isinstance(selected, bool) else None
            ),
            selection_scores=vote_scores,
            selection_evaluation_calls=len(valset) * len(nested),
        )
    selected = max(range(len(nested)), key=lambda index: nested[index].best_score)
    return ExecutionOutcome(
        results=tuple(nested),
        engine_indices=engine_indices,
        selected_index=selected,
    )


def _scoped_evaluator(
    evaluate: Evaluator[CaseT],
    runtime: BackendRuntime,
) -> Evaluator[CaseT]:
    def scoped(
        candidate: CandidateValue,
        example: CaseT,
        opt_state: OptimizationState | None = None,
    ) -> EvaluationOutput:
        with runtime.scope():
            return evaluate(candidate, example, opt_state)

    return scoped


def _scoped_batch_evaluator(
    evaluate: BatchEvaluator[CaseT],
    runtime: BackendRuntime,
) -> BatchEvaluator[CaseT]:
    def scoped(
        pairs: Sequence[tuple[CandidateValue, CaseT]],
        *,
        opt_states: Sequence[OptimizationState | None] | None = None,
    ) -> list[EvaluationOutput]:
        with runtime.scope():
            return evaluate(pairs, opt_states=opt_states)

    return scoped


def _adaptive_evaluator(
    evaluate: Evaluator[CaseT],
    runtimes: Sequence[BackendRuntime],
) -> Evaluator[CaseT]:
    def scoped(
        candidate: CandidateValue,
        example: CaseT,
        opt_state: OptimizationState | None = None,
    ) -> EvaluationOutput:
        runtime = next(
            (item for item in runtimes if item.tracker.is_active(item.engine_index)),
            None,
        )
        if runtime is None:
            return evaluate(candidate, example, opt_state)
        with runtime.scope():
            return evaluate(candidate, example, opt_state)

    return scoped


def _adaptive_batch_evaluator(
    evaluate: BatchEvaluator[CaseT],
    runtimes: Sequence[BackendRuntime],
) -> BatchEvaluator[CaseT]:
    def scoped(
        pairs: Sequence[tuple[CandidateValue, CaseT]],
        *,
        opt_states: Sequence[OptimizationState | None] | None = None,
    ) -> list[EvaluationOutput]:
        runtime = next(
            (item for item in runtimes if item.tracker.is_active(item.engine_index)),
            None,
        )
        if runtime is None:
            return evaluate(pairs, opt_states=opt_states)
        with runtime.scope():
            return evaluate(pairs, opt_states=opt_states)

    return scoped


def _raw_results(value: Any) -> list[EngineResult]:
    if not isinstance(value, list):
        return []
    try:
        from gepa.oa.engine import Result
    except ImportError:
        return []
    return [cast("EngineResult", item) for item in value if isinstance(item, Result)]


def _adaptive_schedule(value: Any, *, expected: int) -> tuple[AdaptiveSliceSummary, ...]:
    if not isinstance(value, list):
        raise ConfigurationError("Adaptive Optimize Anything result omitted its schedule.")
    schedule: list[AdaptiveSliceSummary] = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ConfigurationError("Adaptive Optimize Anything schedule is malformed.")
        index = entry.get("engine_idx")
        start = entry.get("eval_start")
        end = entry.get("eval_end")
        calls = entry.get("eval_delta")
        engine = entry.get("engine")
        improved = entry.get("improved")
        if not isinstance(index, int) or isinstance(index, bool):
            raise ConfigurationError("Adaptive Optimize Anything schedule has no engine index.")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or not isinstance(calls, int)
            or isinstance(calls, bool)
        ):
            raise ConfigurationError("Adaptive Optimize Anything schedule has invalid eval counts.")
        if not isinstance(engine, str) or not isinstance(improved, bool):
            raise ConfigurationError(
                "Adaptive Optimize Anything schedule is missing stage evidence."
            )
        schedule.append(
            AdaptiveSliceSummary(
                engine_index=index,
                engine=engine,
                evaluation_start=start,
                evaluation_end=end,
                evaluation_calls=calls,
                score_before=_finite_score(entry.get("best_before")),
                score_after=_finite_score(entry.get("best_after")),
                improved=improved,
                optimizer_cost=_nonnegative_float(entry.get("adapter_cost")),
            )
        )
    if len(schedule) != expected:
        raise ConfigurationError(
            "Adaptive Optimize Anything schedule does not align with engine results."
        )
    return tuple(schedule)


def _finite_score(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if isfinite(number):
            return number
    return None


def _nonnegative_float(value: Any) -> float | None:
    number = _finite_score(value)
    return number if number is not None and number >= 0 else None


def _is_budget_exhausted(error: BaseException) -> bool:
    try:
        from gepa.oa.budget import BudgetExhausted
    except ImportError:
        return False
    return isinstance(error, BudgetExhausted)


__all__ = ("OptimizeAnythingFn",)
