from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeAlias, TypeVar, cast

from pydantic import BaseModel, ConfigDict

from ..adapter import PydanticGEPAAdapter
from ..asi import PydanticEvalTrajectory, report_case_record
from ..candidates import Candidate
from ..compat import EvaluationBatch
from ..configuration import BudgetConfig, ConfigurationError, EvaluationSetConfig, GEPAConfig
from ..errors import EvaluationHarnessError, OptimizationDependencyError
from ..optimizer import GEPAResult, result_from_gepa
from ..reflection import ReflectionFunction
from ..results import PydanticGEPAResult
from ..state import CompatibilityFingerprint, FileRunStore, RunState, content_fingerprint
from ..values import JsonValue, SerializableValue

CaseT = TypeVar("CaseT")
RolloutOutputT = TypeVar("RolloutOutputT")
EvaluatorT = TypeVar("EvaluatorT")

SideInfo: TypeAlias = dict[str, JsonValue]
OptimizeAnythingFn = Callable[..., GEPAResult]


class OptimizationStateView(Protocol):
    @property
    def best_example_evals(self) -> Sequence[Mapping[str, JsonValue]]: ...


class EngineConfigView(Protocol):
    @property
    def max_metric_calls(self) -> int | None: ...


class GEPAConfigView(Protocol):
    @property
    def engine(self) -> EngineConfigView: ...


GEPAConfigFactory: TypeAlias = Callable[..., GEPAConfigView]


@dataclass(frozen=True)
class LocalEngineConfig:
    max_metric_calls: int | None = None
    max_reflection_cost: float | None = None
    run_dir: str | None = None


@dataclass(frozen=True)
class LocalReflectionConfig:
    reflection_lm: str | ReflectionFunction | None = None


@dataclass(frozen=True)
class LocalGEPAConfig:
    engine: LocalEngineConfig
    reflection: LocalReflectionConfig


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
        return self.adapter.make_reflective_dataset(candidate, eval_batch, components_to_update)

    def component_names(self) -> list[str]:
        return self.adapter.component_names()

    def evaluator(
        self,
        candidate: Mapping[str, str],
        *,
        example: CaseT,
        opt_state: OptimizationStateView | None = None,
    ) -> tuple[float, SideInfo]:
        del opt_state
        active_candidate = self.normalize_candidate(candidate)
        eval_batch = self.adapter.evaluate([example], active_candidate, capture_traces=True)
        if len(eval_batch.scores) != 1:
            raise EvaluationHarnessError(
                "optimize_anything backend expected exactly one evaluation result per example."
            )
        score = eval_batch.scores[0]
        objective_scores = _first_objective_scores(eval_batch)
        return score, _build_side_info(
            adapter=self.adapter,
            candidate=active_candidate,
            eval_batch=eval_batch,
            score=score,
            objective_scores=objective_scores,
        )

    def normalize_candidate(self, candidate: Mapping[str, str]) -> dict[str, str]:
        return self.adapter.normalize_candidate(candidate)


class PydanticOptimizeAnythingOptimizer(BaseModel, Generic[CaseT, RolloutOutputT, EvaluatorT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter: PydanticOptimizeAnythingAdapter[CaseT, RolloutOutputT, EvaluatorT]
    initial_candidate: Candidate
    optimization_objective: str | None = None
    background: str | None = None
    optimize_fn: OptimizeAnythingFn | None = None

    def optimize(
        self,
        *,
        trainset: Sequence[CaseT],
        valset: Sequence[CaseT] | None = None,
        initial_candidate: Candidate | None = None,
        config: GEPAConfig | None = None,
        max_metric_calls: int | None = None,
        allow_same_train_val: bool | None = None,
        objective: str | None = None,
        background: str | None = None,
    ) -> PydanticGEPAResult:
        if config is not None and (
            max_metric_calls is not None or allow_same_train_val is not None
        ):
            raise ConfigurationError(
                "GEPAConfig cannot be combined with max_metric_calls or allow_same_train_val."
            )
        if objective is not None and not isinstance(objective, str):
            raise TypeError("objective override must be a string.")
        if background is not None and not isinstance(background, str):
            raise TypeError("background override must be a string.")
        active_config = config or GEPAConfig(
            budget=BudgetConfig(max_metric_calls=max_metric_calls or 50),
            evaluation_sets=EvaluationSetConfig(
                allow_same_train_validation=bool(allow_same_train_val)
            ),
        )
        active_max_metric_calls = active_config.budget.max_metric_calls
        allow_reuse = active_config.evaluation_sets.allow_same_train_validation
        active_valset = list(trainset if valset is None else valset)
        if valset is None and not allow_reuse:
            raise OptimizationDependencyError(
                "valset is required unless allow_same_train_val=True."
            )

        seed_candidate = initial_candidate or self.initial_candidate
        store: FileRunStore | None = None
        run_state: RunState | None = None
        run_dir: str | None = None
        if active_config.run.directory is not None:
            store = FileRunStore(
                active_config.run.directory,
                run_id=active_config.run.id,
                resume=active_config.run.resume,
                fresh=active_config.run.fresh,
            )
            fingerprint = CompatibilityFingerprint.from_dimensions(
                {
                    "backend": "optimize_anything",
                    "candidate_schema": content_fingerprint(sorted(seed_candidate.values)),
                    "trainset": content_fingerprint(trainset),
                    "valset": content_fingerprint(active_valset),
                    "budget": str(active_max_metric_calls),
                    **active_config.run.compatibility,
                }
            )
            run_state = store.prepare(
                fingerprint=fingerprint,
                initial_candidate=seed_candidate,
            )
            completed = store.load_result(PydanticGEPAResult)
            if run_state.status == "completed" and completed is not None:
                return completed
            run_dir = str(store.backend_directory)

        optimize_fn = self.optimize_fn
        gepa_config_factory: GEPAConfigFactory | None = None
        if optimize_fn is None:
            optimize_fn, gepa_config_factory = _load_gepa_optimize_anything()

        try:
            raw_result = optimize_fn(
                seed_candidate=seed_candidate.to_gepa_dict(),
                evaluator=self.adapter.evaluator,
                dataset=list(trainset),
                valset=active_valset,
                objective=objective if objective is not None else self.optimization_objective,
                background=background if background is not None else self.background,
                config=_build_optimize_anything_config(
                    config=active_config,
                    run_dir=run_dir,
                    gepa_config_factory=gepa_config_factory,
                ),
            )
            result = result_from_gepa(
                raw_result,
                backend="optimize_anything",
                run_id=active_config.run.id,
                budget_limit=active_max_metric_calls,
            )
            result = result.normalize_candidates(self.adapter.normalize_candidate)
        except Exception as exc:
            if store is not None and run_state is not None:
                store.checkpoint(
                    run_state.model_copy(update={"status": "failed", "error": str(exc)})
                )
            raise
        if store is not None and run_state is not None:
            store.write_candidate(result.best_candidate)
            store.write_result(result)
            store.checkpoint(
                run_state.model_copy(
                    update={
                        "status": "completed",
                        "accepted_candidate": result.best_candidate,
                        "metric_calls": result.total_metric_calls or 0,
                        "backend_checkpoint": run_dir,
                    }
                )
            )
        return result


def _build_side_info(
    *,
    adapter: PydanticGEPAAdapter[CaseT, RolloutOutputT, EvaluatorT],
    candidate: dict[str, str],
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT | None],
    score: float,
    objective_scores: dict[str, float] | None,
) -> SideInfo:
    side_info: SideInfo = {"scores": {adapter.objective.score_key: score}}
    trajectories = eval_batch.trajectories
    if trajectories is not None and len(trajectories) == 1:
        report_case = trajectories[0].report_case
        record = report_case_record(
            report_case,
            score=score,
            include_case_metadata=adapter.asi_builder.include_case_metadata,
            include_expected_output=adapter.asi_builder.include_expected_output,
            include_scores=adapter.asi_builder.include_scores,
            include_assertions=adapter.asi_builder.include_assertions,
            include_errors=adapter.asi_builder.include_errors,
        )
        side_info.update(_record_side_info(record))

    if objective_scores:
        side_info["objective_scores"] = {
            name: float(value) for name, value in objective_scores.items()
        }

    reflective_dataset = adapter.make_reflective_dataset(
        candidate,
        eval_batch,
        adapter.component_names(),
    )
    for component, records in reflective_dataset.items():
        if records:
            side_info[f"{component}_specific_info"] = {
                "examples": [_record_json(record) for record in records]
            }
    return side_info


def _record_side_info(record: Mapping[str, SerializableValue]) -> SideInfo:
    side_info: SideInfo = {}
    for key, value in record.items():
        if key == "scores":
            side_info["observed_scores"] = cast("JsonValue", value)
        else:
            side_info[key] = cast("JsonValue", value)
    return side_info


def _record_json(record: Mapping[str, SerializableValue]) -> dict[str, JsonValue]:
    return {name: cast("JsonValue", value) for name, value in record.items()}


def _first_objective_scores(
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT | None],
) -> dict[str, float] | None:
    objective_scores = eval_batch.objective_scores
    if objective_scores is None or len(objective_scores) != 1:
        return None
    return objective_scores[0]


def _build_optimize_anything_config(
    *,
    config: GEPAConfig,
    run_dir: str | None,
    gepa_config_factory: GEPAConfigFactory | None,
) -> GEPAConfigView:
    if gepa_config_factory is None:
        return LocalGEPAConfig(
            engine=LocalEngineConfig(
                max_metric_calls=config.budget.max_metric_calls,
                max_reflection_cost=config.budget.max_reflection_cost,
                run_dir=run_dir,
            ),
            reflection=LocalReflectionConfig(reflection_lm=config.reflection.model),
        )
    backend = config.to_backend_kwargs()
    unsupported: list[str] = []
    if config.budget.max_reflection_cost is not None:
        unsupported.append("budget.max_reflection_cost")
    if config.reflection.model_kwargs:
        unsupported.append("reflection.model_kwargs")
    if config.selection.acceptance != "strict_improvement":
        unsupported.append("selection.acceptance")
    if config.tracking.backend_callbacks or config.tracking.observers:
        unsupported.append("tracking callbacks")
    if config.tracking.wandb_attach_existing:
        unsupported.append("tracking.wandb_attach_existing")
    if config.tracking.mlflow_attach_existing:
        unsupported.append("tracking.mlflow_attach_existing")
    if config.tracking.key_prefix:
        unsupported.append("tracking.key_prefix")
    if unsupported:
        names = ", ".join(unsupported)
        raise ConfigurationError(
            f"Installed GEPA Optimize Anything backend does not support configured options: {names}."
        )
    return gepa_config_factory(
        engine={
            "run_dir": run_dir,
            "seed": config.run.seed,
            "display_progress_bar": config.progress.display_bar,
            "raise_on_exception": config.run.raise_on_exception,
            "use_cloudpickle": config.run.use_cloudpickle,
            "track_best_outputs": config.tracking.track_best_outputs,
            "max_metric_calls": config.budget.max_metric_calls,
            "val_evaluation_policy": config.selection.validation or "full_eval",
            "candidate_selection_strategy": config.selection.candidate,
            "frontier_type": config.selection.frontier,
            "cache_evaluation": config.run.cache_evaluations,
        },
        reflection={
            "skip_perfect_score": config.reflection.skip_perfect_score,
            "perfect_score": config.reflection.perfect_score,
            "batch_sampler": config.selection.batch_sampler,
            "reflection_minibatch_size": config.reflection.minibatch_size,
            "module_selector": config.selection.component,
            "reflection_lm": config.reflection.model,
            "reflection_prompt_template": config.reflection.prompt_template,
            "custom_candidate_proposer": config.reflection.proposer,
        },
        merge=(
            {
                "max_merge_invocations": config.merge.max_invocations,
                "merge_val_overlap_floor": config.merge.validation_overlap_floor,
            }
            if config.merge.enabled
            else None
        ),
        tracking={
            "logger": config.tracking.logger,
            "use_wandb": config.tracking.use_wandb,
            "wandb_api_key": config.tracking.wandb_api_key,
            "wandb_init_kwargs": config.tracking.wandb_init or None,
            "use_mlflow": config.tracking.use_mlflow,
            "mlflow_tracking_uri": config.tracking.mlflow_tracking_uri,
            "mlflow_experiment_name": config.tracking.mlflow_experiment_name,
        },
        stop_callbacks=backend["stop_callbacks"],
    )


def _load_gepa_optimize_anything() -> tuple[OptimizeAnythingFn, GEPAConfigFactory]:
    try:
        from gepa.optimize_anything import GEPAConfig, optimize_anything
    except ImportError as exc:  # pragma: no cover - depends on optional integration
        raise OptimizationDependencyError(
            "GEPA is not installed. Install pydantic-gepa[integrations]."
        ) from exc
    return (
        cast("OptimizeAnythingFn", optimize_anything),
        cast("GEPAConfigFactory", GEPAConfig),
    )


__all__ = (
    "OptimizeAnythingFn",
    "PydanticOptimizeAnythingAdapter",
    "PydanticOptimizeAnythingOptimizer",
    "SideInfo",
)
