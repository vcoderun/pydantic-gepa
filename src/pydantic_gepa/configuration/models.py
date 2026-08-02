from __future__ import annotations as _annotations

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator

from ..adapter import ProposalFn
from ..errors import PydanticGEPAError
from ..events import Observer, ObserverPolicy, compose_observers
from ..recorder import GEPAEventBridge
from ..reflection import ReflectionFunction
from ..values import JsonValue, ReprSerializable, SerializableValue

CandidateSelection = Literal["pareto", "current_best", "epsilon_greedy", "top_k_pareto"]
Frontier = Literal["instance", "objective", "hybrid", "cartesian"]
BatchSampler = Literal["epoch_shuffled"]
ComponentSelection = Literal["round_robin", "all"]
ValidationEvaluation = Literal["full_eval"]
Acceptance = Literal["strict_improvement", "improvement_or_equal"]
ResumeMode = Literal["never", "if_exists", "required"]
OpaqueSetting: TypeAlias = SkipValidation[ReprSerializable]
PromptTemplate: TypeAlias = str | dict[str, str]


class ConfigurationError(PydanticGEPAError):
    """Raised when typed GEPA settings conflict or cannot be migrated."""


class ReflectionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    model: str | ReflectionFunction | None = None
    model_kwargs: dict[str, JsonValue] = Field(default_factory=dict)
    minibatch_size: int | None = Field(default=None, gt=0)
    perfect_score: float = 1.0
    skip_perfect_score: bool = True
    prompt_template: PromptTemplate | None = None
    proposer: ProposalFn | None = None

    @model_validator(mode="after")
    def validate_model_kwargs(self) -> ReflectionConfig:
        if self.model_kwargs and not isinstance(self.model, str):
            raise ValueError("reflection.model_kwargs require a string model identifier.")
        return self


class SelectionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    candidate: CandidateSelection | OpaqueSetting = "pareto"
    frontier: Frontier = "instance"
    component: ComponentSelection | OpaqueSetting = "round_robin"
    batch_sampler: BatchSampler | OpaqueSetting = "epoch_shuffled"
    validation: ValidationEvaluation | OpaqueSetting | None = None
    acceptance: Acceptance | OpaqueSetting = "strict_improvement"


class MergeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    max_invocations: int = Field(default=5, ge=0)
    validation_overlap_floor: int = Field(default=5, ge=0)

    @model_validator(mode="after")
    def validate_limits(self) -> MergeConfig:
        if self.enabled and self.max_invocations == 0:
            raise ValueError("merge.max_invocations must be positive when merge is enabled.")
        return self


class BudgetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    max_metric_calls: int | None = Field(default=50, gt=0)
    max_reflection_cost: float | None = Field(default=None, gt=0)
    stop: tuple[OpaqueSetting, ...] = ()

    @model_validator(mode="after")
    def validate_stop_condition(self) -> BudgetConfig:
        if self.max_metric_calls is None and self.max_reflection_cost is None and not self.stop:
            raise ValueError("At least one metric, reflection, or callback budget is required.")
        return self


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(default="run", min_length=1)
    directory: Path | None = None
    resume: ResumeMode = "never"
    fresh: bool = False
    checkpoint_interval: int = Field(default=1, gt=0)
    compatibility: dict[str, str] = Field(default_factory=dict)
    seed: int = 0
    use_cloudpickle: bool = False
    cache_evaluations: bool = False
    raise_on_exception: bool = True

    @model_validator(mode="after")
    def validate_lifecycle(self) -> RunConfig:
        if self.fresh and self.resume != "never":
            raise ValueError("run.fresh cannot be combined with a resume mode.")
        if self.directory is None and (self.fresh or self.resume != "never"):
            raise ValueError("run.directory is required for fresh or resumed runs.")
        return self


class TrackingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    logger: OpaqueSetting | None = None
    backend_callbacks: tuple[OpaqueSetting, ...] = ()
    observers: tuple[Observer, ...] = ()
    observer_errors: ObserverPolicy = "raise"
    track_best_outputs: bool = True
    key_prefix: str = ""
    use_wandb: bool = False
    wandb_api_key: str | None = None
    wandb_init: dict[str, JsonValue] = Field(default_factory=dict)
    wandb_attach_existing: bool = False
    use_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_attach_existing: bool = False

    @model_validator(mode="after")
    def validate_tracking(self) -> TrackingConfig:
        if self.wandb_attach_existing and not self.use_wandb:
            raise ValueError("wandb_attach_existing requires use_wandb=True.")
        if self.mlflow_attach_existing and not self.use_mlflow:
            raise ValueError("mlflow_attach_existing requires use_mlflow=True.")
        return self


class ProgressConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    display_bar: bool = False


class EvaluationSetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    allow_same_train_validation: bool = False


class GEPAConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    merge: MergeConfig = Field(default_factory=MergeConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    run: RunConfig = Field(default_factory=RunConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    progress: ProgressConfig = Field(default_factory=ProgressConfig)
    evaluation_sets: EvaluationSetConfig = Field(default_factory=EvaluationSetConfig)

    def to_backend_kwargs(self) -> dict[str, SerializableValue]:
        reflection_model: str | ReflectionFunction | None = self.reflection.model
        stop_callbacks: ReprSerializable | tuple[ReprSerializable, ...] | None
        if not self.budget.stop:
            stop_callbacks = None
        elif len(self.budget.stop) == 1:
            stop_callbacks = self.budget.stop[0]
        else:
            stop_callbacks = self.budget.stop
        callbacks: tuple[ReprSerializable, ...] = self.tracking.backend_callbacks
        if self.tracking.observers:
            callbacks += (
                GEPAEventBridge(
                    run_id=self.run.id,
                    on_event=compose_observers(
                        *self.tracking.observers,
                        on_error=self.tracking.observer_errors,
                    ),
                ),
            )
        return {
            "reflection_lm": reflection_model,
            "reflection_lm_kwargs": self.reflection.model_kwargs or None,
            "candidate_selection_strategy": self.selection.candidate,
            "frontier_type": self.selection.frontier,
            "skip_perfect_score": self.reflection.skip_perfect_score,
            "batch_sampler": self.selection.batch_sampler,
            "reflection_minibatch_size": self.reflection.minibatch_size,
            "perfect_score": self.reflection.perfect_score,
            "reflection_prompt_template": self.reflection.prompt_template,
            "custom_candidate_proposer": self.reflection.proposer,
            "module_selector": self.selection.component,
            "use_merge": self.merge.enabled,
            "max_merge_invocations": self.merge.max_invocations,
            "merge_val_overlap_floor": self.merge.validation_overlap_floor,
            "max_metric_calls": self.budget.max_metric_calls,
            "max_reflection_cost": self.budget.max_reflection_cost,
            "stop_callbacks": stop_callbacks,
            "logger": self.tracking.logger,
            "run_dir": str(self.run.directory) if self.run.directory is not None else None,
            "callbacks": callbacks or None,
            "use_wandb": self.tracking.use_wandb,
            "wandb_api_key": self.tracking.wandb_api_key,
            "wandb_init_kwargs": self.tracking.wandb_init or None,
            "wandb_attach_existing": self.tracking.wandb_attach_existing,
            "use_mlflow": self.tracking.use_mlflow,
            "mlflow_tracking_uri": self.tracking.mlflow_tracking_uri,
            "mlflow_experiment_name": self.tracking.mlflow_experiment_name,
            "mlflow_attach_existing": self.tracking.mlflow_attach_existing,
            "tracking_key_prefix": self.tracking.key_prefix,
            "track_best_outputs": self.tracking.track_best_outputs,
            "display_progress_bar": self.progress.display_bar,
            "use_cloudpickle": self.run.use_cloudpickle,
            "cache_evaluation": self.run.cache_evaluations,
            "seed": self.run.seed,
            "raise_on_exception": self.run.raise_on_exception,
            "val_evaluation_policy": self.selection.validation,
            "acceptance_criterion": self.selection.acceptance,
        }

    @classmethod
    def from_legacy_kwargs(
        cls,
        values: Mapping[str, SerializableValue],
    ) -> Self:
        unknown = set(values) - _LEGACY_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ConfigurationError(
                f"Unsupported GEPA options: {names}. Use typed GEPAConfig fields."
            )
        warnings.warn(
            "Passing GEPA options as keyword arguments is deprecated; use GEPAConfig.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.model_validate(_legacy_config(values))


_LEGACY_KEYS = frozenset(
    {
        "acceptance_criterion",
        "batch_sampler",
        "cache_evaluation",
        "callbacks",
        "candidate_selection_strategy",
        "custom_candidate_proposer",
        "display_progress_bar",
        "frontier_type",
        "logger",
        "max_merge_invocations",
        "max_metric_calls",
        "max_reflection_cost",
        "merge_val_overlap_floor",
        "mlflow_attach_existing",
        "mlflow_experiment_name",
        "mlflow_tracking_uri",
        "module_selector",
        "perfect_score",
        "raise_on_exception",
        "reflection_lm",
        "reflection_lm_kwargs",
        "reflection_minibatch_size",
        "reflection_prompt_template",
        "run_dir",
        "seed",
        "skip_perfect_score",
        "stop_callbacks",
        "track_best_outputs",
        "tracking_key_prefix",
        "use_cloudpickle",
        "use_merge",
        "use_mlflow",
        "use_wandb",
        "val_evaluation_policy",
        "wandb_api_key",
        "wandb_attach_existing",
        "wandb_init_kwargs",
    }
)


def _legacy_config(values: Mapping[str, SerializableValue]) -> dict[str, SerializableValue]:
    return {
        "reflection": {
            "model": values.get("reflection_lm"),
            "model_kwargs": values.get("reflection_lm_kwargs") or {},
            "minibatch_size": values.get("reflection_minibatch_size"),
            "perfect_score": values.get("perfect_score", 1.0),
            "skip_perfect_score": values.get("skip_perfect_score", True),
            "prompt_template": values.get("reflection_prompt_template"),
            "proposer": values.get("custom_candidate_proposer"),
        },
        "selection": {
            "candidate": values.get("candidate_selection_strategy", "pareto"),
            "frontier": values.get("frontier_type", "instance"),
            "component": values.get("module_selector", "round_robin"),
            "batch_sampler": values.get("batch_sampler", "epoch_shuffled"),
            "validation": values.get("val_evaluation_policy"),
            "acceptance": values.get("acceptance_criterion", "strict_improvement"),
        },
        "merge": {
            "enabled": values.get("use_merge", False),
            "max_invocations": values.get("max_merge_invocations", 5),
            "validation_overlap_floor": values.get("merge_val_overlap_floor", 5),
        },
        "budget": {
            "max_metric_calls": values.get("max_metric_calls", 50),
            "max_reflection_cost": values.get("max_reflection_cost"),
            "stop": _legacy_sequence(values.get("stop_callbacks")),
        },
        "run": {
            "directory": values.get("run_dir"),
            "seed": values.get("seed", 0),
            "use_cloudpickle": values.get("use_cloudpickle", False),
            "cache_evaluations": values.get("cache_evaluation", False),
            "raise_on_exception": values.get("raise_on_exception", True),
        },
        "tracking": {
            "logger": values.get("logger"),
            "backend_callbacks": _legacy_sequence(values.get("callbacks")),
            "track_best_outputs": values.get("track_best_outputs", True),
            "key_prefix": values.get("tracking_key_prefix", ""),
            "use_wandb": values.get("use_wandb", False),
            "wandb_api_key": values.get("wandb_api_key"),
            "wandb_init": values.get("wandb_init_kwargs") or {},
            "wandb_attach_existing": values.get("wandb_attach_existing", False),
            "use_mlflow": values.get("use_mlflow", False),
            "mlflow_tracking_uri": values.get("mlflow_tracking_uri"),
            "mlflow_experiment_name": values.get("mlflow_experiment_name"),
            "mlflow_attach_existing": values.get("mlflow_attach_existing", False),
        },
        "progress": {"display_bar": values.get("display_progress_bar", False)},
    }


def _legacy_sequence(value: SerializableValue | None) -> tuple[SerializableValue, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple | list):
        return tuple(value)
    return (value,)


__all__ = (
    "Acceptance",
    "BatchSampler",
    "BudgetConfig",
    "CandidateSelection",
    "ComponentSelection",
    "ConfigurationError",
    "EvaluationSetConfig",
    "Frontier",
    "GEPAConfig",
    "MergeConfig",
    "ProgressConfig",
    "PromptTemplate",
    "ReflectionConfig",
    "RunConfig",
    "ResumeMode",
    "SelectionConfig",
    "TrackingConfig",
    "ValidationEvaluation",
)
