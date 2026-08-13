from __future__ import annotations as _annotations

import inspect
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from types import FunctionType, MethodType
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...configuration import GEPAConfig, RunConfig
from ...configuration.models import TrackingConfig
from ...values import JsonValue

CandidateMode = Literal["components", "text"]
EngineKind = Literal["gepa", "autoresearch", "meta_harness", "best_of_n", "custom"]
CompositionKind = Literal[
    "single",
    "sequential",
    "parallel",
    "best_of",
    "vote",
    "adaptive_sequential",
    "pipeline",
]


@runtime_checkable
class CustomEngine(Protocol):
    name: str

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult: ...

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None: ...


class OptimizationTask(Protocol):
    name: str
    seed_candidate: str | dict[str, str] | None
    objective: str
    background: str
    train_set: list[Any] | None
    val_set: list[Any] | None
    test_set: list[Any] | None


class EvaluationServer(Protocol):
    url: str
    task: OptimizationTask

    def evaluate(
        self,
        candidate: str | dict[str, str],
        example: Any | None = None,
    ) -> tuple[float, dict[str, Any]]: ...

    def evaluate_examples(
        self,
        candidate: str | dict[str, str],
        example_ids: list[str] | None = None,
        split: Literal["train", "val", "all"] | None = None,
    ) -> tuple[float, dict[str, Any]]: ...


class EngineResult(Protocol):
    best_candidate: str | dict[str, str]
    best_score: float
    total_evals: int
    eval_log: list[dict[str, Any]]
    metadata: dict[str, Any]


class AutoResearchOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "claude-sonnet-4-6"
    ralph: bool = True
    max_no_eval_seconds: float | None = Field(default=None, gt=0)
    handoffs: tuple[dict[str, JsonValue], ...] = ()
    effort: str | None = None
    max_thinking_tokens: int | None = Field(default=None, gt=0)


class MetaHarnessOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "claude-sonnet-4-6"
    max_iterations: int | None = Field(default=None, gt=0)
    max_candidates_per_iteration: int = Field(default=3, gt=0)
    effort: str | None = None
    max_thinking_tokens: int | None = Field(default=None, gt=0)


class BestOfNOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "claude-sonnet-4-6"
    temperature: float = Field(default=1.0, ge=0)
    max_samples: int | None = Field(default=None, gt=0)
    model_options: dict[str, JsonValue] = Field(default_factory=dict)
    effort: str | None = None
    max_thinking_tokens: int | None = Field(default=None, gt=0)


class Engine(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    kind: EngineKind
    name: str = Field(min_length=1)
    candidate_mode: CandidateMode
    max_evals: int | None = Field(default=100, gt=0)
    max_token_cost: float | None = Field(default=None, gt=0)
    max_concurrency: int = Field(default=8, gt=0)
    output_dir: Path | None = None
    run_dir: Path | None = None
    stop_at_score: float | None = None
    sandbox: bool = True
    gepa_config: GEPAConfig | None = None
    autoresearch_options: AutoResearchOptions | None = None
    meta_harness_options: MetaHarnessOptions | None = None
    best_of_n_options: BestOfNOptions | None = None
    custom_instance: CustomEngine | None = None
    custom_fingerprint: str | None = None

    @model_validator(mode="after")
    def validate_options(self) -> Engine:
        selected = {
            "gepa": self.gepa_config,
            "autoresearch": self.autoresearch_options,
            "meta_harness": self.meta_harness_options,
            "best_of_n": self.best_of_n_options,
            "custom": self.custom_instance,
        }
        if selected[self.kind] is None:
            raise ValueError(f"Engine '{self.kind}' requires its typed configuration.")
        if any(value is not None for name, value in selected.items() if name != self.kind):
            raise ValueError("Engine configuration must match exactly one engine kind.")
        return self

    @classmethod
    def gepa(
        cls,
        config: GEPAConfig | None = None,
        *,
        candidate_mode: CandidateMode = "components",
        max_evals: int | None = None,
        max_token_cost: float | None = None,
        max_concurrency: int = 8,
        output_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        stop_at_score: float | None = None,
        name: str = "gepa",
    ) -> Engine:
        active = config or GEPAConfig()
        return cls(
            kind="gepa",
            name=name,
            candidate_mode=candidate_mode,
            max_evals=max_evals if max_evals is not None else active.budget.max_metric_calls,
            max_token_cost=(
                max_token_cost if max_token_cost is not None else active.budget.max_reflection_cost
            ),
            max_concurrency=max_concurrency,
            output_dir=None if output_dir is None else Path(output_dir),
            run_dir=None if run_dir is None else Path(run_dir),
            stop_at_score=stop_at_score,
            sandbox=False,
            gepa_config=active,
        )

    @classmethod
    def autoresearch(
        cls,
        *,
        model: str = "claude-sonnet-4-6",
        ralph: bool = True,
        max_no_eval_seconds: float | None = None,
        handoffs: Sequence[Mapping[str, JsonValue]] = (),
        effort: str | None = None,
        max_thinking_tokens: int | None = None,
        max_evals: int | None = 100,
        max_token_cost: float | None = None,
        max_concurrency: int = 8,
        output_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        stop_at_score: float | None = None,
        sandbox: bool = True,
        name: str = "autoresearch",
    ) -> Engine:
        return cls(
            kind="autoresearch",
            name=name,
            candidate_mode="text",
            max_evals=max_evals,
            max_token_cost=max_token_cost,
            max_concurrency=max_concurrency,
            output_dir=None if output_dir is None else Path(output_dir),
            run_dir=None if run_dir is None else Path(run_dir),
            stop_at_score=stop_at_score,
            sandbox=sandbox,
            autoresearch_options=AutoResearchOptions(
                model=model,
                ralph=ralph,
                max_no_eval_seconds=max_no_eval_seconds,
                handoffs=tuple(dict(handoff) for handoff in handoffs),
                effort=effort,
                max_thinking_tokens=max_thinking_tokens,
            ),
        )

    @classmethod
    def meta_harness(
        cls,
        *,
        model: str = "claude-sonnet-4-6",
        max_iterations: int | None = None,
        max_candidates_per_iteration: int = 3,
        effort: str | None = None,
        max_thinking_tokens: int | None = None,
        max_evals: int | None = 100,
        max_token_cost: float | None = None,
        max_concurrency: int = 8,
        output_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        stop_at_score: float | None = None,
        sandbox: bool = True,
        name: str = "meta_harness",
    ) -> Engine:
        return cls(
            kind="meta_harness",
            name=name,
            candidate_mode="text",
            max_evals=max_evals,
            max_token_cost=max_token_cost,
            max_concurrency=max_concurrency,
            output_dir=None if output_dir is None else Path(output_dir),
            run_dir=None if run_dir is None else Path(run_dir),
            stop_at_score=stop_at_score,
            sandbox=sandbox,
            meta_harness_options=MetaHarnessOptions(
                model=model,
                max_iterations=max_iterations,
                max_candidates_per_iteration=max_candidates_per_iteration,
                effort=effort,
                max_thinking_tokens=max_thinking_tokens,
            ),
        )

    @classmethod
    def best_of_n(
        cls,
        *,
        model: str = "claude-sonnet-4-6",
        temperature: float = 1.0,
        max_samples: int | None = None,
        model_options: Mapping[str, JsonValue] | None = None,
        effort: str | None = None,
        max_thinking_tokens: int | None = None,
        max_evals: int | None = 100,
        max_token_cost: float | None = None,
        max_concurrency: int = 8,
        output_dir: str | Path | None = None,
        stop_at_score: float | None = None,
        name: str = "best_of_n",
    ) -> Engine:
        return cls(
            kind="best_of_n",
            name=name,
            candidate_mode="text",
            max_evals=max_evals,
            max_token_cost=max_token_cost,
            max_concurrency=max_concurrency,
            output_dir=None if output_dir is None else Path(output_dir),
            stop_at_score=stop_at_score,
            sandbox=False,
            best_of_n_options=BestOfNOptions(
                model=model,
                temperature=temperature,
                max_samples=max_samples,
                model_options=dict(model_options or {}),
                effort=effort,
                max_thinking_tokens=max_thinking_tokens,
            ),
        )

    @classmethod
    def custom(
        cls,
        engine: CustomEngine,
        *,
        candidate_mode: CandidateMode,
        max_evals: int | None = 100,
        max_token_cost: float | None = None,
        max_concurrency: int = 8,
        output_dir: str | Path | None = None,
        run_dir: str | Path | None = None,
        stop_at_score: float | None = None,
        sandbox: bool = True,
        fingerprint: str | None = None,
        name: str | None = None,
    ) -> Engine:
        return cls(
            kind="custom",
            name=name or engine.name,
            candidate_mode=candidate_mode,
            max_evals=max_evals,
            max_token_cost=max_token_cost,
            max_concurrency=max_concurrency,
            output_dir=None if output_dir is None else Path(output_dir),
            run_dir=None if run_dir is None else Path(run_dir),
            stop_at_score=stop_at_score,
            sandbox=sandbox,
            custom_instance=engine,
            custom_fingerprint=fingerprint,
        )

    def declaration(self) -> dict[str, JsonValue]:
        values = self.model_dump(
            mode="json",
            exclude={"custom_instance"},
            fallback=_runtime_reference,
        )
        values["custom_instance"] = (
            None if self.custom_instance is None else _runtime_reference(self.custom_instance)
        )
        return values


class Single(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["single"] = "single"
    engine: Engine


class Sequential(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["sequential"] = "sequential"
    engines: tuple[Engine, ...] = Field(min_length=1)


class Parallel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["parallel"] = "parallel"
    engines: tuple[Engine, ...] = Field(min_length=1)
    max_workers: int | None = Field(default=None, gt=0)


class BestOf(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["best_of"] = "best_of"
    engines: tuple[Engine, ...] = Field(min_length=1)
    max_workers: int | None = Field(default=None, gt=0)


class Vote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["vote"] = "vote"
    engines: tuple[Engine, ...] = Field(min_length=1)
    max_workers: int | None = Field(default=None, gt=0)


class AdaptiveSequential(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["adaptive_sequential"] = "adaptive_sequential"
    engines: tuple[Engine, ...] = Field(min_length=1)
    plateau_evals: int = Field(gt=0)
    max_evals: int | None = Field(default=100, gt=0)
    patience: int = Field(default=1, gt=0)
    min_evals_per_stage: int = Field(default=0, ge=0)
    improvement_epsilon: float = Field(default=0, ge=0)
    cycle: bool = True
    max_switches: int | None = Field(default=None, ge=0)
    max_concurrency: int = Field(default=8, gt=0)
    output_dir: Path | None = None


PipelineStep: TypeAlias = Annotated[
    Single | Sequential | Parallel | BestOf | Vote | AdaptiveSequential,
    Field(discriminator="kind"),
]


class Pipeline(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["pipeline"] = "pipeline"
    steps: tuple[PipelineStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_selection_boundaries(self) -> Pipeline:
        if any(isinstance(step, Parallel) for step in self.steps):
            raise ValueError(
                "Pipeline cannot contain Parallel because it has no selected output. "
                "Use BestOf or Vote at that boundary."
            )
        return self


Composition: TypeAlias = Annotated[
    Single | Sequential | Parallel | BestOf | Vote | AdaptiveSequential | Pipeline,
    Field(discriminator="kind"),
]


class OptimizeAnythingConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    engine: Engine | None = None
    composition: Composition | None = None
    component: str | None = None
    objective: str | None = None
    background: str | None = None
    run: RunConfig = Field(default_factory=RunConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def normalize_target(self) -> OptimizeAnythingConfig:
        if self.engine is None and self.composition is None:
            raise ValueError("Provide engine or composition.")
        if self.engine is not None and self.composition is not None:
            raise ValueError("engine and composition cannot be combined.")
        if self.component is not None and not self.component:
            raise ValueError("component cannot be empty.")
        modes = {engine.candidate_mode for engine in _engines(self.active_composition)}
        if len(modes) > 1:
            raise ValueError(
                "A composition must use one candidate mode. Configure GEPA with "
                "candidate_mode='text' when composing it with text engines."
            )
        _validate_budgets(self.active_composition)
        return self

    @property
    def active_composition(self) -> Composition:
        if self.composition is not None:
            return self.composition
        if self.engine is None:
            raise RuntimeError("Validated OptimizeAnythingConfig has no engine.")
        return Single(engine=self.engine)

    def declaration(self) -> Mapping[str, JsonValue]:
        payload: dict[str, JsonValue] = {
            "component": self.component,
            "objective": self.objective,
            "background": self.background,
            "run": self.run.model_dump(mode="json"),
        }
        if self.engine is not None:
            payload["engine"] = self.engine.declaration()
        else:
            payload["composition"] = _composition_declaration(self.active_composition)
        return payload


def _engines(composition: Composition) -> tuple[Engine, ...]:
    if isinstance(composition, Single):
        return (composition.engine,)
    if isinstance(composition, Pipeline):
        return tuple(engine for step in composition.steps for engine in _engines(step))
    return composition.engines


def _composition_declaration(composition: Composition) -> dict[str, JsonValue]:
    if isinstance(composition, Single):
        return {"kind": composition.kind, "engine": composition.engine.declaration()}
    if isinstance(composition, Pipeline):
        return {
            "kind": composition.kind,
            "steps": [_composition_declaration(step) for step in composition.steps],
        }
    payload: dict[str, JsonValue] = {
        "kind": composition.kind,
        "engines": [engine.declaration() for engine in composition.engines],
    }
    if isinstance(composition, Parallel | BestOf | Vote):
        payload["max_workers"] = composition.max_workers
    if isinstance(composition, AdaptiveSequential):
        payload.update(
            {
                "plateau_evals": composition.plateau_evals,
                "max_evals": composition.max_evals,
                "patience": composition.patience,
                "min_evals_per_stage": composition.min_evals_per_stage,
                "improvement_epsilon": composition.improvement_epsilon,
                "cycle": composition.cycle,
                "max_switches": composition.max_switches,
                "max_concurrency": composition.max_concurrency,
                "output_dir": (
                    None if composition.output_dir is None else str(composition.output_dir)
                ),
            }
        )
    return payload


def _validate_budgets(composition: Composition) -> None:
    if isinstance(composition, Pipeline):
        for step in composition.steps:
            _validate_budgets(step)
        return
    if isinstance(composition, AdaptiveSequential):
        if composition.max_evals is None and any(
            engine.max_token_cost is None for engine in composition.engines
        ):
            raise ValueError(
                "AdaptiveSequential requires max_evals or a max_token_cost on every engine."
            )
        return
    for engine in _engines(composition):
        if engine.max_evals is None and engine.max_token_cost is None:
            raise ValueError(f"Engine '{engine.name}' requires max_evals or max_token_cost.")


def _runtime_reference(value: Any) -> dict[str, str]:
    source_target = value
    if isinstance(value, FunctionType | MethodType):
        reference = f"{value.__module__}.{value.__qualname__}"
    else:
        runtime_type = type(value)
        source_target = runtime_type
        reference = f"{runtime_type.__module__}.{runtime_type.__qualname__}"
    try:
        source = inspect.getsource(source_target)
    except (OSError, TypeError):
        return {"python": reference}
    return {
        "python": reference,
        "source_sha256": sha256(source.encode()).hexdigest(),
    }


__all__ = (
    "AdaptiveSequential",
    "AutoResearchOptions",
    "BestOf",
    "BestOfNOptions",
    "CandidateMode",
    "Composition",
    "CompositionKind",
    "CustomEngine",
    "Engine",
    "EngineResult",
    "EngineKind",
    "EvaluationServer",
    "MetaHarnessOptions",
    "OptimizeAnythingConfig",
    "OptimizationTask",
    "Parallel",
    "Pipeline",
    "PipelineStep",
    "Sequential",
    "Single",
    "Vote",
)
