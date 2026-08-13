from __future__ import annotations as _annotations

import json
import warnings
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

from pydantic import BaseModel

from .adapter import PydanticGEPAAdapter
from .candidates import Candidate, CandidateComponent
from .components import ComponentCatalog
from .configuration import ConfigurationError, GEPAConfig, RunConfig
from .configuration.models import BudgetConfig, ReflectionConfig, TrackingConfig
from .evaluation.data import DataSplit
from .evaluation.models import Example
from .events import Observer
from .experimental.optimize_anything import (
    OptimizeAnythingConfig,
    OptimizeAnythingFn,
    PydanticOptimizeAnythingAdapter,
    PydanticOptimizeAnythingOptimizer,
)
from .injections import CandidateInjection
from .objectives import MetricResult, ScoreInput, ScoreObjective
from .optimizer import OptimizeFn, PydanticGEPAOptimizer
from .results import PydanticGEPAResult
from .values import SerializableValue

InputsT = TypeVar("InputsT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")
InputsT_co = TypeVar("InputsT_co", covariant=True)
OutputT_co = TypeVar("OutputT_co", covariant=True)
MetadataT_co = TypeVar("MetadataT_co", covariant=True)
InputsT_contra = TypeVar("InputsT_contra", contravariant=True)
OutputT_contra = TypeVar("OutputT_contra", contravariant=True)
MetadataT_contra = TypeVar("MetadataT_contra", contravariant=True)
OutputModelT = TypeVar("OutputModelT", bound=BaseModel)

EvaluationScalar: TypeAlias = bool | int | float | str
OptimizationBackend: TypeAlias = Literal["standard", "optimize_anything"]
ScoreOutput: TypeAlias = ScoreInput | Mapping[str, ScoreInput]
ScoreFunction: TypeAlias = Callable[["EvaluationContext[InputsT, OutputT, MetadataT]"], ScoreOutput]


@dataclass(frozen=True, slots=True)
class EvaluationContext(Generic[InputsT, OutputT, MetadataT]):
    inputs: InputsT
    output: OutputT
    expected_output: OutputT | None = None
    name: str | None = None
    metadata: MetadataT | None = None
    duration: float | None = None


@runtime_checkable
class EvalContextView(Protocol[InputsT_co, OutputT_co, MetadataT_co]):
    @property
    def name(self) -> str | None: ...

    @property
    def inputs(self) -> InputsT_co: ...

    @property
    def output(self) -> OutputT_co: ...

    @property
    def expected_output(self) -> OutputT_co | None: ...

    @property
    def metadata(self) -> MetadataT_co | None: ...

    @property
    def duration(self) -> float: ...


@runtime_checkable
class EvalCaseView(Protocol[InputsT_co, OutputT_co, MetadataT_co]):
    @property
    def name(self) -> str | None: ...

    @property
    def inputs(self) -> InputsT_co: ...

    @property
    def expected_output(self) -> OutputT_co | None: ...

    @property
    def metadata(self) -> MetadataT_co | None: ...


@runtime_checkable
class PydanticEvaluator(Protocol[InputsT_contra, OutputT_contra, MetadataT_contra]):
    def evaluate(
        self,
        ctx: EvalContextView[InputsT_contra, OutputT_contra, MetadataT_contra],
    ) -> EvaluationOutput | Awaitable[EvaluationOutput]: ...


@runtime_checkable
class EvaluationReasonView(Protocol):
    @property
    def value(self) -> EvaluationScalar: ...

    @property
    def reason(self) -> str | None: ...


class EvaluationReasonFactory(Protocol):
    def __call__(
        self,
        value: EvaluationScalar,
        reason: str | None = None,
    ) -> EvaluationReasonView: ...


EvaluationValue: TypeAlias = EvaluationScalar | EvaluationReasonView
EvaluationOutput: TypeAlias = EvaluationValue | Mapping[str, EvaluationValue]
EvaluationPrimitiveInput: TypeAlias = EvaluationScalar | None


@dataclass(frozen=True, slots=True)
class Optimization(Generic[InputsT, OutputT, MetadataT]):
    adapter: (
        PydanticGEPAAdapter[
            EvalCaseView[InputsT, OutputT, MetadataT],
            OutputT,
            PydanticEvaluator[InputsT, OutputT, MetadataT],
        ]
        | PydanticOptimizeAnythingAdapter[
            EvalCaseView[InputsT, OutputT, MetadataT],
            OutputT,
            PydanticEvaluator[InputsT, OutputT, MetadataT],
        ]
    )
    optimizer: (
        PydanticGEPAOptimizer[
            EvalCaseView[InputsT, OutputT, MetadataT],
            OutputT,
            PydanticEvaluator[InputsT, OutputT, MetadataT],
        ]
        | PydanticOptimizeAnythingOptimizer[
            EvalCaseView[InputsT, OutputT, MetadataT],
            OutputT,
            PydanticEvaluator[InputsT, OutputT, MetadataT],
        ]
    )
    trainset: list[EvalCaseView[InputsT, OutputT, MetadataT]]
    valset: list[EvalCaseView[InputsT, OutputT, MetadataT]]
    initial_candidate: Candidate
    testset: list[EvalCaseView[InputsT, OutputT, MetadataT]] = field(default_factory=list)
    backend: OptimizationBackend = "standard"

    @property
    def objective(self) -> ScoreObjective:
        if isinstance(self.adapter, PydanticGEPAAdapter):
            return self.adapter.objective
        return self.adapter.adapter.objective

    @classmethod
    def from_examples(
        cls,
        *,
        examples: Sequence[Example[InputsT, OutputT, MetadataT]] | None = None,
        data: DataSplit[InputsT, OutputT, MetadataT] | None = None,
        task: Callable[[InputsT], OutputT],
        score: ScoreFunction[InputsT, OutputT, MetadataT] | None = None,
        score_key: str = "score",
        val_examples: Sequence[Example[InputsT, OutputT, MetadataT]] | None = None,
        test_examples: Sequence[Example[InputsT, OutputT, MetadataT]] = (),
        dataset_name: str | None = None,
        injections: Sequence[CandidateInjection] = (),
        components: ComponentCatalog | Sequence[CandidateComponent] | None = None,
        initial_candidate: Candidate | None = None,
        objective: ScoreObjective | None = None,
        evaluators: Sequence[PydanticEvaluator[InputsT, OutputT, MetadataT]] = (),
        optimize_fn: OptimizeFn | OptimizeAnythingFn | None = None,
        max_concurrency: int = 5,
        backend: OptimizationBackend = "standard",
        optimization_objective: str | None = None,
        background: str | None = None,
    ) -> Optimization[InputsT, OutputT, MetadataT]:
        if score is None and not evaluators:
            raise ValueError("Either score or evaluators must be provided.")
        if data is not None and (examples is not None or val_examples is not None or test_examples):
            raise ValueError(
                "data cannot be combined with examples, val_examples, or test_examples."
            )
        if data is None and examples is None:
            raise ValueError("Provide examples or data.")

        train_examples = data.train if data is not None else tuple(examples or ())
        active_test_examples = data.test if data is not None else test_examples

        case_type, dataset_type, evaluator_type, evaluation_reason_type = _load_pydantic_evals()
        trainset = [_to_case(case_type, example) for example in train_examples]
        if data is not None:
            active_val_examples = data.validation
        elif val_examples is None:
            warnings.warn(
                "Reusing training examples for validation is deprecated; pass val_examples "
                "explicitly or use DataSplit.",
                DeprecationWarning,
                stacklevel=2,
            )
            active_val_examples = train_examples
        else:
            active_val_examples = val_examples
        valset = [_to_case(case_type, example) for example in active_val_examples]
        testset = [_to_case(case_type, example) for example in active_test_examples]

        active_evaluators = list(evaluators)
        if score is not None:
            active_evaluators.append(
                _callable_score_evaluator(
                    evaluator_type=evaluator_type,
                    evaluation_reason_type=evaluation_reason_type,
                    score=score,
                    score_key=score_key,
                )
            )

        dataset = dataset_type(
            name=dataset_name or "pydantic-gepa",
            cases=[],
            evaluators=cast("Sequence[Any]", active_evaluators),
        )
        active_objective = objective or ScoreObjective(score_key=score_key)
        active_components = (
            components
            if isinstance(components, ComponentCatalog) or components is None
            else ComponentCatalog.from_components(components)
        )
        active_initial_candidate = initial_candidate or (
            active_components.to_candidate() if active_components is not None else Candidate()
        )
        adapter = PydanticGEPAAdapter.from_dataset(
            dataset=dataset,
            task=cast("Callable[[EvalCaseView[InputsT, OutputT, MetadataT]], OutputT]", task),
            injections=list(injections),
            objective=active_objective,
            components=active_components,
            max_concurrency=max_concurrency,
        )
        if backend == "standard":
            active_adapter = adapter
            optimizer = PydanticGEPAOptimizer(
                adapter=active_adapter,
                initial_candidate=active_initial_candidate,
                optimize_fn=cast("OptimizeFn | None", optimize_fn),
            )
        else:
            active_adapter = PydanticOptimizeAnythingAdapter(adapter=adapter)
            optimizer = PydanticOptimizeAnythingOptimizer(
                adapter=active_adapter,
                initial_candidate=active_initial_candidate,
                optimization_objective=optimization_objective
                or _default_optimization_objective(active_objective),
                background=background,
                optimize_fn=cast(
                    "OptimizeAnythingFn[EvalCaseView[InputsT, OutputT, MetadataT]] | None",
                    optimize_fn,
                ),
            )
        return cls(
            adapter=cast(
                "PydanticGEPAAdapter[EvalCaseView[InputsT, OutputT, MetadataT], OutputT, PydanticEvaluator[InputsT, OutputT, MetadataT]] | PydanticOptimizeAnythingAdapter[EvalCaseView[InputsT, OutputT, MetadataT], OutputT, PydanticEvaluator[InputsT, OutputT, MetadataT]]",
                active_adapter,
            ),
            optimizer=cast(
                "PydanticGEPAOptimizer[EvalCaseView[InputsT, OutputT, MetadataT], OutputT, PydanticEvaluator[InputsT, OutputT, MetadataT]] | PydanticOptimizeAnythingOptimizer[EvalCaseView[InputsT, OutputT, MetadataT], OutputT, PydanticEvaluator[InputsT, OutputT, MetadataT]]",
                optimizer,
            ),
            trainset=trainset,
            valset=valset,
            testset=testset,
            initial_candidate=active_initial_candidate,
            backend=backend,
        )

    def optimize(
        self,
        *,
        initial_candidate: Candidate | None = None,
        config: GEPAConfig | OptimizeAnythingConfig | None = None,
        max_metric_calls: int | None = None,
        allow_same_train_val: bool | None = None,
        objective: str | None = None,
        background: str | None = None,
        **kwargs: SerializableValue,
    ) -> PydanticGEPAResult:
        if isinstance(self.optimizer, PydanticOptimizeAnythingOptimizer):
            if kwargs:
                names = ", ".join(sorted(kwargs))
                raise ConfigurationError(
                    f"Unsupported Optimize Anything options: {names}. Use typed GEPAConfig fields."
                )
            optimizer = cast(
                "PydanticOptimizeAnythingOptimizer[EvalCaseView[InputsT, OutputT, MetadataT], OutputT, PydanticEvaluator[InputsT, OutputT, MetadataT]]",
                self.optimizer,
            )
            return optimizer.optimize(
                trainset=self.trainset,
                valset=self.valset,
                testset=self.testset,
                initial_candidate=initial_candidate,
                config=config,
                max_metric_calls=max_metric_calls,
                allow_same_train_val=allow_same_train_val,
                objective=objective,
                background=background,
            )
        if objective is not None or background is not None:
            raise ConfigurationError(
                "objective and background overrides are only supported by the Optimize Anything backend."
            )
        if config is not None and not isinstance(config, GEPAConfig):
            raise ConfigurationError("OptimizeAnythingConfig requires backend='optimize_anything'.")
        optimizer = cast(
            "PydanticGEPAOptimizer[EvalCaseView[InputsT, OutputT, MetadataT], OutputT, PydanticEvaluator[InputsT, OutputT, MetadataT]]",
            self.optimizer,
        )
        return optimizer.optimize(
            trainset=self.trainset,
            valset=self.valset,
            initial_candidate=initial_candidate,
            config=config,
            max_metric_calls=max_metric_calls,
            allow_same_train_val=allow_same_train_val,
            **kwargs,
        )

    run = optimize


PydanticGEPAOptimization = Optimization


def optimize(
    *,
    train: Sequence[Example[InputsT, OutputT, MetadataT]],
    validation: Sequence[Example[InputsT, OutputT, MetadataT]],
    task: Callable[[InputsT], OutputT],
    score: ScoreFunction[InputsT, OutputT, MetadataT] | None = None,
    evaluators: Sequence[PydanticEvaluator[InputsT, OutputT, MetadataT]] = (),
    components: ComponentCatalog | Sequence[CandidateComponent] | None = None,
    injections: Sequence[CandidateInjection] = (),
    initial_candidate: Candidate | None = None,
    objective: ScoreObjective | None = None,
    score_key: str = "score",
    config: GEPAConfig | None = None,
    budget: int | None = None,
    reflection: str | None = None,
    run: RunConfig | None = None,
    on_event: Observer | Sequence[Observer] | None = None,
    optimize_fn: OptimizeFn | None = None,
    max_concurrency: int = 5,
) -> PydanticGEPAResult:
    if config is not None and any(
        value is not None for value in (budget, reflection, run, on_event)
    ):
        raise ValueError(
            "config cannot be combined with budget, reflection, run, or on_event shortcuts."
        )
    active_config = config
    if active_config is None:
        if on_event is None:
            observers: tuple[Observer, ...] = ()
        elif callable(on_event):
            observers = (cast("Observer", on_event),)
        else:
            observers = tuple(on_event)
        active_config = GEPAConfig(
            budget=BudgetConfig(max_metric_calls=budget or 50),
            reflection=ReflectionConfig(model=reflection),
            run=run or RunConfig(),
            tracking=TrackingConfig(observers=observers),
        )
    optimization = Optimization.from_examples(
        examples=train,
        val_examples=validation,
        task=task,
        score=score,
        evaluators=evaluators,
        components=components,
        injections=injections,
        initial_candidate=initial_candidate,
        objective=objective,
        score_key=score_key,
        optimize_fn=optimize_fn,
        max_concurrency=max_concurrency,
    )
    return optimization.run(config=active_config)


def model_field_accuracy(
    *fields: str,
) -> ScoreFunction[InputsT, OutputModelT, MetadataT]:
    if not fields:
        raise ValueError("At least one field must be provided.")

    def score(ctx: EvaluationContext[InputsT, OutputModelT, MetadataT]) -> float:
        expected_output = ctx.expected_output
        if expected_output is None:
            return 1.0
        output_values = ctx.output.model_dump()
        expected_values = expected_output.model_dump()
        correct = sum(
            1 for field in fields if output_values.get(field) == expected_values.get(field)
        )
        return correct / len(fields)

    return score


def _load_pydantic_evals():
    try:
        from pydantic_evals import Case, Dataset
        from pydantic_evals.evaluators import EvaluationReason, Evaluator
    except ImportError as exc:  # pragma: no cover - covered with patched import
        from .errors import OptimizationDependencyError

        raise OptimizationDependencyError(
            "pydantic-evals is not installed. Install pydantic-gepa[integrations]."
        ) from exc
    return Case, Dataset, Evaluator, EvaluationReason


def _to_case(
    case_type,
    example: Example[InputsT, OutputT, MetadataT],
) -> EvalCaseView[InputsT, OutputT, MetadataT]:
    return case_type(
        name=example.name or example.id,
        inputs=example.inputs,
        expected_output=example.expected_output,
        metadata=example.metadata,
    )


def _callable_score_evaluator(
    *,
    evaluator_type,
    evaluation_reason_type: EvaluationReasonFactory,
    score: ScoreFunction[InputsT, OutputT, MetadataT],
    score_key: str,
) -> PydanticEvaluator[InputsT, OutputT, MetadataT]:
    class CallableScoreEvaluator(evaluator_type):
        evaluation_name = score_key

        @classmethod
        def get_serialization_name(cls) -> str:
            return score_key

        def evaluate(
            self,
            ctx: EvalContextView[InputsT, OutputT, MetadataT],
        ) -> EvaluationOutput:
            output = score(
                EvaluationContext(
                    name=ctx.name,
                    inputs=ctx.inputs,
                    output=ctx.output,
                    expected_output=ctx.expected_output,
                    metadata=ctx.metadata,
                    duration=ctx.duration,
                )
            )
            return _to_evaluation_output(
                output,
                score_key=score_key,
                evaluation_reason_type=evaluation_reason_type,
            )

    return CallableScoreEvaluator()


def _to_evaluation_output(
    output: ScoreOutput,
    *,
    score_key: str,
    evaluation_reason_type: EvaluationReasonFactory,
) -> EvaluationOutput:
    if isinstance(output, Mapping):
        normalized: dict[str, EvaluationValue] = {}
        typed_output = cast("Mapping[str, ScoreInput]", output)
        for key, value in typed_output.items():
            normalized.update(
                _score_input_mapping(
                    key,
                    value,
                    evaluation_reason_type=evaluation_reason_type,
                )
            )
        return normalized
    if isinstance(output, MetricResult):
        return _metric_result_mapping(
            score_key,
            output,
            evaluation_reason_type=evaluation_reason_type,
        )
    return _score_input_value(output)


def _score_input_mapping(
    key: str,
    value: ScoreInput,
    *,
    evaluation_reason_type: EvaluationReasonFactory,
) -> dict[str, EvaluationValue]:
    if isinstance(value, MetricResult):
        return _metric_result_mapping(
            key,
            value,
            evaluation_reason_type=evaluation_reason_type,
        )
    return {key: _score_input_value(value)}


def _metric_result_mapping(
    key: str,
    value: MetricResult,
    *,
    evaluation_reason_type: EvaluationReasonFactory,
) -> dict[str, EvaluationValue]:
    normalized: dict[str, EvaluationValue] = {
        key: evaluation_reason_type(value.score, value.feedback)
    }
    for side_key, side_value in value.side_info.items():
        if side_value is not None:
            normalized[f"{key}.{side_key}"] = (
                side_value
                if isinstance(side_value, str | int | float | bool)
                else json.dumps(side_value, sort_keys=True, separators=(",", ":"))
            )
    return normalized


def _score_input_value(
    value: EvaluationPrimitiveInput,
) -> EvaluationValue:
    if value is None:
        return 0.0
    return value


def _default_optimization_objective(objective: ScoreObjective) -> str:
    verb = "Maximize" if objective.direction == "maximize" else "Minimize"
    return f"{verb} {objective.score_key} on the evaluation dataset."


__all__ = (
    "EvalContextView",
    "EvaluationContext",
    "EvaluationOutput",
    "EvaluationReasonView",
    "EvaluationScalar",
    "EvaluationValue",
    "Example",
    "OptimizationBackend",
    "Optimization",
    "PydanticEvaluator",
    "PydanticGEPAOptimization",
    "ScoreFunction",
    "ScoreOutput",
    "model_field_accuracy",
    "optimize",
)
