from __future__ import annotations as _annotations

import builtins
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import pytest
from pydantic import BaseModel

import pydantic_gepa.examples as example_module
from pydantic_gepa import (
    Candidate,
    CandidateComponent,
    ComponentCatalog,
    EvaluationContext,
    Example,
    MetricResult,
    NoopInjection,
    OptimizationDependencyError,
    PydanticEvaluator,
    PydanticGEPAOptimization,
    ScoreObjective,
    model_field_accuracy,
    optimize,
)
from pydantic_gepa.configuration import (
    BudgetConfig,
    ConfigurationError,
    GEPAConfig,
    ReflectionConfig,
)
from pydantic_gepa.experimental.optimize_anything import (
    PydanticOptimizeAnythingAdapter,
    PydanticOptimizeAnythingOptimizer,
)


def test_model_field_accuracy_scores_pydantic_model_fields() -> None:
    score = model_field_accuracy("name", "identifier")
    exact_score = score(
        EvaluationContext(
            inputs=_Input(text="image"),
            output=_Output(name="Ada", identifier="1"),
            expected_output=_Output(name="Ada", identifier="1"),
        )
    )
    partial_score = score(
        EvaluationContext(
            inputs=_Input(text="image"),
            output=_Output(name="Ada", identifier="wrong"),
            expected_output=_Output(name="Ada", identifier="1"),
        )
    )
    missing_expected_score = score(
        EvaluationContext(
            inputs=_Input(text="image"),
            output=_Output(name="Ada", identifier="1"),
        )
    )

    assert exact_score == 1.0
    assert partial_score == 0.5
    assert missing_expected_score == 1.0
    with pytest.raises(ValueError, match="At least one field"):
        model_field_accuracy()


def test_example_pipeline_builds_internal_evals_dataset_and_optimizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_pydantic_evals(monkeypatch)
    component = CandidateComponent(name="instructions", initial_text="Extract fields.")
    examples = [
        Example(
            name="match",
            inputs=_Input(text="Ada:1"),
            expected_output=_Output(name="Ada", identifier="1"),
            metadata={"slice": "easy"},
        ),
        Example(
            name="missing",
            inputs=_Input(text="n/a"),
            expected_output=_Output(name="Unknown", identifier="0"),
            metadata={"slice": "hard"},
        ),
    ]

    pipeline = PydanticGEPAOptimization.from_examples(
        examples=examples,
        val_examples=examples[:1],
        dataset_name="extraction",
        task=parse_output,
        score=lambda ctx: {
            "accuracy": MetricResult(
                score=1.0 if ctx.output == ctx.expected_output else 0.0,
                feedback="field comparison",
                side_info={"field_count": 2, "ignored": None},
            )
        },
        score_key="accuracy",
        components=[component],
        optimize_fn=fake_optimize,
    )
    result = pipeline.optimize(
        config=GEPAConfig(
            reflection=ReflectionConfig(
                model="test:reflection",
                model_kwargs={"temperature": 0.1},
            ),
            budget=BudgetConfig(max_metric_calls=3),
        )
    )

    assert result.best_candidate.values == {"instructions": component.initial_value}
    assert result.best_score == 1.0

    standard = PydanticGEPAOptimization.from_examples(
        examples=examples,
        val_examples=examples,
        task=parse_output,
        score=lambda ctx: 1.0,
        optimize_fn=fake_optimize,
    )
    with pytest.raises(ConfigurationError, match="only supported"):
        standard.optimize(objective="experimental objective")
    assert pipeline.initial_candidate.values == {"instructions": component.initial_value}
    assert [case.name for case in pipeline.trainset] == ["match", "missing"]
    assert [case.name for case in pipeline.valset] == ["match"]


def test_one_shot_optimize_builds_internal_pipeline_and_typed_shortcuts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_pydantic_evals(monkeypatch)
    component = CandidateComponent(name="instructions", initial_text="Extract fields.")
    injection = NoopInjection(component="instructions")
    examples = [
        Example(
            name="match",
            inputs=_Input(text="Ada:1"),
            expected_output=_Output(name="Ada", identifier="1"),
        ),
        Example(
            name="missing",
            inputs=_Input(text="n/a"),
            expected_output=_Output(name="Unknown", identifier="0"),
        ),
    ]
    backend_configs: list[dict[str, Any]] = []

    def one_shot_backend(**kwargs: Any) -> _RawResult:
        backend_configs.append(kwargs)
        batch = kwargs["adapter"].evaluate(
            kwargs["trainset"],
            kwargs["seed_candidate"],
            capture_traces=True,
        )
        assert kwargs["max_metric_calls"] == 3
        assert batch.scores == [1.0, 1.0]
        return _RawResult(dict(kwargs["seed_candidate"]), 1.0)

    result = optimize(
        train=examples,
        validation=examples,
        task=parse_output,
        score=lambda ctx: {
            "accuracy": MetricResult(
                score=float(ctx.output == ctx.expected_output),
                side_info={"field_count": 2},
            )
        },
        score_key="accuracy",
        components=ComponentCatalog.from_components([component]),
        injections=[injection],
        budget=3,
        reflection="test:reflection",
        optimize_fn=one_shot_backend,
    )

    assert result.best_score == 1.0
    assert result.best_candidate.values == {"instructions": component.initial_value}
    assert backend_configs[0]["adapter"].injections == [injection]
    assert backend_configs[0]["reflection_lm"] == "test:reflection"
    for event_handlers in (lambda event: None, [lambda event: None]):
        optimize(
            train=examples,
            validation=examples,
            task=parse_output,
            score=lambda ctx: 1.0,
            budget=3,
            on_event=event_handlers,
            optimize_fn=one_shot_backend,
        )
    optimize(
        train=examples,
        validation=examples,
        task=parse_output,
        score=lambda ctx: 1.0,
        config=GEPAConfig(
            reflection=ReflectionConfig(model="test:reflection"),
            budget=BudgetConfig(max_metric_calls=3),
        ),
        optimize_fn=one_shot_backend,
    )
    assert backend_configs[1]["callbacks"]
    assert backend_configs[2]["callbacks"]
    with pytest.raises(ValueError, match="config cannot be combined"):
        optimize(
            train=examples,
            validation=examples,
            task=parse_output,
            score=lambda ctx: 1.0,
            config=GEPAConfig(),
            budget=1,
        )


def test_example_pipeline_accepts_custom_evaluators_without_score_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_pydantic_evals(monkeypatch)
    evaluator = StaticEvaluator()
    examples = [Example(inputs=_Input(text="Ada:1"), id="case-id")]

    pipeline = PydanticGEPAOptimization.from_examples(
        examples=examples,
        val_examples=examples,
        task=parse_output,
        evaluators=[cast("PydanticEvaluator[_Input, _Output, dict[str, str]]", evaluator)],
        objective=ScoreObjective(score_key="static"),
        initial_candidate=Candidate(values={"instructions": "seed"}),
        optimize_fn=fake_optimize,
    )
    batch = pipeline.adapter.evaluate(
        pipeline.trainset,
        pipeline.initial_candidate.to_gepa_dict(),
        capture_traces=True,
    )

    assert batch.scores == [0.5]
    assert pipeline.initial_candidate.values == {"instructions": "seed"}
    assert pipeline.trainset[0].name == "case-id"


def test_example_pipeline_can_use_optimize_anything_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_pydantic_evals(monkeypatch)
    component = CandidateComponent(name="instructions", initial_text="Extract fields.")
    examples = [
        Example(
            name="match",
            inputs=_Input(text="Ada:1"),
            expected_output=_Output(name="Ada", identifier="1"),
            metadata={"slice": "easy"},
        ),
        Example(
            name="missing",
            inputs=_Input(text="n/a"),
            expected_output=_Output(name="Unknown", identifier="0"),
            metadata={"slice": "hard"},
        ),
    ]

    pipeline = PydanticGEPAOptimization.from_examples(
        examples=examples,
        val_examples=examples[:1],
        dataset_name="extraction",
        task=parse_output,
        score=lambda ctx: {
            "accuracy": MetricResult(
                score=1.0 if ctx.output == ctx.expected_output else 0.0,
                feedback="field comparison",
                side_info={"field_count": 2},
            )
        },
        score_key="accuracy",
        components=ComponentCatalog.from_components([component]),
        backend="optimize_anything",
        background="Use the benchmark records to improve extraction instructions.",
        optimize_fn=fake_optimize_anything,
    )
    result = pipeline.optimize(max_metric_calls=7)

    assert isinstance(pipeline.adapter, PydanticOptimizeAnythingAdapter)
    assert isinstance(pipeline.optimizer, PydanticOptimizeAnythingOptimizer)
    assert pipeline.backend == "optimize_anything"
    assert pipeline.objective == ScoreObjective(score_key="accuracy")
    assert result.best_candidate.values == {"instructions": component.initial_value}
    assert result.best_score == 1.0
    with pytest.raises(ConfigurationError, match="Unsupported Optimize Anything options"):
        pipeline.optimize(unknown_backend_option=True)


def test_example_pipeline_validates_score_sources_and_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_pydantic_evals(monkeypatch)
    with pytest.raises(ValueError, match="Either score or evaluators"):
        PydanticGEPAOptimization.from_examples(
            examples=[Example(inputs=_Input(text="Ada:1"))],
            task=parse_output,
        )

    original_import = builtins.__import__

    def missing_evals_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "pydantic_evals" or name.startswith("pydantic_evals."):
            raise ImportError("missing pydantic-evals")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_evals_import)
    with pytest.raises(OptimizationDependencyError, match="pydantic-evals is not installed"):
        PydanticGEPAOptimization.from_examples(
            examples=[Example(inputs=_Input(text="Ada:1"))],
            task=parse_output,
            score=lambda ctx: 1.0,
        )


def test_optimize_anything_backend_reports_missing_gepa_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_pydantic_evals(monkeypatch)
    examples = [Example(inputs=_Input(text="Ada:1"))]
    pipeline = PydanticGEPAOptimization.from_examples(
        examples=examples,
        val_examples=examples,
        task=parse_output,
        score=lambda ctx: 1.0,
        backend="optimize_anything",
    )
    original_import = builtins.__import__

    def missing_gepa_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "gepa.optimize_anything":
            raise ImportError("missing gepa optimize_anything")
        return original_import(name, globals, locals, fromlist, level)

    math_module = missing_gepa_import("math")
    assert isinstance(math_module, types.ModuleType)
    monkeypatch.delitem(sys.modules, "gepa", raising=False)
    monkeypatch.delitem(sys.modules, "gepa.optimize_anything", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_gepa_import)

    with pytest.raises(OptimizationDependencyError, match="GEPA is not installed"):
        pipeline.optimize(allow_same_train_val=True)


def test_example_pipeline_warns_before_implicit_validation_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_pydantic_evals(monkeypatch)

    with pytest.warns(DeprecationWarning, match="Reusing training examples"):
        pipeline = PydanticGEPAOptimization.from_examples(
            examples=[Example(inputs=_Input(text="Ada:1"))],
            task=parse_output,
            score=lambda ctx: 1.0,
            optimize_fn=fake_optimize,
        )

    assert pipeline.trainset == pipeline.valset


def test_score_output_normalization_handles_direct_metric_and_scalar_mapping() -> None:
    metric_output = example_module._to_evaluation_output(
        MetricResult(
            score=0.25,
            feedback="needs work",
            side_info={"attempts": 2, "details": {"field": "name"}},
        ),
        score_key="accuracy",
        evaluation_reason_type=FakeEvaluationReason,
    )
    mapping_output = example_module._to_evaluation_output(
        {"accuracy": 0.75, "missing": None},
        score_key="accuracy",
        evaluation_reason_type=FakeEvaluationReason,
    )
    scalar_output = example_module._to_evaluation_output(
        0.5,
        score_key="accuracy",
        evaluation_reason_type=FakeEvaluationReason,
    )
    base_score = FakeEvaluator().evaluate(
        FakeEvaluatorContext(
            name="base",
            inputs=_Input(text="Ada:1"),
            output=_Output(name="Ada", identifier="1"),
            expected_output=None,
            metadata=None,
            duration=0.01,
        )
    )

    assert isinstance(metric_output, Mapping)
    metric_mapping = cast("Mapping[str, Any]", metric_output)
    assert fake_score(metric_mapping["accuracy"]).reason == "needs work"
    assert fake_score(metric_mapping["accuracy.attempts"]).value == 2
    assert fake_score(metric_mapping["accuracy.details"]).value == '{"field":"name"}'
    assert mapping_output == {"accuracy": 0.75, "missing": 0.0}
    assert scalar_output == 0.5
    assert base_score == 0.0
    with pytest.raises(TypeError, match="Unsupported fake score"):
        fake_score([])


class _Input(BaseModel):
    text: str


class _Output(BaseModel):
    name: str
    identifier: str


def parse_output(value: _Input) -> _Output:
    if ":" not in value.text:
        return _Output(name="Unknown", identifier="0")
    name, identifier = value.text.split(":", maxsplit=1)
    return _Output(name=name, identifier=identifier)


def fake_optimize(**kwargs: Any) -> _RawResult:
    adapter = kwargs["adapter"]
    trainset = kwargs["trainset"]
    candidate = kwargs["seed_candidate"]
    batch = adapter.evaluate(trainset, candidate, capture_traces=True)

    assert kwargs["max_metric_calls"] == 3
    assert kwargs["reflection_lm_kwargs"] == {"temperature": 0.1}
    assert batch.scores == [1.0, 1.0]
    assert batch.objective_scores == [
        {"accuracy": 1.0, "accuracy.field_count": 2.0},
        {"accuracy": 1.0, "accuracy.field_count": 2.0},
    ]
    return _RawResult(best_candidate=dict(candidate), best_score=1.0)


def fake_optimize_anything(**kwargs: Any) -> _RawResult:
    assert kwargs["seed_candidate"] == {"instructions": "Extract fields."}
    assert kwargs["objective"] == "Maximize accuracy on the evaluation dataset."
    assert kwargs["background"] == "Use the benchmark records to improve extraction instructions."
    config = kwargs["config"]
    assert config.engine.max_metric_calls == 7

    dataset = kwargs["dataset"]
    valset = kwargs["valset"]
    assert [case.name for case in dataset] == ["match", "missing"]
    assert [case.name for case in valset] == ["match"]

    score, side_info = kwargs["evaluator"](
        kwargs["seed_candidate"],
        example=dataset[0],
    )

    assert score == 1.0
    assert side_info["scores"] == {"accuracy": 1.0}
    assert side_info["observed_scores"] == {"accuracy": 1.0, "accuracy.field_count": 2}
    assert side_info["objective_scores"] == {"accuracy": 1.0, "accuracy.field_count": 2.0}
    assert side_info["case_name"] == "match"
    assert side_info["inputs"] == {"text": "Ada:1"}
    assert side_info["actual_output"] == {"name": "Ada", "identifier": "1"}
    assert side_info["instructions_specific_info"] == {
        "examples": [
            {
                "case_name": "match",
                "inputs": {"text": "Ada:1"},
                "score": 1.0,
                "success": True,
                "failure_category": None,
                "expected_output": {"name": "Ada", "identifier": "1"},
                "metadata": {},
                "actual_output": {"name": "Ada", "identifier": "1"},
                "scores": {"accuracy": 1.0, "accuracy.field_count": 2},
            }
        ]
    }
    return _RawResult(best_candidate=dict(kwargs["seed_candidate"]), best_score=1.0)


def install_fake_pydantic_evals(monkeypatch: pytest.MonkeyPatch) -> None:
    package = types.ModuleType("pydantic_evals")
    evaluators = types.ModuleType("pydantic_evals.evaluators")
    package.__dict__["Case"] = FakeCase
    package.__dict__["Dataset"] = FakeDataset
    evaluators.__dict__["Evaluator"] = FakeEvaluator
    evaluators.__dict__["EvaluationReason"] = FakeEvaluationReason
    monkeypatch.setitem(sys.modules, "pydantic_evals", package)
    monkeypatch.setitem(sys.modules, "pydantic_evals.evaluators", evaluators)


@dataclass(frozen=True)
class FakeCase:
    name: str | None
    inputs: _Input
    expected_output: _Output | None = None
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class FakeEvaluationReason:
    value: bool | int | float | str
    reason: str | None = None


@dataclass(frozen=True)
class FakeEvaluatorContext:
    name: str | None
    inputs: _Input
    output: _Output
    expected_output: _Output | None
    metadata: dict[str, str] | None
    duration: float


@dataclass(frozen=True)
class FakeScore:
    value: bool | int | float | str
    reason: str | None = None


@dataclass(frozen=True)
class FakeReportCase:
    name: str | None
    inputs: _Input
    expected_output: _Output | None
    output: _Output
    scores: dict[str, FakeScore]


@dataclass(frozen=True)
class FakeReport:
    cases: list[FakeReportCase]
    failures: list[str]


class FakeEvaluator:
    evaluation_name = "score"

    def evaluate(self, ctx: FakeEvaluatorContext) -> float:
        del ctx
        return 0.0


class StaticEvaluator(FakeEvaluator):
    evaluation_name = "static"

    def evaluate(self, ctx: FakeEvaluatorContext) -> float:
        return 0.5


class FakeDataset:
    def __init__(
        self,
        *,
        name: str | None = None,
        cases: Sequence[FakeCase],
        evaluators: Sequence[FakeEvaluator],
    ) -> None:
        self.name = name
        self.cases = list(cases)
        self.evaluators = list(evaluators)

    def evaluate(
        self,
        task: Any,
        *,
        max_concurrency: int,
        progress: bool,
    ) -> FakeReport:
        del max_concurrency, progress
        cases: list[FakeReportCase] = []
        for case in self.cases:
            output = task(case.inputs)
            context = FakeEvaluatorContext(
                name=case.name,
                inputs=case.inputs,
                output=output,
                expected_output=case.expected_output,
                metadata=case.metadata,
                duration=0.01,
            )
            scores: dict[str, FakeScore] = {}
            for evaluator in self.evaluators:
                result = evaluator.evaluate(context)
                scores.update(normalize_fake_scores(evaluator.evaluation_name, result))
            cases.append(
                FakeReportCase(
                    name=case.name,
                    inputs=case.inputs,
                    expected_output=case.expected_output,
                    output=output,
                    scores=scores,
                )
            )
        return FakeReport(cases=cases, failures=[])


def normalize_fake_scores(name: str, result: Any) -> dict[str, FakeScore]:
    if isinstance(result, Mapping):
        return {key: fake_score(value) for key, value in result.items() if isinstance(key, str)}
    return {name: fake_score(result)}


def fake_score(value: Any) -> FakeScore:
    if isinstance(value, FakeEvaluationReason):
        return FakeScore(value=value.value, reason=value.reason)
    if isinstance(value, bool | int | float | str):
        return FakeScore(value=value)
    raise TypeError(f"Unsupported fake score value: {value!r}")


@dataclass(frozen=True)
class _RawResult:
    best_candidate: dict[str, str]
    best_score: float
