from __future__ import annotations as _annotations

import sys
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from pydantic import ValidationError

import pydantic_gepa.optimizer as optimizer_module
from pydantic_gepa import (
    CallableReflectionModel,
    Candidate,
    GEPAConfig,
    PydanticGEPAAdapter,
    PydanticGEPAOptimizer,
    ReflectionModel,
    ScoreObjective,
)
from pydantic_gepa.configuration import (
    BudgetConfig,
    ConfigurationError,
    EvaluationSetConfig,
    MergeConfig,
    ProgressConfig,
    ReflectionConfig,
    RunConfig,
    SelectionConfig,
    TrackingConfig,
)
from pydantic_gepa.errors import RunStoreError
from pydantic_gepa.events import Event
from pydantic_gepa.integrations.pydantic_ai import PydanticAIReflectionModel
from pydantic_gepa.optimizer import OptimizeFn
from pydantic_gepa.reflection import (
    ReflectionPrompt,
    ReflectionResponse,
    ReflectionUsage,
)
from pydantic_gepa.values import SerializableValue

if TYPE_CHECKING:
    from pydantic_ai import Agent


def test_gepa_config_maps_every_standard_backend_option() -> None:
    callback = _Opaque("callback")
    stopper = _Opaque("stopper")
    logger = _Opaque("logger")
    config = GEPAConfig(
        reflection=ReflectionConfig(
            model="provider:model",
            model_kwargs={"temperature": 0.2},
            minibatch_size=4,
            perfect_score=0.99,
            skip_perfect_score=False,
            prompt_template={"prompt": "<curr_param> <side_info>"},
            proposer=_propose,
        ),
        selection=SelectionConfig(
            candidate="top_k_pareto",
            frontier="hybrid",
            component="all",
            batch_sampler="epoch_shuffled",
            validation="full_eval",
            acceptance="improvement_or_equal",
        ),
        merge=MergeConfig(
            enabled=True,
            max_invocations=3,
            validation_overlap_floor=2,
        ),
        budget=BudgetConfig(
            max_metric_calls=20,
            max_reflection_cost=1.5,
            stop=(stopper,),
        ),
        run=RunConfig(
            directory=Path("runs/demo"),
            seed=7,
            use_cloudpickle=True,
            cache_evaluations=True,
            raise_on_exception=False,
        ),
        tracking=TrackingConfig(
            logger=logger,
            backend_callbacks=(callback,),
            track_best_outputs=False,
            key_prefix="demo/",
            use_wandb=True,
            wandb_api_key="secret",
            wandb_init={"project": "demo"},
            wandb_attach_existing=True,
            use_mlflow=True,
            mlflow_tracking_uri="sqlite:///mlflow.db",
            mlflow_experiment_name="demo",
            mlflow_attach_existing=True,
        ),
        progress=ProgressConfig(display_bar=True),
        evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
    )

    kwargs = config.to_backend_kwargs()

    assert set(kwargs) == {
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
    assert kwargs["reflection_lm"] == "provider:model"
    assert kwargs["reflection_lm_kwargs"] == {"temperature": 0.2}
    assert kwargs["custom_candidate_proposer"] is _propose
    assert kwargs["stop_callbacks"] is stopper
    assert kwargs["callbacks"] == (callback,)
    assert kwargs["run_dir"] == "runs/demo"
    assert kwargs["display_progress_bar"] is True
    assert kwargs["acceptance_criterion"] == "improvement_or_equal"


def test_config_defaults_and_multiple_stoppers_map_without_noise() -> None:
    first = _Opaque("first")
    second = _Opaque("second")
    config = GEPAConfig(
        budget=BudgetConfig(max_metric_calls=None, stop=(first, second)),
    )
    kwargs = config.to_backend_kwargs()

    assert kwargs["stop_callbacks"] == (first, second)
    assert kwargs["reflection_lm_kwargs"] is None
    assert kwargs["callbacks"] is None
    assert kwargs["wandb_init_kwargs"] is None
    assert kwargs["run_dir"] is None


def test_config_rejects_unknown_and_conflicting_settings() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        GEPAConfig.model_validate({"unknown": True})
    with pytest.raises(ValidationError, match="string model identifier"):
        ReflectionConfig(model=lambda prompt: str(prompt), model_kwargs={"temperature": 0.2})
    with pytest.raises(ValidationError, match="max_invocations"):
        MergeConfig(enabled=True, max_invocations=0)
    with pytest.raises(ValidationError, match="At least one"):
        BudgetConfig(max_metric_calls=None)
    with pytest.raises(ValidationError, match="wandb_attach_existing"):
        TrackingConfig(wandb_attach_existing=True)
    with pytest.raises(ValidationError, match="mlflow_attach_existing"):
        TrackingConfig(mlflow_attach_existing=True)


def test_legacy_kwargs_map_to_typed_config_and_reject_unknown_options() -> None:
    callback = _Opaque("callback")
    with pytest.warns(DeprecationWarning, match="GEPAConfig"):
        config = GEPAConfig.from_legacy_kwargs(
            {
                "reflection_lm": "provider:model",
                "max_metric_calls": 12,
                "callbacks": callback,
                "stop_callbacks": [callback],
                "display_progress_bar": True,
            }
        )

    assert config.reflection.model == "provider:model"
    assert config.budget.max_metric_calls == 12
    assert config.budget.stop == (callback,)
    assert config.tracking.backend_callbacks == (callback,)
    assert config.progress.display_bar is True

    with pytest.raises(ConfigurationError, match="Unsupported GEPA options: typo"):
        GEPAConfig.from_legacy_kwargs({"typo": True})
    with pytest.warns(DeprecationWarning):
        seed_only = GEPAConfig.from_legacy_kwargs({"seed": 3})
    assert seed_only.run.seed == 3


def test_callable_reflection_normalizes_usage_chat_prompts_and_async_results() -> None:
    async def reflect(prompt: ReflectionPrompt) -> ReflectionResponse:
        assert isinstance(prompt, list)
        return ReflectionResponse(
            text="improved",
            usage=ReflectionUsage(
                requests=1,
                input_tokens=10,
                output_tokens=4,
                cost=0.02,
            ),
            metadata={"provider": "test"},
        )

    model = CallableReflectionModel(reflect)
    prompt: ReflectionPrompt = [{"role": "user", "content": "improve"}]

    assert isinstance(model, ReflectionModel)
    assert model(prompt) == "improved"
    assert model.total_tokens_in == 10
    assert model.total_tokens_out == 4
    assert model.total_cost == 0.02
    assert model.records[0].response is not None
    assert model.records[0].response.metadata == {"provider": "test"}


def test_callable_reflection_estimates_usage_retries_and_handles_failures() -> None:
    attempts = 0

    def flaky(prompt: ReflectionPrompt) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("retry")
        return f"improved:{prompt}"

    model = CallableReflectionModel(flaky, retries=1)
    assert model("prompt") == "improved:prompt"
    assert model.total_tokens_in == 1
    assert model.total_tokens_out == 3
    assert model.records[0].attempts == 2

    empty = CallableReflectionModel(
        lambda prompt: _raise_reflection(prompt),
        on_error="empty",
    )
    assert empty("prompt") == ""
    assert empty.records[0].error is not None
    assert empty.records[0].error.type == "RuntimeError"

    raising = CallableReflectionModel(lambda prompt: _raise_reflection(prompt))
    with pytest.raises(RuntimeError, match="reflection failed"):
        raising("prompt")
    assert raising.records[0].error is not None

    with pytest.raises(ValueError, match="retries cannot be negative"):
        CallableReflectionModel(lambda prompt: str(prompt), retries=-1)


def test_pydantic_ai_reflection_uses_agent_settings_usage_and_message_prompts() -> None:
    agent = _Agent()
    model = PydanticAIReflectionModel(
        agent=cast("Agent[None, str]", agent),
        deps=None,
        model_settings={"temperature": 0.1},
        max_output_tokens=120,
        timeout=3.0,
    )

    output = model(
        [
            {"role": "system", "content": "reflect"},
            {"role": "user", "content": "improve"},
        ]
    )

    assert output == "agent reflection"
    assert agent.prompt == "system: reflect\nuser: improve"
    assert agent.settings == {"temperature": 0.1, "max_tokens": 120, "timeout": 3.0}
    assert model.total_tokens_in == 8
    assert model.total_tokens_out == 3
    assert model.total_cost == 0.0
    assert model.records[0].response is not None
    assert model.records[0].response.metadata["tool_calls"] == 2


def test_pydantic_ai_reflection_can_construct_an_agent_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("pydantic_ai")
    module.__dict__["Agent"] = _AgentFactory
    monkeypatch.setitem(sys.modules, "pydantic_ai", module)

    model = PydanticAIReflectionModel.from_model("test:model")

    assert model("prompt") == "agent reflection"
    assert isinstance(model.agent, _AgentFactory)
    assert model.agent.model == "test:model"
    assert model.agent.output_type is str


def test_optimizer_accepts_typed_config_and_rejects_configuration_conflicts() -> None:
    adapter: PydanticGEPAAdapter[_Case, str, None] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(),
        task=lambda case: case.name,
        injections=[],
        objective=ScoreObjective(score_key="score"),
    )
    seen: list[dict[str, SerializableValue]] = []

    def optimize(**kwargs: SerializableValue) -> _RawResult:
        seen.append(dict(kwargs))
        return _RawResult()

    optimizer = PydanticGEPAOptimizer(
        adapter=adapter,
        initial_candidate=Candidate(values={"prompt": "seed"}),
        optimize_fn=cast("OptimizeFn", optimize),
    )
    config = GEPAConfig(
        reflection=ReflectionConfig(model="test:model"),
        evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
    )

    result = optimizer.optimize(trainset=[_Case("case")], config=config)
    assert result.best_score == 1.0
    assert seen[0]["reflection_lm"] == "test:model"

    with pytest.warns(DeprecationWarning, match="GEPAConfig"):
        legacy_result = optimizer.optimize(
            trainset=[_Case("case")],
            max_metric_calls=3,
            allow_same_train_val=True,
            seed=4,
        )
    assert legacy_result.best_score == 1.0
    assert seen[1]["seed"] == 4

    direct_result = optimizer.optimize(
        trainset=[_Case("case")],
        max_metric_calls=2,
        allow_same_train_val=True,
    )
    assert direct_result.best_score == 1.0

    with pytest.raises(ConfigurationError, match="cannot be combined"):
        optimizer.optimize(
            trainset=[_Case("case")],
            config=config,
            max_metric_calls=2,
        )

    with pytest.raises(ConfigurationError, match="reflection.model is required"):
        PydanticGEPAOptimizer(
            adapter=adapter,
            initial_candidate=Candidate(values={"prompt": "seed"}),
        ).optimize(
            trainset=[_Case("case")],
            config=GEPAConfig(
                evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True)
            ),
        )

    proposer_adapter = adapter.model_copy(update={"propose_new_texts": _propose})
    with pytest.raises(ConfigurationError, match="either the adapter or GEPAConfig"):
        PydanticGEPAOptimizer(
            adapter=proposer_adapter,
            initial_candidate=Candidate(values={"prompt": "seed"}),
            optimize_fn=cast("OptimizeFn", optimize),
        ).optimize(
            trainset=[_Case("case")],
            config=GEPAConfig(
                reflection=ReflectionConfig(proposer=_propose),
                evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
            ),
        )

    assert _propose({"prompt": "seed"}, {}, ["prompt"]) == {"prompt": "seed"}
    assert (
        _Dataset()
        .evaluate(
            lambda case: case.name,
            max_concurrency=1,
            progress=False,
        )
        .cases
        == ()
    )


def test_optimizer_adapts_typed_defaults_to_the_backend_signature() -> None:
    adapter: PydanticGEPAAdapter[_Case, str, None] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(),
        task=lambda case: case.name,
        injections=[],
        objective=ScoreObjective(score_key="score"),
    )
    reflection_models: list[str | None] = []
    metric_limits: list[int | None] = []

    def versioned_backend(
        *,
        seed_candidate: dict[str, str],
        trainset: list[_Case],
        valset: list[_Case],
        adapter: PydanticGEPAAdapter[_Case, str, None],
        reflection_lm: str | None,
        max_metric_calls: int | None,
    ) -> _RawResult:
        assert seed_candidate == {"prompt": "seed"}
        assert [case.name for case in trainset] == ["train"]
        assert [case.name for case in valset] == ["validation"]
        assert adapter.objective.score_key == "score"
        reflection_models.append(reflection_lm)
        metric_limits.append(max_metric_calls)
        return _RawResult()

    optimizer = PydanticGEPAOptimizer(
        adapter=adapter,
        initial_candidate=Candidate(values={"prompt": "seed"}),
        optimize_fn=versioned_backend,
    )
    result = optimizer.optimize(
        trainset=[_Case("train")],
        valset=[_Case("validation")],
        config=GEPAConfig(
            reflection=ReflectionConfig(model="provider:model"),
            budget=BudgetConfig(max_metric_calls=3),
        ),
    )

    assert result.best_score == 1.0
    assert reflection_models == ["provider:model"]
    assert metric_limits == [3]

    with pytest.raises(ConfigurationError, match="reflection_lm_kwargs"):
        optimizer.optimize(
            trainset=[_Case("train")],
            valset=[_Case("validation")],
            config=GEPAConfig(
                reflection=ReflectionConfig(
                    model="provider:model",
                    model_kwargs={"temperature": 0.2},
                )
            ),
        )


def test_optimizer_uses_owned_backend_checkpoint_and_resumes_before_model_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_version(distribution: str) -> str:
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(optimizer_module, "version", missing_version)
    adapter: PydanticGEPAAdapter[_Case, str, None] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(),
        task=lambda case: case.name,
        injections=[],
        objective=ScoreObjective(score_key="score"),
    )
    attempts: list[dict[str, SerializableValue]] = []

    def optimize(**kwargs: SerializableValue) -> _RawResult:
        attempts.append(dict(kwargs))
        if len(attempts) == 1:
            raise RuntimeError("interrupted")
        return _RawResult()

    optimizer = PydanticGEPAOptimizer(
        adapter=adapter,
        initial_candidate=Candidate(values={"prompt": "seed"}),
        optimize_fn=cast("OptimizeFn", optimize),
    )
    directory = tmp_path / "optimizer"
    first = GEPAConfig(
        run=RunConfig(id="optimizer", directory=directory),
        evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
    )
    with pytest.raises(RuntimeError, match="interrupted"):
        optimizer.optimize(trainset=[_Case("case")], config=first)

    resumed = first.model_copy(update={"run": first.run.model_copy(update={"resume": "if_exists"})})
    result = optimizer.optimize(trainset=[_Case("case")], config=resumed)
    cached = optimizer.optimize(trainset=[_Case("case")], config=resumed)

    assert result.best_candidate.values == {"prompt": "optimized"}
    assert cached.model_dump() == result.model_dump()
    assert len(attempts) == 2
    assert attempts[1]["run_dir"] == str(directory / "backend")
    assert (directory / "result.json").is_file()

    with pytest.raises(RunStoreError, match="incompatible.*trainset"):
        optimizer.optimize(trainset=[_Case("different")], config=resumed)
    assert len(attempts) == 2


def test_typed_config_keeps_observers_out_of_backend_configuration() -> None:
    events: list[Event] = []
    config = GEPAConfig(
        run=RunConfig(id="events"),
        tracking=TrackingConfig(observers=(events.append,)),
    )
    assert config.to_backend_kwargs()["callbacks"] is None
    assert events == []


def test_optimizer_propagates_backend_failure_without_a_run_store() -> None:
    adapter: PydanticGEPAAdapter[_Case, str, None] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(),
        task=lambda case: case.name,
        injections=[],
        objective=ScoreObjective(score_key="score"),
    )

    def fail(**kwargs: SerializableValue) -> _RawResult:
        del kwargs
        raise RuntimeError("backend failed")

    optimizer = PydanticGEPAOptimizer(
        adapter=adapter,
        initial_candidate=Candidate(values={"prompt": "seed"}),
        optimize_fn=cast("OptimizeFn", fail),
    )
    with pytest.raises(RuntimeError, match="backend failed"):
        optimizer.optimize(
            trainset=[_Case("case")],
            allow_same_train_val=True,
        )


def _propose(
    candidate: dict[str, str],
    reflective_dataset: Mapping[str, Sequence[Mapping[str, SerializableValue]]],
    components: list[str],
) -> dict[str, str]:
    del reflective_dataset, components
    return candidate


def _raise_reflection(prompt: ReflectionPrompt) -> str:
    del prompt
    raise RuntimeError("reflection failed")


@dataclass(frozen=True, slots=True)
class _Opaque:
    name: str


@dataclass(frozen=True, slots=True)
class _Case:
    name: str


class _Report:
    cases: tuple[()] = ()
    failures: tuple[()] = ()


class _Dataset:
    name: str | None = None
    evaluators: tuple[None, ...] = ()

    def __init__(
        self,
        *,
        cases: Sequence[_Case] = (),
        evaluators: Sequence[None] = (),
    ) -> None:
        self.cases = tuple(cases)
        self.evaluators = tuple(evaluators)

    def evaluate(
        self,
        task: Callable[[_Case], str],
        *,
        max_concurrency: int,
        progress: bool,
    ) -> _Report:
        del task, max_concurrency, progress
        return _Report()


class _RawResult:
    best_candidate = {"prompt": "optimized"}
    best_score = 1.0


@dataclass(frozen=True, slots=True)
class _Usage:
    requests: int = 1
    input_tokens: int = 8
    output_tokens: int = 3
    tool_calls: int = 2
    details: dict[str, int] = field(default_factory=lambda: {"cached": 1})


class _AgentResult:
    output = "agent reflection"

    def usage(self) -> _Usage:
        return _Usage()


class _Agent:
    prompt: str | None = None
    settings: Mapping[str, int | float] | None = None

    def run_sync(
        self,
        prompt: str,
        *,
        deps: None,
        model_settings: Mapping[str, int | float] | None,
    ) -> _AgentResult:
        del deps
        self.prompt = prompt
        self.settings = model_settings
        return _AgentResult()


class _AgentFactory(_Agent):
    def __init__(self, model: str, *, output_type: type[str]) -> None:
        self.model = model
        self.output_type = output_type
