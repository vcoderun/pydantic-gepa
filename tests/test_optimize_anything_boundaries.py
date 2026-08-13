from __future__ import annotations as _annotations

import builtins
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gepa.oa
import gepa.oa.ensemble
import pytest
from gepa.oa.budget import BudgetExhausted
from gepa.oa.engine import Result
from gepa.oa.task import Task

import pydantic_gepa.experimental.optimize_anything.models as models_module
from pydantic_gepa import Candidate, GEPAConfig, OptimizationDependencyError
from pydantic_gepa.configuration import ConfigurationError
from pydantic_gepa.events import Event, _dispatcher
from pydantic_gepa.experimental.optimize_anything import (
    BestOf,
    CandidateCodec,
    Engine,
    EngineResult,
    EvaluationServer,
    OptimizationTask,
    OptimizeAnythingConfig,
    Sequential,
    Vote,
)
from pydantic_gepa.experimental.optimize_anything.adapter import (
    CandidateValue,
    EvaluationOutput,
    OptimizationState,
)
from pydantic_gepa.experimental.optimize_anything.backend import (
    BackendRuntime,
    ExecutionTracker,
    ObservedEngine,
    _adaptive_batch_evaluator,
    _adaptive_evaluator,
    _adaptive_schedule,
    _finite_score,
    _is_budget_exhausted,
    _nonnegative_float,
    _raw_results,
    configure_engine,
    execute,
)
from pydantic_gepa.recorder import GEPAEventBridge


@dataclass(slots=True)
class FixedEngine:
    name: str
    candidate: str = "winner"
    score: float = 1.0

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        del task, server
        return cast(
            "EngineResult",
            Result(best_candidate=self.candidate, best_score=self.score),
        )

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        del result, output_dir


@dataclass(slots=True)
class ExhaustingEngine:
    name: str = "exhausting"

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        del task, server
        raise BudgetExhausted("evaluation limit")

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        del result, output_dir


def _engine(implementation: FixedEngine | ExhaustingEngine, *, name: str | None = None) -> Engine:
    return Engine.custom(
        implementation,
        candidate_mode="text",
        max_evals=3,
        max_token_cost=1.0,
        name=name,
    )


def _runtime(
    engine: Engine,
    *,
    engine_index: int = 0,
    engines: Sequence[Engine] | None = None,
    observe_engine: bool = False,
) -> tuple[BackendRuntime, ExecutionTracker, list[Event]]:
    events: list[Event] = []
    dispatcher = _dispatcher(
        run_id="boundary",
        backend="optimize_anything",
        local_observers=(events.append,),
    )
    declared = tuple(engines or (engine,))
    tracker = ExecutionTracker(
        dispatcher=dispatcher,
        codec=CandidateCodec(seed=Candidate(values={"prompt": "seed"}), mode="text"),
        engines=declared,
        composition="single" if len(declared) == 1 else "best_of",
        pipeline_id="pipeline",
        step_id="step-0",
        parent_execution_id=None,
    )
    callback = GEPAEventBridge(
        run_id="boundary",
        on_event=dispatcher.emit,
        lifecycle="backend_only",
    )
    return (
        configure_engine(
            engine,
            engine_index=engine_index,
            name=f"engine-{engine_index}",
            run_dir=Path("run") / str(engine_index),
            callback=callback,
            tracker=tracker,
            observe_engine=observe_engine,
        ),
        tracker,
        events,
    )


def _evaluate(
    candidate: CandidateValue,
    example: str,
    opt_state: OptimizationState | None = None,
) -> EvaluationOutput:
    del opt_state
    text = candidate if isinstance(candidate, str) else candidate["prompt"]
    return float(text == example), {"candidate": text}


def _batch_evaluate(
    pairs: Sequence[tuple[CandidateValue, str]],
    *,
    opt_states: Sequence[OptimizationState | None] | None = None,
) -> list[EvaluationOutput]:
    del opt_states
    return [_evaluate(candidate, example) for candidate, example in pairs]


def test_builtin_engine_configuration_maps_typed_options_to_upstream() -> None:
    def callback(_event: Any) -> None:
        return None

    callback(None)
    gepa_config = GEPAConfig().model_copy(
        update={
            "tracking": GEPAConfig().tracking.model_copy(update={"backend_callbacks": (callback,)})
        }
    )
    engines = (
        Engine.gepa(gepa_config, candidate_mode="text", max_concurrency=2),
        Engine.autoresearch(
            handoffs=({"name": "review", "prompt": "Review it."},),
            max_evals=3,
        ),
        Engine.meta_harness(max_candidates_per_iteration=4, max_evals=3),
        Engine.best_of_n(max_samples=5, model_options={"timeout": 2}, max_evals=3),
        _engine(FixedEngine("custom")),
    )

    runtimes = [_runtime(engine)[0] for engine in engines]
    gepa_settings = runtimes[0].config.engine_config
    assert gepa_settings["engine"]["max_workers"] == 2
    assert gepa_settings["callbacks"][-1].__class__ is GEPAEventBridge
    assert runtimes[1].config.engine_config["handoffs"] == [
        {"name": "review", "prompt": "Review it."}
    ]
    assert runtimes[2].config.engine_config["max_candidates_per_iter"] == 4
    assert runtimes[3].config.engine_config["max_n"] == 5
    assert runtimes[3].config.engine_config["lm_kwargs"] == {"timeout": 2}
    assert runtimes[4].config.engine_config == {}
    direct = FixedEngine("direct")
    direct_result = direct.run(
        cast("OptimizationTask", None),
        cast("EvaluationServer", None),
    )
    direct.process_result(direct_result, None)
    ExhaustingEngine().process_result(direct_result, None)


@pytest.mark.parametrize(
    ("engine", "message"),
    [
        (
            Engine.gepa(candidate_mode="text").model_copy(update={"gepa_config": None}),
            "no GEPAConfig",
        ),
        (
            Engine.autoresearch().model_copy(update={"autoresearch_options": None}),
            "no options",
        ),
        (
            Engine.meta_harness().model_copy(update={"meta_harness_options": None}),
            "no options",
        ),
        (
            Engine.best_of_n().model_copy(update={"best_of_n_options": None}),
            "no options",
        ),
        (
            _engine(FixedEngine("custom")).model_copy(update={"custom_instance": None}),
            "no instance",
        ),
    ],
)
def test_malformed_engine_models_fail_at_the_upstream_boundary(
    engine: Engine,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _runtime(engine)


def test_observed_builtin_engine_uses_public_factory_and_reports_missing_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = Engine.gepa(candidate_mode="text", max_evals=2)

    class Factory:
        def __init__(self, _config: Any) -> None:
            pass

        name = "factory"

        def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
            del task, server
            return cast("EngineResult", Result(best_candidate="winner", best_score=1.0))

        def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
            del result, output_dir

    monkeypatch.setattr(gepa.oa, "get_engine_cls", lambda _kind: Factory)
    runtime, _, _ = _runtime(engine, observe_engine=True)
    assert isinstance(runtime.config.engine, ObservedEngine)
    factory = Factory(None)
    factory_result = factory.run(
        cast("OptimizationTask", None),
        cast("EvaluationServer", None),
    )
    factory.process_result(factory_result, None)

    monkeypatch.delattr(gepa.oa, "get_engine_cls")
    with pytest.raises(OptimizationDependencyError, match="engine API"):
        _runtime(engine, observe_engine=True)


def test_budget_exhaustion_is_deferred_until_upstream_returns_its_result() -> None:
    engine = _engine(ExhaustingEngine())
    runtime, tracker, events = _runtime(engine)
    observed = ObservedEngine(
        cast("ExhaustingEngine", engine.custom_instance),
        tracker=tracker,
        engine_index=0,
        name="exhausting",
    )
    task = Task(
        name="budget",
        seed_candidate="seed",
        objective="",
        background="",
        train_set=["winner"],
        val_set=["winner"],
    )
    with pytest.raises(BudgetExhausted, match="evaluation limit"):
        observed.run(task, cast("EvaluationServer", None))

    result = cast(
        "EngineResult",
        Result(
            best_candidate="seed",
            best_score=0.0,
            metadata={"budget": {"exhausted": True}},
        ),
    )
    invocation = tracker.resolve(result, engine_index=runtime.engine_index)
    assert invocation.execution_id.endswith("branch-0")
    assert [event.kind for event in events].count("stage.completed") == 1
    assert "stage.failed" not in [event.kind for event in events]

    fresh_runtime, fresh_tracker, fresh_events = _runtime(engine)
    unresolved = cast(
        "EngineResult",
        Result(best_candidate="winner", best_score=1.0),
    )
    assert fresh_tracker.resolve(unresolved, engine_index=fresh_runtime.engine_index).branch_id == (
        "branch-0"
    )
    assert fresh_events[-1].kind == "stage.completed"


def test_observed_engine_reports_non_budget_failures() -> None:
    class FailingEngine:
        name = "failing"

        def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
            del task, server
            raise RuntimeError("engine failed")

        def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
            del result, output_dir

    implementation = FailingEngine()
    engine = Engine.custom(
        implementation,
        candidate_mode="text",
        max_evals=1,
    )
    _, tracker, events = _runtime(engine)
    observed = ObservedEngine(
        implementation,
        tracker=tracker,
        engine_index=0,
        name="failing",
    )
    task = Task(
        name="failure",
        seed_candidate="seed",
        objective="",
        background="",
        train_set=["winner"],
        val_set=["winner"],
    )
    with pytest.raises(RuntimeError, match="engine failed"):
        observed.run(task, cast("EvaluationServer", None))
    observed.process_result(
        cast("EngineResult", Result(best_candidate="seed", best_score=0.0)),
        None,
    )
    assert events[-1].kind == "stage.failed"


def test_adaptive_evaluators_work_with_and_without_an_active_engine() -> None:
    engine = _engine(FixedEngine("adaptive"))
    runtime, tracker, events = _runtime(engine)
    scalar = _adaptive_evaluator(_evaluate, (runtime,))
    grouped = _adaptive_batch_evaluator(_batch_evaluate, (runtime,))

    assert scalar("winner", "winner")[0] == 1.0
    assert grouped([("seed", "winner")])[0][0] == 0.0
    invocation = tracker.start(0, "seed")
    assert scalar("winner", "winner")[0] == 1.0
    assert grouped([("winner", "winner")])[0][0] == 1.0
    tracker.fail(invocation, RuntimeError("done"))
    assert events[-1].engine_execution_id == invocation.execution_id


def test_composition_execution_rejects_custom_runner_and_empty_vote_validation() -> None:
    engine = _engine(FixedEngine("one"))
    runtime, _, _ = _runtime(engine)
    codec = CandidateCodec(seed=Candidate(values={"prompt": "seed"}), mode="text")

    with pytest.raises(ConfigurationError, match="only execute Single"):
        execute(
            Sequential(engines=(engine,)),
            codec=codec,
            evaluate=_evaluate,
            batch_evaluate=_batch_evaluate,
            trainset=["winner"],
            valset=["winner"],
            objective=None,
            background=None,
            runtimes=[runtime],
            optimize_fn=cast("Any", lambda: None),
        )
    with pytest.raises(ConfigurationError, match="validation example"):
        execute(
            Vote(engines=(engine,)),
            codec=codec,
            evaluate=_evaluate,
            batch_evaluate=_batch_evaluate,
            trainset=["winner"],
            valset=[],
            objective=None,
            background=None,
            runtimes=[runtime],
            optimize_fn=None,
        )


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"all_results": [], "vote_scores": "bad"}, "aligned selection scores"),
        ({"all_results": [], "vote_scores": [math.nan]}, "non-finite"),
    ],
)
def test_vote_rejects_malformed_upstream_selection_metadata(
    monkeypatch: pytest.MonkeyPatch,
    metadata: dict[str, Any],
    message: str,
) -> None:
    engine = _engine(FixedEngine("vote"))
    runtime, _, _ = _runtime(engine)
    raw = Result(best_candidate="winner", best_score=1.0, metadata=metadata)
    monkeypatch.setattr(
        gepa.oa.ensemble,
        "optimize_vote_with_server",
        lambda *_args, **_kwargs: raw,
    )

    with pytest.raises(ConfigurationError, match=message):
        execute(
            Vote(engines=(engine,)),
            codec=CandidateCodec(seed=Candidate(values={"prompt": "seed"}), mode="text"),
            evaluate=_evaluate,
            batch_evaluate=_batch_evaluate,
            trainset=["winner"],
            valset=["winner"],
            objective=None,
            background=None,
            runtimes=[runtime],
            optimize_fn=None,
        )


def test_composition_prefers_tracker_results_when_upstream_metadata_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _engine(FixedEngine("first"))
    second = _engine(FixedEngine("second"))
    first_runtime, tracker, _events = _runtime(first, engines=(first, second))
    callback = GEPAEventBridge(
        run_id="boundary",
        on_event=tracker.dispatcher.emit,
        lifecycle="backend_only",
    )
    second_runtime = configure_engine(
        second,
        engine_index=1,
        name="engine-1",
        run_dir=Path("run/1"),
        callback=callback,
        tracker=tracker,
        observe_engine=False,
    )
    first_result = cast(
        "EngineResult",
        Result(best_candidate="first", best_score=0.5),
    )
    second_result = cast(
        "EngineResult",
        Result(best_candidate="second", best_score=1.0),
    )
    tracker.complete(tracker.start(0, "seed"), first_result)
    tracker.complete(tracker.start(1, "seed"), second_result)
    incomplete = Result(
        best_candidate="second",
        best_score=1.0,
        metadata={"all_results": [first_result]},
    )
    monkeypatch.setattr(
        gepa.oa.ensemble,
        "optimize_best_of_with_server",
        lambda *_args, **_kwargs: incomplete,
    )

    outcome = execute(
        BestOf(engines=(first, second)),
        codec=CandidateCodec(seed=Candidate(values={"prompt": "seed"}), mode="text"),
        evaluate=_evaluate,
        batch_evaluate=_batch_evaluate,
        trainset=["winner"],
        valset=["winner"],
        objective=None,
        background=None,
        runtimes=[first_runtime, second_runtime],
        optimize_fn=None,
    )

    assert outcome.results == (first_result, second_result)
    assert outcome.engine_indices == (0, 1)


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        (None, "omitted its schedule"),
        (["bad"], "schedule is malformed"),
        ([{"engine_idx": True}], "no engine index"),
        (
            [
                {
                    "engine_idx": 0,
                    "eval_start": True,
                    "eval_end": 1,
                    "eval_delta": 1,
                    "engine": "one",
                    "improved": True,
                }
            ],
            "invalid eval counts",
        ),
        (
            [
                {
                    "engine_idx": 0,
                    "eval_start": 0,
                    "eval_end": 1,
                    "eval_delta": 1,
                    "engine": 1,
                    "improved": True,
                }
            ],
            "missing stage evidence",
        ),
        ([], "does not align"),
    ],
)
def test_adaptive_schedule_rejects_malformed_upstream_metadata(
    schedule: Any,
    message: str,
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        _adaptive_schedule(schedule, expected=1)


def test_upstream_metadata_helpers_filter_dynamic_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    valid = Result(best_candidate="winner", best_score=1.0)
    assert _raw_results("not-a-list") == []
    assert _raw_results([valid, "bad"]) == [valid]
    assert _finite_score(True) is None
    assert _finite_score(math.inf) is None
    assert _nonnegative_float(-1) is None
    assert _is_budget_exhausted(BudgetExhausted("limit")) is True

    original_import = builtins.__import__

    def missing_engine_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name in {"gepa.oa.engine", "gepa.oa.budget"}:
            raise ImportError("missing engine")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_engine_import)
    assert missing_engine_import("math").__name__ == "math"
    assert _raw_results([valid]) == []
    assert _is_budget_exhausted(RuntimeError("not budget")) is False


def test_runtime_declaration_falls_back_to_a_stable_python_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(FixedEngine("opaque"))

    def unavailable_source(_target: Any) -> str:
        raise OSError("source unavailable")

    monkeypatch.setattr(models_module.inspect, "getsource", unavailable_source)
    declaration = engine.declaration()
    assert declaration["custom_instance"] == {
        "python": f"{FixedEngine.__module__}.{FixedEngine.__qualname__}"
    }


def test_single_and_composition_dependency_errors_are_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine(FixedEngine("dependency"))
    runtime, _, _ = _runtime(engine)
    codec = CandidateCodec(seed=Candidate(values={"prompt": "seed"}), mode="text")
    original_import = builtins.__import__

    def missing_omni_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name in {"gepa.optimize_anything", "gepa.oa.budget"}:
            raise ImportError("missing omni")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_omni_import)
    assert missing_omni_import("math").__name__ == "math"
    with pytest.raises(OptimizationDependencyError, match="Optimize Anything Omni"):
        execute(
            cast("Any", OptimizeAnythingConfig(engine=engine).active_composition),
            codec=codec,
            evaluate=_evaluate,
            batch_evaluate=_batch_evaluate,
            trainset=["winner"],
            valset=["winner"],
            objective=None,
            background=None,
            runtimes=[runtime],
            optimize_fn=None,
        )
    with pytest.raises(OptimizationDependencyError, match="Optimize Anything Omni"):
        execute(
            BestOf(engines=(engine,)),
            codec=codec,
            evaluate=_evaluate,
            batch_evaluate=_batch_evaluate,
            trainset=["winner"],
            valset=["winner"],
            objective=None,
            background=None,
            runtimes=[runtime],
            optimize_fn=None,
        )
