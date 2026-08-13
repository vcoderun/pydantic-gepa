from __future__ import annotations as _annotations

import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any, cast

import pytest
from gepa.oa.engine import Result
from pydantic import ValidationError

import pydantic_gepa.experimental.optimize_anything.backend as backend_module
import pydantic_gepa.experimental.optimize_anything.execution as execution_module
import pydantic_gepa.experimental.optimize_anything.results as result_module
from pydantic_gepa import (
    Candidate,
    CandidateComponent,
    CandidateContext,
    ComponentCatalog,
    DataSplit,
    DerivedValueInjection,
    Example,
    Optimization,
    PydanticEvaluator,
    RunConfig,
)
from pydantic_gepa.configuration import ConfigurationError, GEPAConfig
from pydantic_gepa.configuration.models import TrackingConfig
from pydantic_gepa.errors import (
    CandidateComponentError,
    EvaluationHarnessError,
    RunStoreError,
)
from pydantic_gepa.events import (
    BackendProgress,
    BudgetUpdated,
    EvaluationStarted,
    Event,
    RunCompleted,
    RunStarted,
    StageCompleted,
    _dispatcher,
)
from pydantic_gepa.examples import EvalCaseView
from pydantic_gepa.experimental.optimize_anything import (
    AdaptiveSequential,
    AutoResearchOptions,
    BestOf,
    CandidateCodec,
    Engine,
    EngineResult,
    EvaluationServer,
    OptimizationTask,
    OptimizeAnythingConfig,
    Parallel,
    Pipeline,
    PydanticOptimizeAnythingAdapter,
    Sequential,
    Single,
    Vote,
)


@dataclass(slots=True)
class DeterministicEngine:
    name: str
    candidate: str
    score: float
    evaluate: bool = False
    fail_after: int | None = None
    cancel: bool = False
    runs: int = 0
    seeds: list[str | dict[str, str] | None] = field(default_factory=list)
    visible_splits: list[tuple[int, int, int]] = field(default_factory=list)
    processed_outputs: list[Path | None] = field(default_factory=list)

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        self.runs += 1
        self.seeds.append(task.seed_candidate)
        self.visible_splits.append(
            (
                len(task.train_set or []),
                len(task.val_set or []),
                len(task.test_set or []),
            )
        )
        if self.fail_after is not None and self.runs > self.fail_after:
            raise RuntimeError(f"{self.name} interrupted")
        if self.cancel:
            raise KeyboardInterrupt(f"{self.name} cancelled")
        score = self.score
        if self.evaluate:
            score, _ = server.evaluate_examples(self.candidate, split="train")
        return cast(
            "EngineResult",
            Result(
                best_candidate=self.candidate,
                best_score=score,
                metadata={
                    "adapter_cost": 0.25,
                    "work_dir": f"work/{self.name}",
                },
            ),
        )

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        del result
        self.processed_outputs.append(output_dir)


@dataclass(slots=True)
class ComponentEngine:
    name: str = "components"

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        del server
        seed = task.seed_candidate
        if not isinstance(seed, dict):
            raise TypeError("component engine requires a mapping seed")
        candidate = dict(seed)
        candidate["prompt"] = "improved"
        return ComponentResult(best_candidate=candidate, best_score=0.8)

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        del result, output_dir


@dataclass(slots=True)
class ComponentResult:
    best_candidate: str | dict[str, str]
    best_score: float
    total_evals: int = 0
    eval_log: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _optimization(
    *,
    expected: str = "winner",
    with_test: bool = False,
    components: tuple[CandidateComponent, ...] | None = None,
) -> Optimization[str, str, dict[str, str]]:
    prompt = CandidateContext[str]("candidate.prompt")
    active_components = components or (
        CandidateComponent(name="prompt", initial_text="seed", kind="system_prompt"),
    )
    data = DataSplit.from_sets(
        train=[Example(inputs="train", expected_output=expected, id="train")],
        validation=[Example(inputs="validation", expected_output=expected, id="validation")],
        test=([Example(inputs="test", expected_output=expected, id="test")] if with_test else []),
    )
    return Optimization.from_examples(
        data=data,
        task=lambda _value: prompt.require(),
        score=lambda ctx: 1.0 if ctx.output == ctx.expected_output else 0.0,
        components=ComponentCatalog.from_components(active_components),
        injections=[
            DerivedValueInjection(
                component="prompt",
                context=prompt,
                required_components=("prompt",),
                derive_value=lambda candidate: candidate["prompt"],
            )
        ],
        backend="optimize_anything",
        optimization_objective="Return the expected value.",
    )


def _engine(
    implementation: DeterministicEngine,
    *,
    output_dir: Path | None = None,
) -> Engine:
    return Engine.custom(
        implementation,
        candidate_mode="text",
        max_evals=4,
        max_token_cost=1.0,
        output_dir=output_dir,
    )


def test_single_custom_engine_uses_real_omni_contract_and_heldout_boundary(
    tmp_path: Path,
) -> None:
    implementation = DeterministicEngine("single", "winner", 0.9)
    result = _optimization(with_test=True).optimize(
        config=OptimizeAnythingConfig(
            engine=_engine(implementation, output_dir=tmp_path / "output")
        )
    )

    assert result.best_candidate.values == {"prompt": "winner"}
    assert result.scores.baseline_test == 0.0
    assert result.scores.test == 1.0
    assert result.budget.heldout_evaluation_calls == 2
    assert implementation.visible_splits == [(1, 1, 0)]
    assert implementation.processed_outputs == [tmp_path / "output"]
    assert result.composition is not None
    artifacts = {
        artifact.kind: artifact.path for artifact in result.composition.engine_runs[0].artifacts
    }
    assert artifacts == {
        "output": str(tmp_path / "output"),
        "workspace": "work/single",
    }
    assert "raw_gepa_result" not in result.stable_dump()


def test_component_engine_preserves_untargeted_candidate_values() -> None:
    components = (
        CandidateComponent(name="prompt", initial_text="seed"),
        CandidateComponent(name="tool", initial_text="frozen"),
    )
    optimization = _optimization(components=components)
    result = optimization.optimize(
        config=OptimizeAnythingConfig(
            engine=Engine.custom(
                ComponentEngine(),
                candidate_mode="components",
                max_evals=2,
            )
        )
    )

    assert result.best_candidate.values == {"prompt": "improved", "tool": "frozen"}
    with pytest.raises(TypeError, match="requires a mapping seed"):
        ComponentEngine().run(
            cast("OptimizationTask", type("TextTask", (), {"seed_candidate": "text"})()),
            cast("EvaluationServer", None),
        )


def test_text_engines_require_one_explicit_component() -> None:
    components = (
        CandidateComponent(name="prompt", initial_text="seed"),
        CandidateComponent(name="tool", initial_text="frozen"),
    )
    optimization = _optimization(components=components)
    engine = _engine(DeterministicEngine("text", "winner", 1.0))

    with pytest.raises(CandidateComponentError, match="exactly one component"):
        optimization.optimize(config=OptimizeAnythingConfig(engine=engine))

    result = optimization.optimize(config=OptimizeAnythingConfig(engine=engine, component="prompt"))
    assert result.best_candidate.values == {"prompt": "winner", "tool": "frozen"}


def test_sequential_is_monotonic_and_records_parent_lineage() -> None:
    first = DeterministicEngine("first", "winner", 0.9)
    regression = DeterministicEngine("regression", "worse", 0.1)
    final = DeterministicEngine("final", "final", 1.0)
    result = _optimization().optimize(
        config=OptimizeAnythingConfig(
            composition=Sequential(engines=(_engine(first), _engine(regression), _engine(final)))
        )
    )

    assert regression.seeds == ["winner"]
    assert final.seeds == ["winner"]
    assert result.best_candidate.values == {"prompt": "final"}
    assert result.composition is not None
    runs = result.composition.engine_runs
    assert runs[1].parent_execution_id == runs[0].execution_id
    assert runs[2].parent_execution_id == runs[0].execution_id


def test_parallel_preserves_declaration_order_without_selecting_a_branch() -> None:
    result = _optimization().optimize(
        config=OptimizeAnythingConfig(
            composition=Parallel(
                engines=(
                    _engine(DeterministicEngine("slow", "left", 0.2)),
                    _engine(DeterministicEngine("fast", "right", 0.8)),
                ),
                max_workers=2,
            )
        )
    )

    assert result.composition is not None
    assert [run.engine for run in result.composition.engine_runs] == ["slow", "fast"]
    assert result.composition.selections == ()
    assert result.final_candidate == result.composition.engine_runs[0].input_candidate


def test_best_of_and_vote_keep_search_and_selection_scores_distinct() -> None:
    best_of = _optimization().optimize(
        config=OptimizeAnythingConfig(
            composition=BestOf(
                engines=(
                    _engine(DeterministicEngine("left", "wrong", 0.2)),
                    _engine(DeterministicEngine("right", "winner", 0.8)),
                )
            )
        )
    )
    vote = _optimization().optimize(
        config=OptimizeAnythingConfig(
            composition=Vote(
                engines=(
                    _engine(DeterministicEngine("search-best", "wrong", 0.9)),
                    _engine(DeterministicEngine("vote-best", "winner", 0.1)),
                )
            )
        )
    )

    assert best_of.best_candidate.values == {"prompt": "winner"}
    assert best_of.composition is not None
    assert [run.engine for run in best_of.composition.engine_runs] == ["left", "right"]
    assert best_of.composition.selections[0].method == "best_score"
    assert best_of.composition.selections[0].contender_execution_ids == tuple(
        run.execution_id for run in best_of.composition.engine_runs
    )
    assert vote.best_candidate.values == {"prompt": "winner"}
    assert vote.composition is not None
    vote_runs = vote.composition.engine_runs
    assert [run.search_score for run in vote_runs] == [0.9, 0.1]
    assert [run.selection_score for run in vote_runs] == [0.0, 1.0]
    assert vote.budget.final_rescore_calls == 2


def test_adaptive_sequential_records_shared_budget_schedule() -> None:
    first = DeterministicEngine("first", "winner", 0.0, evaluate=True)
    second = DeterministicEngine("second", "winner", 0.0, evaluate=True)
    result = _optimization().optimize(
        config=OptimizeAnythingConfig(
            composition=AdaptiveSequential(
                engines=(_engine(first), _engine(second)),
                plateau_evals=1,
                max_evals=2,
                patience=1,
            )
        )
    )

    assert result.composition is not None
    assert result.budget.evaluation_calls == 2
    assert result.budget.evaluation_call_limit == 2
    assert result.composition.stop_reason == "budget_exhausted"
    assert [item.evaluation_calls for item in result.composition.adaptive_schedule] == [1, 1]
    assert sum(run.budget.evaluation_calls or 0 for run in result.composition.engine_runs) == 2


def test_omni_pipeline_continues_from_winner_and_resumes_completed_steps(
    tmp_path: Path,
) -> None:
    explorer_a = DeterministicEngine("explorer-a", "wrong", 0.2)
    explorer_b = DeterministicEngine("explorer-b", "winner", 0.9)
    continuation = DeterministicEngine("continuation", "final", 1.0, fail_after=0)
    optimization = _optimization()
    run = RunConfig(id="omni", directory=tmp_path / "run")
    composition = Pipeline(
        steps=(
            BestOf(engines=(_engine(explorer_a), _engine(explorer_b))),
            Single(engine=_engine(continuation)),
        )
    )
    with pytest.raises(RuntimeError, match="continuation interrupted"):
        optimization.optimize(config=OptimizeAnythingConfig(composition=composition, run=run))

    continuation.fail_after = None
    resumed = run.model_copy(update={"resume": "if_exists"})
    result = optimization.optimize(
        config=OptimizeAnythingConfig(composition=composition, run=resumed)
    )

    assert explorer_a.runs == 1
    assert explorer_b.runs == 1
    assert continuation.seeds == ["winner", "winner"]
    assert result.best_candidate.values == {"prompt": "final"}
    assert result.composition is not None
    assert [step.kind for step in result.composition.steps] == ["best_of", "single"]
    first_step, second_step = result.composition.steps
    assert second_step.input_candidate == first_step.output_candidate
    assert (tmp_path / "run" / "stages" / "step-0.json").is_file()


def test_pipeline_resume_rejects_changed_objective_and_engine_declaration(
    tmp_path: Path,
) -> None:
    optimization = _optimization()
    engine = DeterministicEngine("engine", "winner", 1.0)
    run = RunConfig(id="resume", directory=tmp_path / "run")
    config = OptimizeAnythingConfig(
        composition=Pipeline(steps=(Single(engine=_engine(engine)),)),
        objective="first objective",
        run=run,
    )
    optimization.optimize(config=config)

    resumed = run.model_copy(update={"resume": "if_exists"})
    with pytest.raises(RunStoreError, match="objective"):
        optimization.optimize(
            config=config.model_copy(update={"objective": "changed objective", "run": resumed})
        )


def test_engine_and_composition_configuration_fails_before_execution() -> None:
    text = _engine(DeterministicEngine("text", "winner", 1.0))
    components = Engine.custom(ComponentEngine(), candidate_mode="components", max_evals=2)

    with pytest.raises(ValidationError, match="one candidate mode"):
        OptimizeAnythingConfig(composition=BestOf(engines=(text, components)))
    with pytest.raises(ValidationError, match="cannot contain Parallel"):
        Pipeline(steps=(Parallel(engines=(text,)),))
    with pytest.raises(ValidationError, match="requires max_evals or max_token_cost"):
        OptimizeAnythingConfig(
            engine=Engine.custom(
                DeterministicEngine("unbounded", "winner", 1.0),
                candidate_mode="text",
                max_evals=None,
                max_token_cost=None,
            )
        )
    with pytest.raises(ValidationError, match="AdaptiveSequential requires"):
        OptimizeAnythingConfig(
            composition=AdaptiveSequential(
                engines=(
                    Engine.custom(
                        DeterministicEngine("unbounded", "winner", 1.0),
                        candidate_mode="text",
                        max_evals=None,
                        max_token_cost=None,
                    ),
                ),
                plateau_evals=1,
                max_evals=None,
            )
        )


def test_builtin_engine_declarations_are_typed_and_stable(tmp_path: Path) -> None:
    def callback(*_values: Any) -> None:
        return None

    callback()

    gepa = Engine.gepa(
        GEPAConfig().model_copy(
            update={
                "tracking": GEPAConfig().tracking.model_copy(
                    update={"backend_callbacks": (callback,)}
                )
            }
        ),
        candidate_mode="text",
    )
    engines = (
        gepa,
        Engine.autoresearch(
            model="claude-test",
            handoffs=({"name": "review", "prompt": "Review it."},),
            output_dir=tmp_path / "autoresearch",
        ),
        Engine.meta_harness(
            model="claude-test",
            max_iterations=2,
            max_candidates_per_iteration=4,
        ),
        Engine.best_of_n(
            model="test:model",
            max_samples=3,
            model_options={"timeout": 1},
        ),
    )

    declarations = [OptimizeAnythingConfig(engine=engine).declaration() for engine in engines]
    engine_kinds: list[str] = []
    for declaration in declarations:
        engine = declaration["engine"]
        assert isinstance(engine, dict)
        kind = engine["kind"]
        assert isinstance(kind, str)
        engine_kinds.append(kind)
    assert engine_kinds == [
        "gepa",
        "autoresearch",
        "meta_harness",
        "best_of_n",
    ]
    assert declarations[0] == OptimizeAnythingConfig(engine=gepa).declaration()
    assert "source_sha256" in json.dumps(declarations[0], sort_keys=True)


def test_candidate_codec_rejects_lossy_or_mismatched_values() -> None:
    seed = Candidate(values={"prompt": "seed", "tool": "frozen"})
    with pytest.raises(CandidateComponentError, match="not present"):
        CandidateCodec(seed=seed, mode="text", component="missing")

    components = CandidateCodec(seed=seed, mode="components")
    with pytest.raises(CandidateComponentError, match="component-mode"):
        components.decode("plain text")

    text = CandidateCodec(seed=seed, mode="text", component="prompt")
    with pytest.raises(CandidateComponentError, match="component mapping"):
        text.decode({"prompt": "changed"})
    text.component = None
    with pytest.raises(RuntimeError, match="has no component"):
        text.encode_seed()


def test_optimize_anything_configuration_rejects_ambiguous_targets() -> None:
    engine = _engine(DeterministicEngine("configured", "winner", 1.0))
    with pytest.raises(ValidationError, match="Provide engine or composition"):
        OptimizeAnythingConfig()
    with pytest.raises(ValidationError, match="cannot be combined"):
        OptimizeAnythingConfig(engine=engine, composition=Single(engine=engine))
    with pytest.raises(ValidationError, match="component cannot be empty"):
        OptimizeAnythingConfig(engine=engine, component="")
    with pytest.raises(ValidationError, match="requires its typed configuration"):
        Engine(
            kind="autoresearch",
            name="missing-options",
            candidate_mode="text",
            max_evals=1,
        )
    with pytest.raises(ValidationError, match="exactly one engine kind"):
        Engine(
            kind="custom",
            name="ambiguous",
            candidate_mode="text",
            max_evals=1,
            custom_instance=DeterministicEngine("custom", "winner", 1.0),
            autoresearch_options=AutoResearchOptions(),
        )

    invalid = OptimizeAnythingConfig.model_construct()
    with pytest.raises(RuntimeError, match="has no engine"):
        _ = invalid.active_composition


def test_optimize_anything_batch_evaluation_preserves_alignment() -> None:
    optimization = _optimization()
    adapter = cast(
        "PydanticOptimizeAnythingAdapter[EvalCaseView[str, str, dict[str, str]], str, PydanticEvaluator[str, str, dict[str, str]]]",
        optimization.adapter,
    )
    case = optimization.trainset[0]
    with pytest.raises(EvaluationHarnessError, match="align one-to-one"):
        adapter.batch_evaluator(
            [({"prompt": "seed"}, case)],
            opt_states=(),
        )
    outputs = adapter.batch_evaluator([({"prompt": "winner"}, case), ({"prompt": "seed"}, case)])
    assert [score for score, _ in outputs] == [1.0, 0.0]


def test_execution_tracker_correlates_repeated_and_deferred_engine_runs() -> None:
    events: list[Event] = []
    dispatcher = _dispatcher(
        run_id="tracker",
        backend="optimize_anything",
        local_observers=(events.append,),
    )
    engine = _engine(DeterministicEngine("tracked", "winner", 1.0))
    codec = CandidateCodec(seed=Candidate(values={"prompt": "seed"}), mode="text")
    tracker = backend_module.ExecutionTracker(
        dispatcher=dispatcher,
        codec=codec,
        engines=(engine,),
        composition="sequential",
        pipeline_id="pipeline",
        step_id="step-0",
        parent_execution_id="parent",
    )

    with tracker.evaluation_scope(0):
        dispatcher.emit(BackendProgress(run_id="tracker", name="outside-engine"))
    seed_invocation = tracker.start(0, None)
    first_result = cast(
        "EngineResult",
        Result(best_candidate="winner", best_score=0.8),
    )
    tracker.complete(seed_invocation, first_result)
    assert tracker.resolve(first_result, engine_index=0) == seed_invocation

    repeated = tracker.start(0, "winner")
    assert repeated.branch_id == "branch-0-run-1"
    assert repeated.parent_execution_id == seed_invocation.execution_id
    tracker.defer_budget_completion(repeated)
    deferred_result = cast(
        "EngineResult",
        Result(best_candidate="winner", best_score=0.9),
    )
    assert tracker.resolve(deferred_result, engine_index=0) == repeated

    unknown = tracker.start(0, "new candidate")
    assert unknown.input_candidate.values == {"prompt": "new candidate"}
    tracker.fail(unknown, RuntimeError("stopped"))
    assert [event.kind for event in events].count("stage.failed") == 1


def test_result_normalization_rejects_missing_runs_and_invalid_pipeline_steps() -> None:
    seed = Candidate(values={"prompt": "seed"})
    with pytest.raises(ValueError, match="no engine runs"):
        result_module.composition_result(
            kind="single",
            pipeline_id="pipeline",
            runs=(),
            initial_candidate=seed,
            run_id="run",
        )

    engine = _engine(DeterministicEngine("result", "winner", 1.0))
    codec = CandidateCodec(seed=seed, mode="text")
    normalized = result_module.engine_result(
        cast(
            "EngineResult",
            Result(
                best_candidate="winner",
                best_score=1.0,
                metadata={
                    "wall_time": -1,
                    "ignored": {"not": "durable"},
                    "engine": "custom",
                },
            ),
        ),
        engine=engine,
        codec=codec,
        input_candidate=seed,
        execution_id="engine-0",
        parent_execution_id=None,
        pipeline_id="pipeline",
        step_id="step-0",
        branch_id="branch-0",
    )
    malformed_step = result_module.composition_result(
        kind="single",
        pipeline_id="step-pipeline",
        runs=(normalized,),
        initial_candidate=seed,
        run_id="step-run",
    ).model_copy(update={"stage_id": "step-0", "composition": None})
    with pytest.raises(ValueError, match="stage and composition summaries"):
        result_module.composition_result(
            kind="pipeline",
            pipeline_id="pipeline",
            runs=(normalized,),
            initial_candidate=seed,
            run_id="run",
            step_results=(malformed_step,),
        )
    assert normalized.duration_seconds is None
    assert normalized.reported == {"engine": "custom"}


def test_engine_lifecycle_wraps_correlated_evaluation_evidence() -> None:
    events: list[Event] = []
    implementation = DeterministicEngine("measured", "winner", 0.0, evaluate=True)

    _optimization().optimize(
        config=OptimizeAnythingConfig(
            engine=_engine(implementation),
            tracking=TrackingConfig(observers=(events.append,)),
        )
    )

    kinds = [event.kind for event in events]
    engine_start = next(
        index
        for index, event in enumerate(events)
        if event.kind == "stage.started" and event.stage_kind == "engine"
    )
    evaluation_start = kinds.index("evaluation.started")
    normalized = kinds.index("candidate.normalized")
    engine_end = next(
        index
        for index, event in enumerate(events)
        if event.kind == "stage.completed" and event.stage_kind == "engine"
    )
    assert engine_start < evaluation_start < normalized < engine_end
    assert [event.sequence for event in events] == list(range(len(events)))

    engine_event = events[engine_start]
    evaluation_events = [
        event
        for event in events
        if event.kind
        in {
            "evaluation.started",
            "case.evaluated",
            "metric.completed",
            "evaluation.completed",
            "candidate.evaluated",
        }
    ]
    assert evaluation_events
    assert all(event.engine == "measured" for event in evaluation_events)
    assert all(event.branch_id == "branch-0" for event in evaluation_events)
    assert all(
        event.engine_execution_id == engine_event.engine_execution_id for event in evaluation_events
    )
    evaluation_event = events[evaluation_start]
    assert isinstance(evaluation_event, EvaluationStarted)
    assert evaluation_event.candidate is not None
    assert evaluation_event.candidate.values == {"prompt": "winner"}

    completed_engine = events[engine_end]
    assert isinstance(completed_engine, StageCompleted)
    assert completed_engine.budget is not None
    assert completed_engine.budget.evaluation_calls == 1
    assert completed_engine.budget.evaluation_call_limit == 4
    assert completed_engine.budget.optimizer_cost == 0.25
    assert completed_engine.budget.evaluation_cost == 0.0
    assert completed_engine.budget.total_cost == 0.25

    budget_event = next(event for event in events if event.kind == "budget.updated")
    assert isinstance(budget_event, BudgetUpdated)
    assert budget_event.optimizer_cost == 0.25
    assert budget_event.evaluation_cost == 0.0
    assert budget_event.total_cost == 0.25
    run_event = events[-1]
    assert isinstance(run_event, RunCompleted)
    assert run_event.budget is not None
    assert run_event.budget.total_cost == 0.25


def test_engine_failure_and_cancellation_emit_one_terminal_run_event() -> None:
    failed_events: list[Event] = []
    with pytest.raises(RuntimeError, match="failed interrupted"):
        _optimization().optimize(
            config=OptimizeAnythingConfig(
                engine=_engine(DeterministicEngine("failed", "winner", 1.0, fail_after=0)),
                tracking=TrackingConfig(observers=(failed_events.append,)),
            )
        )
    assert [event.kind for event in failed_events].count("stage.failed") == 2
    assert [event.kind for event in failed_events].count("run.failed") == 1
    assert "run.completed" not in [event.kind for event in failed_events]

    cancelled_events: list[Event] = []
    with pytest.raises(KeyboardInterrupt, match="cancelled cancelled"):
        _optimization().optimize(
            config=OptimizeAnythingConfig(
                engine=_engine(DeterministicEngine("cancelled", "winner", 1.0, cancel=True)),
                tracking=TrackingConfig(observers=(cancelled_events.append,)),
            )
        )
    assert [event.kind for event in cancelled_events].count("stage.failed") == 2
    assert [event.kind for event in cancelled_events].count("run.cancelled") == 1
    assert "run.failed" not in [event.kind for event in cancelled_events]


def test_durable_omni_run_handles_fresh_reset_missing_result_and_cancellation(
    tmp_path: Path,
) -> None:
    optimization = _optimization()
    directory = tmp_path / "durable"
    run = RunConfig(id="durable", directory=directory)
    config = OptimizeAnythingConfig(
        engine=_engine(DeterministicEngine("durable", "winner", 1.0)),
        run=run,
    )
    optimization.optimize(config=config)

    fresh_events: list[Event] = []
    optimization.optimize(
        config=config.model_copy(
            update={
                "run": run.model_copy(update={"fresh": True}),
                "tracking": TrackingConfig(observers=(fresh_events.append,)),
            }
        )
    )
    assert "checkpoint.reset" in [event.kind for event in fresh_events]

    (directory / "result.json").unlink()
    resumed = config.model_copy(update={"run": run.model_copy(update={"resume": "if_exists"})})
    with pytest.raises(RunStoreError, match="no result artifact"):
        optimization.optimize(config=resumed)
    failed_state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    assert failed_state["status"] == "failed"

    cancelled_directory = tmp_path / "cancelled"
    cancelled_events: list[Event] = []
    with pytest.raises(KeyboardInterrupt, match="cancelled cancelled"):
        optimization.optimize(
            config=OptimizeAnythingConfig(
                engine=_engine(DeterministicEngine("cancelled", "winner", 1.0, cancel=True)),
                run=RunConfig(id="cancelled", directory=cancelled_directory),
                tracking=TrackingConfig(observers=(cancelled_events.append,)),
            )
        )
    cancelled_state = json.loads((cancelled_directory / "state.json").read_text(encoding="utf-8"))
    assert cancelled_state["status"] == "failed"
    assert "checkpoint.written" in [event.kind for event in cancelled_events]


def test_heldout_rescore_reuses_seed_score_and_declares_unbounded_evaluations() -> None:
    events: list[Event] = []
    engine = Engine.custom(
        DeterministicEngine("unchanged", "seed", 0.0),
        candidate_mode="text",
        max_evals=None,
        max_token_cost=1.0,
    )
    result = _optimization(with_test=True).optimize(
        config=OptimizeAnythingConfig(
            engine=engine,
            tracking=TrackingConfig(observers=(events.append,)),
        )
    )

    assert result.scores.baseline_test == result.scores.test == 0.0
    assert result.budget.heldout_evaluation_calls == 1
    started = next(event for event in events if isinstance(event, RunStarted))
    assert started.declaration is not None
    assert started.declaration.evaluation_call_limit is None


def test_omni_rejects_legacy_shortcuts_and_an_invalid_pipeline_boundary() -> None:
    optimization = _optimization()
    engine = _engine(DeterministicEngine("invalid", "winner", 1.0))
    with pytest.raises(ConfigurationError, match="legacy runtime shortcuts"):
        optimization.optimize(
            config=OptimizeAnythingConfig(engine=engine),
            max_metric_calls=2,
        )

    pipeline_result = optimization.optimize(
        config=OptimizeAnythingConfig(
            composition=Pipeline(steps=(Single(engine=engine),)),
        )
    )
    assert pipeline_result.best_candidate.values == {"prompt": "winner"}

    malformed_pipeline = Pipeline.model_construct(
        steps=(Parallel(engines=(engine,)),),
    )
    malformed_config = OptimizeAnythingConfig.model_construct(
        engine=None,
        composition=malformed_pipeline,
    )
    with pytest.raises(RuntimeError, match="no selected candidate"):
        optimization.optimize(config=malformed_config)


def test_pipeline_resume_rejects_inconsistent_and_incomplete_step_snapshots(
    tmp_path: Path,
) -> None:
    optimization = _optimization()

    def completed_config(name: str) -> OptimizeAnythingConfig:
        config = OptimizeAnythingConfig(
            composition=Pipeline(
                steps=(
                    Single(
                        engine=_engine(
                            DeterministicEngine(name, "winner", 1.0),
                        )
                    ),
                )
            ),
            run=RunConfig(id=name, directory=tmp_path / name),
        )
        optimization.optimize(config=config)
        return config

    inconsistent = completed_config("inconsistent")
    inconsistent_path = tmp_path / "inconsistent" / "state.json"
    inconsistent_state = json.loads(inconsistent_path.read_text(encoding="utf-8"))
    inconsistent_state.update({"status": "failed", "next_stage": 0})
    inconsistent_path.write_text(json.dumps(inconsistent_state), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="steps are inconsistent"):
        optimization.optimize(
            config=inconsistent.model_copy(
                update={"run": inconsistent.run.model_copy(update={"resume": "if_exists"})}
            )
        )

    missing_summary = completed_config("missing-summary")
    summary_path = tmp_path / "missing-summary" / "state.json"
    summary_state = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_state["status"] = "failed"
    summary_state["stages"][0]["composition"] = None
    summary_path.write_text(json.dumps(summary_state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no composition summary"):
        optimization.optimize(
            config=missing_summary.model_copy(
                update={"run": missing_summary.run.model_copy(update={"resume": "if_exists"})}
            )
        )

    missing_selection = completed_config("missing-selection")
    selection_path = tmp_path / "missing-selection" / "state.json"
    selection_state = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_state["status"] = "failed"
    selection_state["stages"][0]["best_candidate"]["id"] = None
    selection_path.write_text(json.dumps(selection_state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="no selected execution id"):
        optimization.optimize(
            config=missing_selection.model_copy(
                update={"run": missing_selection.run.model_copy(update={"resume": "if_exists"})}
            )
        )


def test_vote_without_winner_metadata_uses_selection_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = Result(best_candidate="wrong", best_score=0.9)
    second = Result(best_candidate="winner", best_score=0.1)
    raw = Result(
        best_candidate="winner",
        best_score=1.0,
        metadata={
            "all_results": [first, second],
            "vote_scores": [0.0, 1.0],
        },
    )
    monkeypatch.setattr(
        "gepa.oa.ensemble.optimize_vote_with_server",
        lambda *_args, **_kwargs: raw,
    )
    result = _optimization().optimize(
        config=OptimizeAnythingConfig(
            composition=Vote(
                engines=(
                    _engine(DeterministicEngine("first", "wrong", 0.9)),
                    _engine(DeterministicEngine("second", "winner", 0.1)),
                )
            )
        )
    )

    assert result.best_candidate.values == {"prompt": "winner"}
    assert result.composition is not None
    assert result.composition.selections[0].score == 1.0


def test_durable_run_tolerates_missing_gepa_distribution_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_version(_distribution: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(execution_module, "version", missing_version)
    result = _optimization().optimize(
        config=OptimizeAnythingConfig(
            engine=_engine(DeterministicEngine("metadata", "winner", 1.0)),
            run=RunConfig(id="metadata", directory=tmp_path / "metadata"),
        )
    )
    assert result.best_candidate.values == {"prompt": "winner"}
