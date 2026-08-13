from __future__ import annotations as _annotations

import asyncio
import builtins
import sys
import types
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import BaseModel

import pydantic_gepa.experimental.optimize_anything.adapter as optimize_anything_adapter_module
from pydantic_gepa import (
    AgentInstructionsInjection,
    Candidate,
    CandidateComponent,
    CandidateSummary,
    ComponentCatalog,
    EvaluationHarnessError,
    GEPAEventBridge,
    ModelOutputInjection,
    NoopInjection,
    OptimizationDependencyError,
    PydanticEvalsASIBuilder,
    PydanticEvalsHarness,
    PydanticGEPAAdapter,
    PydanticGEPAOptimizer,
    ScoreObjective,
    result_from_gepa,
    run_awaitable_sync,
)
from pydantic_gepa.configuration import (
    BudgetConfig,
    ConfigurationError,
    EvaluationSetConfig,
    GEPAConfig,
    ReflectionConfig,
    RunConfig,
    SelectionConfig,
    TrackingConfig,
)
from pydantic_gepa.eventing import EvaluationEventSink
from pydantic_gepa.events import (
    CaseEvaluated,
    EvaluationCompleted,
    EvaluationStarted,
    Event,
    _dispatcher,
)
from pydantic_gepa.experimental.optimize_anything import (
    Engine,
    OptimizeAnythingConfig,
    OptimizeAnythingFn,
    PydanticOptimizeAnythingAdapter,
    PydanticOptimizeAnythingOptimizer,
)
from pydantic_gepa.recorder import EventValue

if TYPE_CHECKING:
    from gepa.core.callbacks import (
        BudgetUpdatedEvent,
        CandidateAcceptedEvent,
        CandidateRejectedEvent,
        CandidateSelectedEvent,
        ErrorEvent,
        EvaluationEndEvent,
        EvaluationSkippedEvent,
        EvaluationStartEvent,
        IterationEndEvent,
        IterationStartEvent,
        MergeAcceptedEvent,
        MergeAttemptedEvent,
        MergeRejectedEvent,
        MinibatchSampledEvent,
        OptimizationEndEvent,
        OptimizationStartEvent,
        ParetoFrontUpdatedEvent,
        ProposalEndEvent,
        ProposalStartEvent,
        ReflectiveDatasetBuiltEvent,
        StateSavedEvent,
        ValsetEvaluatedEvent,
    )


def test_harness_builds_batch_dataset_and_runs_sync_reports() -> None:
    source_dataset = _Dataset(cases=[_Case("source")], evaluators=["accuracy"])
    harness: PydanticEvalsHarness[_Case, dict[str, str], _Report, str] = PydanticEvalsHarness(
        dataset=source_dataset,
        task=lambda case: {"answer": case.name},
        max_concurrency=3,
    )

    report = harness.evaluate([_Case("case_1")])

    assert report.cases[0].name == "case_1"
    assert report.max_concurrency == 3
    assert report.progress is False
    assert report.evaluators == ["accuracy"]


def test_harness_preserves_dataset_name_when_supported() -> None:
    source_dataset = _NamedDataset(
        name="named-suite", cases=[_Case("source")], evaluators=["accuracy"]
    )
    harness: PydanticEvalsHarness[_Case, dict[str, str], _NamedReport, str] = PydanticEvalsHarness(
        dataset=source_dataset,
        task=lambda case: {"answer": case.name},
    )

    report = harness.evaluate([_Case("case_1")])

    assert report.dataset_name == "named-suite"


def test_harness_falls_back_when_named_dataset_constructor_is_unsupported() -> None:
    source_dataset = _NamedAttributeDataset(cases=[_Case("source")], evaluators=["accuracy"])
    harness: PydanticEvalsHarness[_Case, dict[str, str], _Report, str] = PydanticEvalsHarness(
        dataset=source_dataset,
        task=lambda case: {"answer": case.name},
    )

    report = harness.evaluate([_Case("case_1")])

    assert report.cases[0].name == "case_1"


def test_harness_raises_when_named_dataset_fallback_is_also_unsupported() -> None:
    with pytest.raises(EvaluationHarnessError, match="constructible"):
        harness: PydanticEvalsHarness[_Case, _Case, _Report, str] = PydanticEvalsHarness(
            dataset=_NamedBadDataset(),
            task=lambda case: case,
        )
        harness.evaluate([])


async def test_harness_runs_async_reports_inside_existing_loop() -> None:
    harness: PydanticEvalsHarness[_Case, _Case, _Report, str] = PydanticEvalsHarness(
        dataset=_AsyncDataset(cases=[], evaluators=[]),
        task=lambda case: case,
    )

    report = harness.evaluate([_Case("case_1")])
    direct = run_awaitable_sync(_async_value("ok"))

    assert report.cases[0].name == "case_1"
    assert direct == "ok"
    with pytest.raises(RuntimeError, match="async boom"):
        run_awaitable_sync(_async_error())


def test_run_awaitable_sync_without_running_loop() -> None:
    assert run_awaitable_sync(_async_value("sync")) == "sync"


def test_harness_reports_bad_dataset_shapes() -> None:
    with pytest.raises(EvaluationHarnessError, match="constructible"):
        harness: PydanticEvalsHarness[_Case, _Case, _Report, str] = PydanticEvalsHarness(
            dataset=_BadDataset(), task=lambda case: case
        )
        harness.evaluate([])

    with pytest.raises(EvaluationHarnessError, match="evaluate"):
        harness = PydanticEvalsHarness[_Case, _Case, _Report, str](
            dataset=_NoEvaluateDataset(cases=[], evaluators=[]), task=lambda case: case
        )
        harness.evaluate([])


def test_adapter_evaluates_candidate_with_injection_scores_failures_and_recorder() -> None:
    agent = _FakeAgent()
    recorder = _Recorder()
    adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(cases=[], evaluators=[]),
        task=lambda case: {"answer": case.name},
        injections=[AgentInstructionsInjection(agent=agent)],
        objective=ScoreObjective(score_key="accuracy"),
        recorder=recorder,
    )

    result = adapter.evaluate(
        [_Case("case_1"), _Case("case_2")],
        {"instructions": "Use the case name."},
        capture_traces=True,
    )
    reflective = adapter.make_reflective_dataset(
        {"instructions": "Use the case name."},
        result,
        ["instructions"],
    )

    assert result.outputs == [{"answer": "case_1"}, {"answer": "case_2"}, None]
    assert result.scores == [1.0, 1.0, 0.0]
    assert result.objective_scores == [{"accuracy": 1.0}, {"accuracy": 1.0}, {}]
    assert result.num_metric_calls == 3
    assert result.trajectories is not None
    assert len(result.trajectories) == 3
    assert agent.seen_instructions == ["Use the case name."]
    assert recorder.records[0]["scores"] == [1.0, 1.0, 0.0]
    assert reflective["instructions"][0]["case_name"] == "case_1"


def test_adapter_emits_correlated_events_for_reports_and_failures() -> None:
    events: list[Event] = []
    case = _Case("case_1")
    dispatcher = _dispatcher(
        run_id="evaluation-events",
        backend="gepa",
        local_observers=(events.append,),
    )
    adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(cases=[], evaluators=[]),
        task=lambda value: {"answer": value.name},
        injections=[],
        objective=ScoreObjective(score_key="accuracy"),
    )
    adapter.events = EvaluationEventSink(
        dispatcher,
        objective=adapter.objective,
        trainset=(case,),
        valset=(),
    )

    result = adapter.evaluate([case], {}, capture_traces=True)
    assert result.scores == [1.0, 0.0]
    assert [event.kind for event in events].count("case.evaluated") == 2
    assert all(
        event.split == "train"
        for event in events
        if isinstance(event, EvaluationStarted | EvaluationCompleted | CaseEvaluated)
    )

    def fail_subject(_value: _Case) -> dict[str, str]:
        raise RuntimeError("subject failed")

    failing: PydanticGEPAAdapter[_Case, dict[str, str], str] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(cases=[], evaluators=[]),
        task=fail_subject,
        injections=[],
        objective=ScoreObjective(score_key="accuracy"),
    )
    failing.events = EvaluationEventSink(
        dispatcher,
        objective=failing.objective,
        trainset=(case,),
        valset=(),
    )
    with pytest.raises(RuntimeError, match="subject failed"):
        failing.evaluate([case], {})
    assert events[-2].kind == "evaluation.skipped"
    assert events[-1].kind == "backend.error"

    unobserved = failing.model_copy(update={"events": None})
    with pytest.raises(RuntimeError, match="subject failed"):
        unobserved.evaluate([case], {})

    case_only: PydanticGEPAAdapter[_Case, dict[str, str], str] = PydanticGEPAAdapter.from_dataset(
        dataset=_NoScoreDataset(cases=[], evaluators=[]),
        task=lambda value: {"answer": value.name},
        injections=[],
        objective=ScoreObjective(score_key="accuracy"),
    )
    case_only.events = EvaluationEventSink(
        dispatcher,
        objective=case_only.objective,
        trainset=(case,),
        valset=(),
    )
    assert case_only.evaluate([case], {}).scores == [0.0]


def test_adapter_can_store_custom_proposer() -> None:
    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        del reflective_dataset, components_to_update
        return candidate

    adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(cases=[], evaluators=[]),
        task=lambda case: {"answer": case.name},
        injections=[],
        objective=ScoreObjective(score_key="accuracy"),
        propose_new_texts=propose,
    )

    assert adapter.propose_new_texts is propose
    proposer = adapter.propose_new_texts
    assert proposer is not None
    assert proposer(
        {"instructions": "seed"},
        {"instructions": []},
        ["instructions"],
    ) == {"instructions": "seed"}


def test_adapter_reports_component_catalog_names_or_injection_components() -> None:
    catalog = ComponentCatalog.from_components(
        [CandidateComponent(name="instructions", initial_text="seed")]
    )
    catalog_adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = (
        PydanticGEPAAdapter.from_dataset(
            dataset=_Dataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[],
            components=catalog,
            objective=ScoreObjective(score_key="accuracy"),
        )
    )
    injection_adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = (
        PydanticGEPAAdapter.from_dataset(
            dataset=_Dataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[AgentInstructionsInjection(agent=_FakeAgent())],
            objective=ScoreObjective(score_key="accuracy"),
        )
    )

    assert catalog_adapter.component_names() == ["instructions"]
    assert injection_adapter.component_names() == ["instructions"]


def test_adapter_can_skip_trajectories_and_handles_tuple_report_fields() -> None:
    adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = PydanticGEPAAdapter.from_dataset(
        dataset=_TupleDataset(cases=[], evaluators=[]),
        task=lambda case: {"answer": case.name},
        injections=[],
        objective=ScoreObjective(score_key="accuracy"),
    )

    result = adapter.evaluate([_Case("case_1")], {}, capture_traces=False)

    assert result.outputs == [{"answer": "case_1"}, None]
    assert result.trajectories is None


def test_adapter_uses_failure_score_for_cases_without_scores() -> None:
    adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = PydanticGEPAAdapter.from_dataset(
        dataset=_NoScoreDataset(cases=[], evaluators=[]),
        task=lambda case: {"answer": case.name},
        injections=[],
        objective=ScoreObjective(score_key="accuracy", failure_score=-1.0),
    )

    result = adapter.evaluate([_Case("case_1")], {}, capture_traces=True)

    assert result.outputs == [{"answer": "case_1"}]
    assert result.scores == [-1.0]
    assert result.objective_scores == [{}]
    assert result.trajectories is not None


def test_adapter_handles_reports_with_failures_only() -> None:
    adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = PydanticGEPAAdapter.from_dataset(
        dataset=_FailureOnlyDataset(cases=[], evaluators=[]),
        task=lambda case: {"answer": case.name},
        injections=[],
        objective=ScoreObjective(score_key="accuracy", failure_score=-1.0),
    )

    result = adapter.evaluate([_Case("case_1")], {}, capture_traces=False)

    assert result.outputs == [None]
    assert result.scores == [-1.0]
    assert result.objective_scores == [{}]


def test_optimizer_requires_validation_set_unless_explicitly_allowed() -> None:
    optimizer = PydanticGEPAOptimizer(
        adapter=PydanticGEPAAdapter.from_dataset(
            dataset=_Dataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[],
            objective=ScoreObjective(score_key="accuracy"),
        ),
        initial_candidate=Candidate(values={"instructions": "seed"}),
        optimize_fn=_fake_optimize,
    )

    with pytest.raises(OptimizationDependencyError, match="valset is required"):
        optimizer.optimize(trainset=[_Case("case_1")])

    result = optimizer.optimize(
        trainset=[_Case("case_1")],
        config=GEPAConfig(
            reflection=ReflectionConfig(
                model="test:reflection",
                model_kwargs={"temperature": 0.1},
            ),
            budget=BudgetConfig(max_metric_calls=3),
            evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
        ),
    )

    assert result.best_candidate.values == {"instructions": "optimized"}
    assert result.best_score == 0.9
    assert result.validation_scores == [0.8, 1.0]
    assert result.candidate_history[0].candidate_id == "candidate_0"
    assert result.best_candidate_index is None
    assert result.raw_gepa_result is not None


def test_standard_optimizer_reports_cancellation_and_fresh_checkpoint_reset(
    tmp_path: Path,
) -> None:
    adapter = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(cases=[], evaluators=[]),
        task=lambda case: {"answer": case.name},
        injections=[],
        objective=ScoreObjective(score_key="accuracy"),
    )
    cancelled_events: list[Event] = []

    def cancel(**kwargs: Any) -> _RawResult:
        del kwargs
        raise KeyboardInterrupt("cancelled")

    cancelled = PydanticGEPAOptimizer(
        adapter=adapter,
        initial_candidate=Candidate(values={"instructions": "seed"}),
        optimize_fn=cancel,
    )
    with pytest.raises(KeyboardInterrupt, match="cancelled"):
        cancelled.optimize(
            trainset=[_Case("case")],
            config=GEPAConfig(
                evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
                tracking=TrackingConfig(observers=(cancelled_events.append,)),
            ),
        )
    assert cancelled_events[-1].kind == "run.cancelled"

    def complete(**kwargs: Any) -> _RawResult:
        del kwargs
        return _RawResult()

    optimizer = cancelled.model_copy(update={"optimize_fn": complete})
    directory = tmp_path / "fresh-standard"
    initial = GEPAConfig(
        budget=BudgetConfig(max_metric_calls=3),
        reflection=ReflectionConfig(
            model="test:reflection",
            model_kwargs={"temperature": 0.1},
        ),
        run=RunConfig(id="fresh-standard", directory=directory),
        evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
    )
    optimizer.optimize(trainset=[_Case("case")], config=initial)
    fresh_events: list[Event] = []
    optimizer.optimize(
        trainset=[_Case("case")],
        config=initial.model_copy(
            update={
                "run": initial.run.model_copy(update={"fresh": True}),
                "tracking": TrackingConfig(observers=(fresh_events.append,)),
            }
        ),
    )
    assert "checkpoint.reset" in [event.kind for event in fresh_events]


def test_optimizer_reports_missing_gepa_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = PydanticGEPAOptimizer(
        adapter=PydanticGEPAAdapter.from_dataset(
            dataset=_Dataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[],
            objective=ScoreObjective(score_key="accuracy"),
        ),
        initial_candidate=Candidate(values={"instructions": "seed"}),
    )

    original_import = builtins.__import__

    def missing_gepa_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name == "gepa.api":
            raise ImportError("missing gepa")
        return original_import(name, globals, locals, fromlist, level)

    math_module = missing_gepa_import("math")
    assert isinstance(math_module, types.ModuleType)
    assert math_module.__name__ == "math"
    monkeypatch.delitem(sys.modules, "gepa", raising=False)
    monkeypatch.delitem(sys.modules, "gepa.api", raising=False)
    monkeypatch.setattr(builtins, "__import__", missing_gepa_import)

    with pytest.raises(OptimizationDependencyError, match="GEPA is not installed"):
        optimizer.optimize(
            trainset=[_Case("case_1")],
            config=GEPAConfig(
                reflection=ReflectionConfig(model="test:reflection"),
                evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
            ),
        )


def test_optimizer_can_load_gepa_optimize_from_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = types.ModuleType("gepa")
    package.__path__ = []
    api_module = types.ModuleType("gepa.api")

    def optimize(**kwargs: Any) -> _RawResult:
        assert kwargs["seed_candidate"] == {"instructions": "seed"}
        return _RawResult()

    api_module.__dict__["optimize"] = optimize
    monkeypatch.setitem(sys.modules, "gepa", package)
    monkeypatch.setitem(sys.modules, "gepa.api", api_module)
    optimizer = PydanticGEPAOptimizer(
        adapter=PydanticGEPAAdapter.from_dataset(
            dataset=_Dataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[],
            objective=ScoreObjective(score_key="accuracy"),
        ),
        initial_candidate=Candidate(values={"instructions": "seed"}),
    )

    result = optimizer.optimize(
        trainset=[_Case("case_1")],
        config=GEPAConfig(
            reflection=ReflectionConfig(model="test:reflection"),
            evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
        ),
    )

    assert result.best_score == 0.9


def test_result_from_gepa_handles_sparse_result_shapes() -> None:
    result = result_from_gepa(_SparseResult())
    summary = CandidateSummary(candidate_id="manual", score=1.0, values={"a": "b"})

    assert result.best_candidate.values == {"prompt": "best"}
    assert result.best_score == 0.7
    assert result.validation_scores == []
    assert result.candidate_history == []
    assert result.candidates == []
    assert result.parent_indices == []
    normalized = result_from_gepa(_HistoryResult(summary)).candidate_history[0]
    assert normalized.candidate_id == summary.candidate_id
    assert normalized.status == "proposed"
    assert normalized.deltas[0].component == "a"


def test_result_from_gepa_handles_candidate_objects_and_default_history_fields() -> None:
    candidate = Candidate(values={"instructions": "ready"})
    candidate_result = result_from_gepa(_CandidateObjectResult(candidate))
    validation_result = result_from_gepa(_ValidationOnlyResult())
    history_result = result_from_gepa(_CandidateOnlyHistoryResult())
    score_history_result = result_from_gepa(_ScoreOnlyHistoryResult())

    assert candidate_result.best_candidate is candidate
    assert validation_result.best_candidate.values == {}
    assert validation_result.best_score == 0.0
    assert validation_result.validation_scores == [0.25]
    assert validation_result.best_candidate_index is None
    assert validation_result.objective_scores is None
    assert validation_result.candidate_tree_dot is None
    assert validation_result.candidate_tree_html is None
    assert history_result.candidate_history[0].values == {"instructions": "candidate"}
    assert history_result.candidate_history[0].candidate_id == "candidate_0"
    assert score_history_result.candidate_history[0].score == 0.4
    assert score_history_result.candidate_history[0].values == {}


def test_result_from_gepa_normalizes_actual_gepa_result_shape() -> None:
    result = result_from_gepa(_GepaLikeResult())

    assert result.best_candidate.values == {"instructions": "candidate 1"}
    assert result.best_score == 0.9
    assert result.best_candidate_index == 1
    assert result.validation_scores == [0.4, 0.9]
    assert [candidate.values for candidate in result.candidates] == [
        {"instructions": "candidate 0"},
        {"instructions": "candidate 1"},
    ]
    assert result.parent_indices == [[], [0]]
    assert result.candidate_history[1].candidate_id == "candidate_1"
    assert result.candidate_history[1].parent_ids == ["candidate_0"]
    assert result.candidate_history[1].validation_subscores == {"case_1": 1.0}
    assert result.candidate_history[1].objective_scores == {"accuracy": 0.9}
    assert result.candidate_history[0].status == "proposed"
    assert result.candidate_history[1].status == "best"
    assert result.candidate_history[1].deltas[0].before == "candidate 0"
    assert result.candidate_history[1].deltas[0].after == "candidate 1"
    assert result.objective_scores == [{"accuracy": 0.4}, {"accuracy": 0.9}]
    assert result.total_metric_calls == 12
    assert result.num_full_val_evals == 3
    assert result.run_dir == "runs/demo"
    assert result.seed == 7
    assert result.per_objective_best_candidates == {"accuracy": [1]}
    assert result.objective_pareto_front == {"accuracy": 0.9}
    assert result.candidate_tree_dot == "digraph { candidate_0 -> candidate_1 }"
    assert result.candidate_tree_html == "<html>tree</html>"
    assert result.scores.validation == 0.9
    assert result.budget.metric_calls == 12
    assert result.checkpoints == ("runs/demo",)
    assert result.reported["best_score"] == 0.9
    assert result.derived["candidate_count"] == 2
    assert "raw_gepa_result" not in result.stable_dump()


def test_result_from_gepa_handles_missing_objective_subscores() -> None:
    result = result_from_gepa(_GepaLikeNoSubscoresResult())

    assert result.objective_scores is None


def test_result_lineage_derives_parent_deltas_acceptance_and_feedback() -> None:
    class LineageResult:
        best_idx = 0
        best_candidate = {"prompt": "p0"}
        candidates = [
            {"prompt": "p0"},
            {"prompt": "p1"},
            {"prompt": "p2"},
        ]
        parents = [[], [0], [1]]
        candidate_history = [
            CandidateSummary(candidate_id="candidate_0", score=1.0, values={"prompt": "p0"}),
            CandidateSummary(
                candidate_id="candidate_1",
                score=0.8,
                values={"prompt": "p1"},
                metadata={"accepted": True},
            ),
            CandidateSummary(
                candidate_id="candidate_2",
                score=0.7,
                values={"prompt": "p2"},
                metadata={"accepted": False, "reason": "validation regressed"},
            ),
        ]

    result = result_from_gepa(LineageResult())

    assert [item.status for item in result.candidate_history] == [
        "best",
        "accepted",
        "rejected",
    ]
    assert result.candidate_history[1].deltas[0].before == "p0"
    assert result.candidate_history[2].deltas[0].before == "p1"
    assert result.candidate_history[2].feedback == ("validation regressed",)

    class InvalidParentResult:
        best_candidate = {"prompt": "candidate"}
        candidates = [{"prompt": "candidate"}]
        parents = [[4]]
        candidate_history = [
            CandidateSummary(
                candidate_id="candidate_0",
                score=0.5,
                values={"prompt": "candidate"},
            )
        ]

    invalid_parent = result_from_gepa(InvalidParentResult()).candidate_history[0]
    assert invalid_parent.deltas[0].before is None
    assert invalid_parent.deltas[0].after == "candidate"


def test_result_from_gepa_handles_empty_objectives_and_tree_accessors() -> None:
    result = result_from_gepa(_EmptyObjectiveResult())

    assert result.objective_scores == []
    assert result.candidate_tree_dot is None
    assert result.candidate_tree_html is None


def test_gepa_event_bridge_normalizes_external_callback_payloads() -> None:
    recorder = _EventRecorder()
    bridge = GEPAEventBridge(recorder=recorder)
    bridge.on_optimization_start(
        cast(
            "OptimizationStartEvent",
            {
                "seed_candidate": {"prompt": "seed"},
                "trainset_size": 2,
                "valset_size": 1,
                "config": {"max_metric_calls": 5},
            },
        )
    )
    bridge.on_optimization_end(
        cast(
            "OptimizationEndEvent",
            {
                "best_candidate_idx": 1,
                "total_iterations": 2,
                "total_metric_calls": 4,
                "final_state": None,
            },
        )
    )
    bridge.on_iteration_start(
        cast(
            "IterationStartEvent",
            {"iteration": 2, "state": None, "trainset_loader": None},
        )
    )
    bridge.on_iteration_end(
        cast(
            "IterationEndEvent",
            {"iteration": 2, "state": None, "proposal_accepted": True},
        )
    )
    bridge.on_candidate_selected(
        cast(
            "CandidateSelectedEvent",
            {
                "iteration": 2,
                "candidate_idx": 1,
                "candidate": {"prompt": "selected"},
                "score": 0.7,
            },
        )
    )
    bridge.on_minibatch_sampled(
        cast(
            "MinibatchSampledEvent",
            {"iteration": 2, "minibatch_ids": ["a", "b"], "trainset_size": 4},
        )
    )
    bridge.on_evaluation_start(
        cast(
            "EvaluationStartEvent",
            {
                "iteration": 2,
                "candidate_idx": 1,
                "batch_size": 2,
                "capture_traces": True,
                "parent_ids": [0],
                "inputs": ["a", "b"],
                "is_seed_candidate": False,
            },
        )
    )
    bridge.on_evaluation_end(
        cast(
            "EvaluationEndEvent",
            {
                "iteration": 2,
                "candidate_idx": 1,
                "scores": [0.1, 0.9],
                "has_trajectories": True,
                "parent_ids": [0],
                "outputs": ["left", "right"],
                "trajectories": ["trace-left", "trace-right"],
                "objective_scores": [{"accuracy": 0.1}, {"accuracy": 0.9}],
                "is_seed_candidate": False,
            },
        )
    )
    bridge.on_evaluation_skipped(
        cast(
            "EvaluationSkippedEvent",
            {
                "iteration": 2,
                "candidate_idx": 1,
                "reason": "already evaluated",
                "scores": [0.1, 0.9],
                "is_seed_candidate": False,
            },
        )
    )
    bridge.on_valset_evaluated(
        cast(
            "ValsetEvaluatedEvent",
            {
                "iteration": 2,
                "candidate_idx": 1,
                "candidate": {"prompt": "selected"},
                "scores_by_val_id": {"validation": 0.9},
                "average_score": 0.9,
                "num_examples_evaluated": 1,
                "total_valset_size": 1,
                "parent_ids": [0],
                "is_best_program": True,
                "outputs_by_val_id": {"validation": "output"},
            },
        )
    )
    bridge.on_reflective_dataset_built(
        cast(
            "ReflectiveDatasetBuiltEvent",
            {
                "iteration": 2,
                "iteration_id": "iteration-2",
                "candidate_idx": 1,
                "components": ["prompt"],
                "dataset": {"prompt": [{"feedback": "improve"}]},
            },
        )
    )
    bridge.on_proposal_start(
        cast(
            "ProposalStartEvent",
            {
                "iteration": 2,
                "parent_candidate": {"prompt": "selected"},
                "components": ["prompt"],
                "reflective_dataset": {"prompt": [{"feedback": "improve"}]},
            },
        )
    )
    bridge.on_proposal_end(
        cast(
            "ProposalEndEvent",
            {
                "iteration": 2,
                "new_instructions": {"prompt": "proposed"},
                "prompts": {"prompt": "reflection prompt"},
                "raw_lm_outputs": {"prompt": "proposed"},
            },
        )
    )
    bridge.on_candidate_accepted(
        cast(
            "CandidateAcceptedEvent",
            {
                "iteration": 2,
                "new_candidate_idx": 2,
                "new_score": 0.9,
                "parent_ids": [1],
            },
        )
    )
    bridge.on_candidate_rejected(
        cast(
            "CandidateRejectedEvent",
            {"iteration": 2, "old_score": 0.9, "new_score": 0.8, "reason": "regressed"},
        )
    )
    bridge.on_merge_attempted(
        cast(
            "MergeAttemptedEvent",
            {
                "iteration": 2,
                "parent_ids": [1, 2],
                "merged_candidate": {"prompt": "merged"},
            },
        )
    )
    bridge.on_merge_accepted(
        cast(
            "MergeAcceptedEvent",
            {"iteration": 2, "new_candidate_idx": 3, "parent_ids": [1, 2]},
        )
    )
    bridge.on_merge_rejected(
        cast(
            "MergeRejectedEvent",
            {"iteration": 2, "parent_ids": [1, 2], "reason": "no improvement"},
        )
    )
    bridge.on_pareto_front_updated(
        cast(
            "ParetoFrontUpdatedEvent",
            {"iteration": 2, "new_front": [2, 3], "displaced_candidates": [1]},
        )
    )
    bridge.on_state_saved(cast("StateSavedEvent", {"iteration": 2, "run_dir": "runs/demo"}))
    bridge.on_budget_updated(
        cast(
            "BudgetUpdatedEvent",
            {
                "iteration": 2,
                "metric_calls_used": 4,
                "metric_calls_delta": 2,
                "metric_calls_remaining": 1,
            },
        )
    )
    bridge.on_error(
        cast(
            "ErrorEvent",
            {"iteration": 2, "exception": ValueError("boom"), "will_continue": False},
        )
    )

    assert [record["event_name"] for record in recorder.events] == [
        "run.started",
        "run.completed",
        "iteration.started",
        "iteration.completed",
        "candidate.proposed",
        "backend.progress",
        "evaluation.started",
        "evaluation.completed",
        "evaluation.skipped",
        "candidate.evaluated",
        "backend.progress",
        "reflection.started",
        "reflection.completed",
        "candidate.proposed",
        "candidate.accepted",
        "candidate.rejected",
        "backend.progress",
        "candidate.accepted",
        "candidate.rejected",
        "pareto_front.updated",
        "checkpoint.written",
        "budget.updated",
        "run.failed",
    ]
    payload = cast("Mapping[str, Any]", recorder.events[0]["payload"])
    assert isinstance(payload, Mapping)
    assert payload["metadata"] == {"train_count": 2, "validation_count": 1}
    assert payload["seed"] == {
        "values": {"prompt": "seed"},
        "id": None,
        "parent_id": None,
        "generation": None,
        "metadata": {},
    }


def test_backend_only_bridge_suppresses_root_lifecycle_and_reports_terminal_errors() -> None:
    recorder = _EventRecorder()
    bridge = GEPAEventBridge(
        run_id="backend-only",
        recorder=recorder,
        lifecycle="backend_only",
    )
    bridge.on_optimization_start(
        cast(
            "OptimizationStartEvent",
            {"seed_candidate": {"prompt": "seed"}, "trainset_size": 1, "valset_size": 1},
        )
    )
    bridge.on_optimization_end(
        cast(
            "OptimizationEndEvent",
            {"best_candidate_idx": 0, "total_metric_calls": 1, "total_iterations": 1},
        )
    )
    bridge.on_error(
        cast(
            "ErrorEvent",
            {"iteration": 3, "exception": ValueError("terminal"), "will_continue": False},
        )
    )

    assert [record["event_name"] for record in recorder.events] == ["backend.error"]


def test_result_from_gepa_handles_none_objective_frontier_fields() -> None:
    result = result_from_gepa(_NoneObjectiveFrontierResult())

    assert result.per_objective_best_candidates is None
    assert result.objective_pareto_front is None


def test_optimize_anything_adapter_proxies_and_builds_structured_side_info() -> None:
    class _SuccessDataset:
        def __init__(self, *, cases: Sequence[_Case], evaluators: Sequence[str]) -> None:
            self.cases = list(cases)
            self.evaluators = list(evaluators)

        def evaluate(self, task: Any, *, max_concurrency: int, progress: bool) -> _Report:
            report_cases = [
                _ReportCase(
                    name=case.name,
                    inputs={"name": case.name},
                    expected_output={"answer": case.name},
                    output=task(case),
                    scores={"accuracy": _Score(1.0)},
                )
                for case in self.cases
            ]
            return _Report(
                cases=report_cases,
                failures=[],
                max_concurrency=max_concurrency,
                progress=progress,
                evaluators=list(self.evaluators),
            )

    catalog = ComponentCatalog.from_components(
        [CandidateComponent(name="instructions", initial_text="seed")]
    )
    base_adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = (
        PydanticGEPAAdapter.from_dataset(
            dataset=_SuccessDataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[],
            components=catalog,
            objective=ScoreObjective(score_key="accuracy"),
        )
    )
    adapter = PydanticOptimizeAnythingAdapter(adapter=base_adapter)

    assert adapter.normalize_candidate({"instructions": "seed", "unknown": "raw"}) == {
        "instructions": "seed",
        "unknown": "raw",
    }
    assert adapter.normalize_candidate({"instructions": "seed"}) == {"instructions": "seed"}

    encoded_adapter: PydanticGEPAAdapter[_Case, dict[str, str], str] = (
        PydanticGEPAAdapter.from_dataset(
            dataset=_SuccessDataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[],
            components=ComponentCatalog.from_components(
                [
                    CandidateComponent(
                        name="instructions",
                        initial_text="seed",
                        serialization="json_string",
                    )
                ]
            ),
            objective=ScoreObjective(score_key="accuracy"),
        )
    )
    assert encoded_adapter.normalize_candidate({"instructions": "raw proposal"}) == {
        "instructions": '"raw proposal"'
    }

    eval_batch = adapter.evaluate([_Case("case_1")], {"instructions": "seed"}, capture_traces=True)
    score, side_info = adapter.evaluator(
        {"instructions": "seed"},
        example=_Case("case_1"),
    )
    reflective = adapter.make_reflective_dataset(
        {"instructions": "seed"},
        eval_batch,
        ["instructions"],
    )

    assert eval_batch.scores == [1.0]
    assert score == 1.0
    assert adapter.component_names() == ["instructions"]
    assert reflective["instructions"][0]["case_name"] == "case_1"
    assert side_info["scores"] == {"accuracy": 1.0}
    assert side_info["observed_scores"] == {"accuracy": 1.0}
    assert side_info["case_name"] == "case_1"
    assert side_info["actual_output"] == {"answer": "case_1"}
    assert side_info["objective_scores"] == {"accuracy": 1.0}
    assert side_info["instructions_specific_info"] == {
        "examples": [
            {
                "case_name": "case_1",
                "inputs": {"name": "case_1"},
                "score": 1.0,
                "success": True,
                "failure_category": None,
                "expected_output": {"answer": "case_1"},
                "metadata": {},
                "actual_output": {"answer": "case_1"},
                "scores": {"accuracy": 1.0},
            }
        ]
    }

    class Output(BaseModel):
        answer: str

    output_injection = ModelOutputInjection(Output)
    inferred = PydanticOptimizeAnythingAdapter(
        adapter=PydanticGEPAAdapter.from_dataset(
            dataset=_NoScoreDataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[
                AgentInstructionsInjection(agent=_FakeAgent()),
                output_injection,
                NoopInjection(component="untyped"),
            ],
            objective=ScoreObjective(score_key="accuracy"),
        )
    )
    field_component = output_injection.components.components[0]
    assert inferred.normalize_candidate(
        {
            "instructions": "raw instructions",
            field_component.name: "raw description",
        }
    ) == {
        "instructions": "raw instructions",
        field_component.name: "raw description",
    }


def test_optimize_anything_adapter_handles_empty_objective_scores_and_invalid_shapes() -> None:
    no_score_adapter: PydanticOptimizeAnythingAdapter[_Case, dict[str, str], str] = (
        PydanticOptimizeAnythingAdapter(
            adapter=PydanticGEPAAdapter.from_dataset(
                dataset=_NoScoreDataset(cases=[], evaluators=[]),
                task=lambda case: {"answer": case.name},
                injections=[],
                objective=ScoreObjective(score_key="accuracy"),
            )
        )
    )
    score, side_info = no_score_adapter.evaluator({}, example=_Case("case_1"))

    assert score == 0.0
    assert "observed_scores" not in side_info
    assert "objective_scores" not in side_info

    invalid_adapter: PydanticOptimizeAnythingAdapter[_Case, dict[str, str], str] = (
        PydanticOptimizeAnythingAdapter(
            adapter=PydanticGEPAAdapter.from_dataset(
                dataset=_Dataset(cases=[], evaluators=[]),
                task=lambda case: {"answer": case.name},
                injections=[],
                objective=ScoreObjective(score_key="accuracy"),
            )
        )
    )
    with pytest.raises(EvaluationHarnessError, match="exactly one evaluation result"):
        invalid_adapter.evaluator({}, example=_Case("case_1"))


def test_optimize_anything_optimizer_uses_local_config_and_runtime_overrides() -> None:
    components = ComponentCatalog.from_components(
        [CandidateComponent(name="instructions", initial_text="seed")]
    )
    adapter = PydanticOptimizeAnythingAdapter(
        adapter=PydanticGEPAAdapter.from_dataset(
            dataset=_NoScoreDataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[],
            components=components,
            objective=ScoreObjective(score_key="accuracy"),
        )
    )
    optimizer = PydanticOptimizeAnythingOptimizer(
        adapter=adapter,
        initial_candidate=Candidate(values={"instructions": "seed"}),
        optimization_objective="base objective",
        background="base background",
        optimize_fn=cast("OptimizeAnythingFn[_Case]", _fake_optimize_anything),
    )

    with pytest.raises(OptimizationDependencyError, match="valset is required"):
        optimizer.optimize(trainset=[_Case("case_1")])
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        optimizer.optimize(
            trainset=[_Case("case_1")],
            allow_same_train_val=True,
            config=GEPAConfig(),
            max_metric_calls=2,
        )
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        optimizer.optimize(
            trainset=[_Case("case_1")],
            config=GEPAConfig(),
            allow_same_train_val=True,
        )
    invalid_objective: dict[str, Any] = {"objective": 1}
    with pytest.raises(TypeError, match="objective override must be a string"):
        optimizer.optimize(
            trainset=[_Case("case_1")],
            allow_same_train_val=True,
            **invalid_objective,
        )
    invalid_background: dict[str, Any] = {"background": False}
    with pytest.raises(TypeError, match="background override must be a string"):
        optimizer.optimize(
            trainset=[_Case("case_1")],
            allow_same_train_val=True,
            **invalid_background,
        )

    result = optimizer.optimize(
        trainset=[_Case("case_1")],
        allow_same_train_val=True,
        max_metric_calls=9,
        objective="runtime objective",
        background="runtime background",
    )

    assert result.best_candidate.values == {"instructions": "opt-anything"}
    assert result.final_candidate == result.best_candidate
    assert result.candidate_history[0].values == {"instructions": "opt-anything"}
    assert result.candidate_history[0].metadata == {"engine": "gepa", "branch": "branch-0"}
    assert result.best_score == 0.95
    assert result.backend == "optimize_anything"

    assert result.budget.metric_calls == 9
    assert result.budget.metric_call_limit == 9
    assert result.composition is not None
    assert result.composition.kind == "single"


def test_optimize_anything_preserves_typed_gepa_settings() -> None:
    configured = GEPAConfig(
        budget=BudgetConfig(max_metric_calls=3, max_reflection_cost=1.5),
        reflection=ReflectionConfig(model="test:model", model_kwargs={"temperature": 0.1}),
        selection=SelectionConfig(acceptance="improvement_or_equal"),
        tracking=TrackingConfig(
            backend_callbacks=(lambda *_args: None,),
            use_wandb=True,
            wandb_attach_existing=True,
            use_mlflow=True,
            mlflow_attach_existing=True,
            key_prefix="experiment",
        ),
    )

    engine = Engine.gepa(configured)
    declaration = OptimizeAnythingConfig(engine=engine).declaration()

    assert engine.max_evals == 3
    assert engine.max_token_cost == 1.5
    assert engine.gepa_config is configured
    assert declaration["engine"] is not None


def test_optimize_anything_optimizer_can_load_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    package = types.ModuleType("gepa")
    package.__path__ = []
    optimize_anything_module_fake = types.ModuleType("gepa.optimize_anything")

    @dataclass(frozen=True)
    class _OptimizeAnythingConfig:
        engine: Any
        name: str
        max_evals: int | None
        max_token_cost: float | None
        max_concurrency: int
        output_dir: str | None
        run_dir: str | None
        stop_at_score: float | None
        sandbox: bool
        engine_config: dict[str, Any]

    def optimize_anything(**kwargs: Any) -> _RawResult:
        config = kwargs["config"]
        assert isinstance(config, _OptimizeAnythingConfig)
        assert config.engine == "gepa"
        assert config.max_evals == 11
        assert config.run_dir == str(tmp_path / "optional" / "backend" / "step-0" / "branch-0")
        assert kwargs["objective"] == "custom objective"
        result = _RawResult()
        result.best_candidate = {"instructions": "optional"}
        result.best_score = 0.8
        return result

    optimize_anything_module_fake.__dict__["OptimizeAnythingConfig"] = _OptimizeAnythingConfig
    optimize_anything_module_fake.__dict__["optimize_anything"] = optimize_anything
    monkeypatch.setitem(sys.modules, "gepa", package)
    monkeypatch.setitem(sys.modules, "gepa.optimize_anything", optimize_anything_module_fake)

    optimizer = PydanticOptimizeAnythingOptimizer(
        adapter=PydanticOptimizeAnythingAdapter(
            adapter=PydanticGEPAAdapter.from_dataset(
                dataset=_NoScoreDataset(cases=[], evaluators=[]),
                task=lambda case: {"answer": case.name},
                injections=[],
                objective=ScoreObjective(score_key="accuracy"),
            )
        ),
        initial_candidate=Candidate(values={"instructions": "seed"}),
        optimization_objective="custom objective",
    )
    with pytest.warns(DeprecationWarning, match="OptimizeAnythingConfig"):
        result = optimizer.optimize(
            trainset=[_Case("case_1")],
            config=GEPAConfig(
                budget=BudgetConfig(max_metric_calls=11),
                run=RunConfig(id="optional", directory=tmp_path / "optional"),
                evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
            ),
        )

    assert result.best_candidate.values == {"instructions": "optional"}
    assert result.best_score == 0.8


def test_optimize_anything_uses_shared_run_state_for_failure_resume_and_cache(
    tmp_path: Path,
) -> None:
    adapter = PydanticOptimizeAnythingAdapter(
        adapter=PydanticGEPAAdapter.from_dataset(
            dataset=_NoScoreDataset(cases=[], evaluators=[]),
            task=lambda case: {"answer": case.name},
            injections=[],
            objective=ScoreObjective(score_key="accuracy"),
        )
    )
    attempts: list[Any] = []

    def optimize(**kwargs: Any) -> _RawResult:
        attempts.append(kwargs["config"])
        if len(attempts) == 1:
            raise RuntimeError("interrupted")
        return _RawResult()

    optimizer = PydanticOptimizeAnythingOptimizer(
        adapter=adapter,
        initial_candidate=Candidate(values={"instructions": "seed"}),
        optimize_fn=cast("OptimizeAnythingFn[_Case]", optimize),
    )
    first = GEPAConfig(
        run=RunConfig(id="oa", directory=tmp_path / "oa"),
        evaluation_sets=EvaluationSetConfig(allow_same_train_validation=True),
    )
    with (
        pytest.warns(DeprecationWarning, match="OptimizeAnythingConfig"),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        optimizer.optimize(trainset=[_Case("case")], config=first)

    resumed = first.model_copy(update={"run": first.run.model_copy(update={"resume": "if_exists"})})
    with pytest.warns(DeprecationWarning, match="OptimizeAnythingConfig"):
        result = optimizer.optimize(trainset=[_Case("case")], config=resumed)
    with pytest.warns(DeprecationWarning, match="OptimizeAnythingConfig"):
        cached = optimizer.optimize(trainset=[_Case("case")], config=resumed)
    assert result.backend == "optimize_anything"
    assert cached.model_dump() == result.model_dump()
    assert len(attempts) == 2
    assert attempts[1].run_dir == str(tmp_path / "oa" / "backend" / "step-0" / "branch-0")

    def fail(**kwargs: Any) -> _RawResult:
        del kwargs
        raise RuntimeError("no store")

    with pytest.raises(RuntimeError, match="no store"):
        optimizer.model_copy(update={"optimize_fn": fail}).optimize(
            trainset=[_Case("case")],
            allow_same_train_val=True,
        )


def test_optimize_anything_side_info_helper_handles_missing_trajectories() -> None:
    adapter = PydanticGEPAAdapter.from_dataset(
        dataset=_NoScoreDataset(cases=[], evaluators=[]),
        task=lambda case: {"answer": case.name},
        injections=[],
        components=ComponentCatalog.from_components(
            [CandidateComponent(name="instructions", initial_text="seed")]
        ),
        objective=ScoreObjective(score_key="accuracy"),
    )
    eval_batch = adapter.evaluate([_Case("case_1")], {}, capture_traces=False)
    side_info = optimize_anything_adapter_module._side_info(
        adapter=adapter,
        candidate={},
        evaluation=eval_batch,
        index=0,
        score=1.0,
        objective_scores=None,
    )
    filtered_adapter = PydanticGEPAAdapter.from_dataset(
        dataset=_NoScoreDataset(cases=[], evaluators=[]),
        task=lambda case: {"answer": case.name},
        injections=[],
        components=ComponentCatalog.from_components(
            [CandidateComponent(name="instructions", initial_text="seed")]
        ),
        objective=ScoreObjective(score_key="accuracy"),
        asi_builder=PydanticEvalsASIBuilder(component_selector=lambda **_: []),
    )
    filtered_batch = filtered_adapter.evaluate([_Case("case_1")], {}, capture_traces=True)
    filtered_side_info = optimize_anything_adapter_module._side_info(
        adapter=filtered_adapter,
        candidate={},
        evaluation=filtered_batch,
        index=0,
        score=0.0,
        objective_scores=None,
    )

    assert side_info == {"scores": {"accuracy": 1.0}}
    assert filtered_side_info["scores"] == {"accuracy": 0.0}
    assert "instructions_specific_info" not in filtered_side_info


async def _async_value(value: str) -> str:
    await asyncio.sleep(0)
    return value


async def _async_error() -> str:
    await asyncio.sleep(0)
    raise RuntimeError("async boom")


def _fake_optimize(**kwargs: Any) -> Any:
    assert kwargs["seed_candidate"] == {"instructions": "seed"}
    assert kwargs["max_metric_calls"] == 3
    assert kwargs["reflection_lm_kwargs"] == {"temperature": 0.1}
    assert [case.name for case in kwargs["valset"]] == ["case_1"]
    return _RawResult()


def _fake_optimize_anything(**kwargs: Any) -> Any:
    config = kwargs["config"]
    assert config.engine == "gepa"
    assert config.max_evals == 9
    assert kwargs["objective"] == "runtime objective"
    assert kwargs["background"] == "runtime background"
    result = _RawResult()
    result.best_candidate = {"instructions": "opt-anything"}
    result.best_score = 0.95
    return result


@dataclass(frozen=True)
class _Case:
    name: str


@dataclass(frozen=True)
class _Score:
    value: float


@dataclass(frozen=True)
class _ReportCase:
    name: str
    inputs: dict[str, str]
    expected_output: dict[str, str]
    output: dict[str, str]
    scores: dict[str, _Score]


@dataclass(frozen=True)
class _Failure:
    name: str
    error_stacktrace: str


@dataclass(frozen=True)
class _NoScoreReportCase:
    name: str
    output: dict[str, str]


@dataclass(frozen=True)
class _Report:
    cases: list[_ReportCase] | tuple[_ReportCase, ...]
    failures: list[_Failure] | tuple[_Failure, ...]
    max_concurrency: int
    progress: bool
    evaluators: list[str]


@dataclass(frozen=True)
class _NamedReport(_Report):
    dataset_name: str | None


class _Dataset:
    def __init__(self, *, cases: Sequence[_Case], evaluators: Sequence[str]) -> None:
        self.cases = list(cases)
        self.evaluators = list(evaluators)

    def evaluate(self, task: Any, *, max_concurrency: int, progress: bool) -> _Report:
        report_cases = [
            _ReportCase(
                name=case.name,
                inputs={"name": case.name},
                expected_output={"answer": case.name},
                output=task(case),
                scores={"accuracy": _Score(1.0)},
            )
            for case in self.cases
        ]
        return _Report(
            cases=report_cases,
            failures=[_Failure(name="failed_case", error_stacktrace="boom")],
            max_concurrency=max_concurrency,
            progress=progress,
            evaluators=self.evaluators,
        )


class _TupleDataset(_Dataset):
    def evaluate(self, task: Any, *, max_concurrency: int, progress: bool) -> _Report:
        report = super().evaluate(task, max_concurrency=max_concurrency, progress=progress)
        return _Report(
            cases=tuple(report.cases),
            failures=tuple(report.failures),
            max_concurrency=max_concurrency,
            progress=progress,
            evaluators=self.evaluators,
        )


class _NamedDataset(_Dataset):
    def __init__(
        self, *, name: str | None = None, cases: Sequence[_Case], evaluators: Sequence[str]
    ) -> None:
        super().__init__(cases=cases, evaluators=evaluators)
        self.name = name

    def evaluate(self, task: Any, *, max_concurrency: int, progress: bool) -> _NamedReport:
        report = super().evaluate(task, max_concurrency=max_concurrency, progress=progress)
        return _NamedReport(
            cases=report.cases,
            failures=report.failures,
            max_concurrency=report.max_concurrency,
            progress=report.progress,
            evaluators=report.evaluators,
            dataset_name=self.name,
        )


class _NamedAttributeDataset(_Dataset):
    def __init__(self, *, cases: Sequence[_Case], evaluators: Sequence[str]) -> None:
        super().__init__(cases=cases, evaluators=evaluators)
        self.name = "fallback-suite"


class _NamedBadDataset:
    evaluators: list[str] = []
    name = "broken-suite"

    def __init__(self) -> None:
        pass


@dataclass(frozen=True)
class _NoScoreReport:
    cases: list[_NoScoreReportCase]
    max_concurrency: int
    progress: bool
    evaluators: list[str]


class _NoScoreDataset:
    def __init__(self, *, cases: Sequence[_Case], evaluators: Sequence[str]) -> None:
        self.cases = list(cases)
        self.evaluators = list(evaluators)

    def evaluate(self, task: Any, *, max_concurrency: int, progress: bool) -> _NoScoreReport:
        return _NoScoreReport(
            cases=[_NoScoreReportCase(name=case.name, output=task(case)) for case in self.cases],
            max_concurrency=max_concurrency,
            progress=progress,
            evaluators=self.evaluators,
        )


@dataclass(frozen=True)
class _FailureOnlyReport:
    failures: list[_Failure]
    max_concurrency: int
    progress: bool
    evaluators: list[str]


class _FailureOnlyDataset:
    def __init__(self, *, cases: Sequence[_Case], evaluators: Sequence[str]) -> None:
        self.cases = list(cases)
        self.evaluators = list(evaluators)

    def evaluate(self, task: Any, *, max_concurrency: int, progress: bool) -> _FailureOnlyReport:
        del task
        return _FailureOnlyReport(
            failures=[_Failure(name=case.name, error_stacktrace="boom") for case in self.cases],
            max_concurrency=max_concurrency,
            progress=progress,
            evaluators=self.evaluators,
        )


class _AsyncDataset:
    def __init__(self, *, cases: Sequence[_Case], evaluators: Sequence[str]) -> None:
        self.cases = list(cases)
        self.evaluators = list(evaluators)

    async def evaluate(self, task: Any, *, max_concurrency: int, progress: bool) -> _Report:
        await asyncio.sleep(0)
        report_cases = [
            _ReportCase(
                name=case.name,
                inputs={"name": case.name},
                expected_output={"answer": case.name},
                output=task(case),
                scores={"accuracy": _Score(1.0)},
            )
            for case in self.cases
        ]
        return _Report(
            cases=report_cases,
            failures=[],
            max_concurrency=max_concurrency,
            progress=progress,
            evaluators=self.evaluators,
        )


class _BadDataset:
    evaluators: list[str] = []

    def __init__(self) -> None:
        pass


class _NoEvaluateDataset:
    def __init__(self, *, cases: Sequence[_Case], evaluators: Sequence[str]) -> None:
        self.cases = list(cases)
        self.evaluators = list(evaluators)


class _FakeAgent:
    def __init__(self) -> None:
        self.seen_instructions: list[str] = []

    @contextmanager
    def override(self, *, instructions: str) -> Any:
        self.seen_instructions.append(instructions)
        yield


class _Recorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record_candidate_batch(
        self,
        *,
        candidate: Mapping[str, str],
        batch: Sequence[Any],
        report: Any,
        scores: Sequence[float],
        trajectories: Sequence[Any] | None,
    ) -> None:
        self.records.append(
            {
                "candidate": candidate,
                "batch": list(batch),
                "report": report,
                "scores": list(scores),
                "trajectories": list(trajectories) if trajectories is not None else None,
            }
        )


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[dict[str, EventValue]] = []

    def record_event(
        self,
        *,
        event_name: str,
        payload: Mapping[str, EventValue],
    ) -> None:
        self.events.append({"event_name": event_name, "payload": dict(payload)})


class _HistoryItem:
    candidate_id = "candidate_0"
    parent_ids = ["root"]
    generation = 1
    score = 0.8
    values = {"instructions": "history"}
    metadata = {"accepted": True}


class _RawResult:
    best_candidate = {"instructions": "optimized"}
    best_score = 0.9
    total_evals = 9
    eval_log: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    validation_scores = [0.8, 1.0]
    candidate_history = [_HistoryItem()]
    objective_scores = [{"accuracy": 0.9}]
    candidate_tree_dot = "digraph {}"
    candidate_tree_html = "<html></html>"


class _SparseResult:
    best_program = {"prompt": "best"}
    best_val_score = 0.7


class _HistoryResult:
    best_candidate: dict[str, str] = {}
    best_score = 0.0
    validation_scores: list[float] = []

    def __init__(self, summary: CandidateSummary) -> None:
        self.candidate_history = [summary]


class _CandidateObjectResult:
    best_score = 1.0

    def __init__(self, candidate: Candidate) -> None:
        self.best_candidate = candidate


class _ValidationOnlyResult:
    validation_scores = [0.25]


class _CandidateOnlyHistoryItem:
    candidate = {"instructions": "candidate"}


class _CandidateOnlyHistoryResult:
    best_score = 0.0
    candidate_history = [_CandidateOnlyHistoryItem()]


class _ScoreOnlyHistoryItem:
    score = 0.4


class _ScoreOnlyHistoryResult:
    best_score = 0.0
    candidate_history = [_ScoreOnlyHistoryItem()]


class _GepaLikeResult:
    best_candidate = {"instructions": "candidate 1"}
    best_idx = 1
    candidates = [
        {"instructions": "candidate 0"},
        {"instructions": "candidate 1"},
    ]
    parents = [[], [0]]
    val_aggregate_scores = [0.4, 0.9]
    val_subscores = [
        {"case_0": 0.4},
        {"case_1": 1.0},
    ]
    val_aggregate_subscores = [
        {"accuracy": 0.4},
        {"accuracy": 0.9},
    ]
    total_metric_calls = 12
    num_full_val_evals = 3
    run_dir = "runs/demo"
    seed = 7
    per_objective_best_candidates = {"accuracy": [1]}
    objective_pareto_front = {"accuracy": 0.9}

    def candidate_tree_dot(self) -> str:
        return "digraph { candidate_0 -> candidate_1 }"

    def candidate_tree_html(self) -> str:
        return "<html>tree</html>"


class _GepaLikeNoSubscoresResult:
    best_candidate = {"instructions": "candidate 0"}
    best_idx = 0
    candidates = [{"instructions": "candidate 0"}]
    parents = [[]]
    val_aggregate_scores = [0.5]
    val_aggregate_subscores = None


class _EmptyObjectiveResult:
    best_candidate = {"instructions": "candidate 0"}
    best_score = 0.0
    objective_scores: list[dict[str, float]] = []
    candidate_tree_dot = None
    candidate_tree_html = None


class _NoneObjectiveFrontierResult:
    best_candidate = {"instructions": "candidate 0"}
    best_score = 0.0
    per_objective_best_candidates = None
    objective_pareto_front = None


@dataclass(frozen=True)
class _EventPayload:
    value: str


class _ModelEventPayload(BaseModel):
    value: str
