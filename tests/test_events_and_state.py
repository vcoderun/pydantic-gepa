from __future__ import annotations as _annotations

import json
import sys
import types
from collections.abc import Mapping
from contextvars import Context
from dataclasses import dataclass
from datetime import timedelta
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console

from pydantic_gepa import Candidate, OptimizationDependencyError, Plan, RunConfig, RunStoreError
from pydantic_gepa.eventing import budget_snapshot
from pydantic_gepa.events import (
    BackendProgress,
    BudgetSnapshot,
    BudgetUpdated,
    CandidateNormalized,
    EvaluationCompleted,
    Event,
    IterationStarted,
    RunCompleted,
    RunStarted,
    StageCompleted,
    StageStarted,
    _dispatcher,
    compose_observers,
    event_payload,
    subscribe,
)
from pydantic_gepa.observers import (
    LogfireObserver,
    RichProgress,
    autobench_observer,
    callback_observer,
    logfire_observer,
    rich_progress,
)
from pydantic_gepa.orchestration.models import (
    BudgetUsage,
    PlanResult,
    Stage,
    StageOutput,
    StageResult,
)
from pydantic_gepa.recorder import GEPAEventBridge
from pydantic_gepa.results import BudgetSummary
from pydantic_gepa.state import (
    CompatibilityFingerprint,
    FileRunStore,
    RunState,
    content_fingerprint,
)
from pydantic_gepa.values import JsonValue

if TYPE_CHECKING:
    from gepa.core.callbacks import (
        BudgetUpdatedEvent,
        EvaluationEndEvent,
        IterationStartEvent,
    )


class StoredResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    value: str


@dataclass(frozen=True)
class FingerprintValue:
    name: str


def candidate(value: str = "seed") -> Candidate:
    return Candidate(values={"instructions": value}, id=value)


def stage_result() -> StageResult:
    seed = candidate()
    output = candidate("better")
    return StageResult(
        stage_id="prompt/stage",
        status="completed",
        input_candidate=seed,
        output_candidate=output,
        target_components=("instructions",),
        frozen_components=(),
        score=0.8,
        budget=BudgetUsage(limit=2, used=1, reported=1, exhausted=False),
    )


def test_events_are_serializable_and_observers_have_explicit_failure_policies() -> None:
    seen: list[Event] = []
    started = RunStarted(run_id="demo", seed=candidate())
    normalized = CandidateNormalized(run_id="demo", candidate=candidate("normalized"))

    notify = compose_observers(seen.append, lambda event: seen.append(event))
    notify(started)
    assert seen == [started, started]
    assert event_payload(normalized)["kind"] == "candidate.normalized"

    def fail(_event: Event) -> None:
        raise RuntimeError("observer failed")

    with pytest.raises(RuntimeError, match="observer failed"):
        compose_observers(fail)(started)
    with pytest.warns(RuntimeWarning, match="observer"):
        compose_observers(fail, on_error="warn")(started)
    compose_observers(fail, on_error="ignore")(started)


def test_budget_snapshots_preserve_normalized_cost_and_call_accounting() -> None:
    snapshot = budget_snapshot(
        BudgetSummary(
            metric_calls=4,
            metric_call_limit=8,
            reflection_cost=0.25,
            evaluation_cost=0.75,
            total_cost=1.0,
        )
    )
    completed = RunCompleted(run_id="budgeted", budget=snapshot)

    assert snapshot == BudgetSnapshot(
        evaluation_calls=4,
        evaluation_call_limit=8,
        optimizer_cost=0.25,
        evaluation_cost=0.75,
        total_cost=1.0,
    )
    assert event_payload(completed)["budget"] == {
        "evaluation_calls": 4,
        "evaluation_call_limit": 8,
        "optimizer_cost": 0.25,
        "optimizer_cost_limit": None,
        "evaluation_cost": 0.75,
        "total_cost": 1.0,
    }

    explicit = budget_snapshot(
        BudgetSummary(
            evaluation_calls=3,
            evaluation_call_limit=6,
            optimizer_cost=0.5,
            optimizer_cost_limit=2.0,
        )
    )
    assert explicit.evaluation_calls == 3
    assert explicit.evaluation_call_limit == 6
    assert explicit.optimizer_cost == 0.5
    assert explicit.optimizer_cost_limit == 2.0


def test_typed_gepa_bridge_supports_observers_and_rejects_an_empty_target() -> None:
    with pytest.raises(ValueError, match="requires on_event or recorder"):
        GEPAEventBridge()

    events: list[Event] = []
    bridge = GEPAEventBridge(run_id="typed", on_event=events.append)
    bridge.on_iteration_start(
        cast(
            "IterationStartEvent",
            {"iteration": 1, "state": None, "trainset_loader": None},
        )
    )
    bridge.on_evaluation_end(
        cast(
            "EvaluationEndEvent",
            {
                "iteration": 1,
                "candidate_idx": 2,
                "scores": [0.25, 0.75],
                "has_trajectories": False,
                "parent_ids": [0],
                "outputs": [],
                "trajectories": None,
                "objective_scores": None,
                "is_seed_candidate": False,
            },
        )
    )
    bridge.on_budget_updated(
        cast(
            "BudgetUpdatedEvent",
            {
                "iteration": 1,
                "metric_calls_used": 2,
                "metric_calls_delta": 2,
                "metric_calls_remaining": 3,
            },
        )
    )

    assert isinstance(events[0], IterationStarted)
    assert events[0].iteration == 1
    assert isinstance(events[1], EvaluationCompleted)
    assert events[1].scores == (0.25, 0.75)
    assert events[1].parent_ids == ("0",)
    assert isinstance(events[2], BudgetUpdated)
    assert events[2].used == 2
    assert all(event.engine == "gepa" for event in events)


def test_subscriptions_are_context_local_snapshotted_ordered_and_close_idempotently() -> None:
    events: list[Event] = []
    handle = subscribe(events.append)

    def finish(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        return StageOutput(candidate=candidate, score=1.0, metric_calls=1)

    plan = Plan(
        Stage("prompt", ("instructions",), finish),
        initial_candidate=candidate(),
    )
    result = plan.run(run=RunConfig(id="subscribed"))
    handle.close()
    handle.close()

    assert result.effective_score == 1.0
    assert handle.closed is True
    assert [event.sequence for event in events] == list(range(len(events)))
    assert all(event.execution_id == events[0].execution_id for event in events)
    assert all(event.backend == "plan" for event in events)
    assert all(event.occurred_at.utcoffset() == timedelta(0) for event in events)
    assert [event.monotonic_ns for event in events] == sorted(
        event.monotonic_ns for event in events
    )

    observed_count = len(events)
    plan.run(run=RunConfig(id="after-close"))
    assert len(events) == observed_count

    isolated_events: list[Event] = []
    isolated_handle = subscribe(isolated_events.append)
    Context().run(lambda: plan.run(run=RunConfig(id="isolated")))
    isolated_handle.close()
    assert isolated_events == []


def test_subscription_and_local_observer_error_policies_are_independent() -> None:
    local_events: list[Event] = []
    integration_events: list[Event] = []

    def fail(_event: Event) -> None:
        raise RuntimeError("observer failed")

    plan = Plan(
        Stage(
            "prompt",
            ("instructions",),
            lambda candidate, _limit: StageOutput(
                candidate=candidate,
                score=1.0,
                metric_calls=1,
            ),
        ),
        initial_candidate=candidate(),
    )
    with (
        subscribe(integration_events.append),
        pytest.raises(RuntimeError, match="observer failed"),
    ):
        plan.run(run=RunConfig(id="local-failure"), on_event=fail)
    assert [event.kind for event in integration_events] == ["run.started"]

    with subscribe(fail, on_error="ignore"):
        plan.run(run=RunConfig(id="ignored-integration"), on_event=local_events.append)
    assert local_events[0].kind == "run.started"
    assert local_events[-1].kind == "run.completed"

    warned = False

    def fail_once(_event: Event) -> None:
        nonlocal warned
        if not warned:
            warned = True
            raise RuntimeError("warned observer")

    with (
        subscribe(fail_once, on_error="warn"),
        pytest.warns(RuntimeWarning, match="warned observer"),
    ):
        plan.run(run=RunConfig(id="warned-integration"))

    with (
        subscribe(fail, on_error="raise"),
        pytest.raises(RuntimeError, match="observer failed"),
    ):
        _dispatcher(run_id="subscriber-failure", backend="plan").emit(
            RunStarted(
                run_id="subscriber-failure",
                execution_id="explicit-execution",
                backend="plan",
            )
        )


def test_nested_plans_inherit_parent_execution_and_stage_scope() -> None:
    events: list[Event] = []
    child = Plan(
        Stage(
            "child-stage",
            ("instructions",),
            lambda candidate, _limit: StageOutput(
                candidate=candidate,
                score=1.0,
                metric_calls=1,
            ),
        ),
        initial_candidate=candidate(),
    )

    def run_child(candidate_value: Candidate, limit: int) -> StageOutput:
        del limit
        result = child.run(run=RunConfig(id="child"))
        assert result.effective_score is not None
        return StageOutput(
            candidate=result.final_candidate,
            score=result.effective_score,
            metric_calls=result.total_metric_calls,
        )

    parent = Plan(
        Stage("parent-stage", ("instructions",), run_child),
        initial_candidate=candidate(),
    )
    with subscribe(events.append):
        parent.run(run=RunConfig(id="parent"))

    parent_started = next(
        event for event in events if event.kind == "run.started" and event.run_id == "parent"
    )
    child_events = [event for event in events if event.run_id == "child"]
    assert child_events
    assert child_events[0].sequence == 0
    assert all(event.parent_execution_id == parent_started.execution_id for event in child_events)
    assert child_events[0].stage_id == "parent-stage"
    assert all(event.stage_id in {"parent-stage", "child-stage"} for event in child_events)


def test_content_fingerprints_normalize_models_dataclasses_sets_and_opaque_values() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "opaque"

    first = {
        "model": StoredResult(value="x"),
        "dataclass": FingerprintValue(name="demo"),
        "set": {3, 1, 2},
        "opaque": Opaque(),
        "scalar": None,
    }
    second = {
        "scalar": None,
        "opaque": Opaque(),
        "set": {2, 3, 1},
        "dataclass": FingerprintValue(name="demo"),
        "model": StoredResult(value="x"),
    }
    assert content_fingerprint(first) == content_fingerprint(second)


def test_file_run_store_writes_and_resumes_owned_state(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    fingerprint = CompatibilityFingerprint.from_dimensions({"dataset": "v1"})
    store = FileRunStore(directory, run_id="demo")
    assert store.load_state() is None
    state = store.prepare(fingerprint=fingerprint, initial_candidate=candidate())

    candidate_path = store.write_candidate(candidate("better"))
    stage_path = store.write_stage(stage_result())
    result_path = store.write_result(StoredResult(value="done"))
    advanced = state.model_copy(
        update={
            "next_stage": 1,
            "metric_calls": 1,
            "accepted_candidate": candidate("better"),
            "stages": (stage_result().model_dump(mode="json"),),
        }
    )
    store.checkpoint(advanced)

    assert candidate_path.is_file()
    assert stage_path.name == "prompt_stage.json"
    assert result_path.is_file()
    assert store.backend_directory == directory / "backend"
    assert store.load_result(StoredResult) == StoredResult(value="done")

    resumed = FileRunStore(directory, run_id="demo", resume="if_exists").prepare(
        fingerprint=fingerprint,
        initial_candidate=candidate(),
    )
    assert resumed.model_dump() == advanced.model_dump()
    assert resumed.resumed is True
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint"]["dimensions"] == {"dataset": "v1"}


def test_file_run_store_rejects_unsafe_incompatible_and_corrupted_state(
    tmp_path: Path,
) -> None:
    fingerprint = CompatibilityFingerprint.from_dimensions({"dataset": "v1"})
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "user.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(RunStoreError, match="Refusing to own non-empty"):
        FileRunStore(unowned, run_id="demo").prepare(
            fingerprint=fingerprint,
            initial_candidate=candidate(),
        )

    directory = tmp_path / "owned"
    store = FileRunStore(directory, run_id="demo")
    state = store.prepare(fingerprint=fingerprint, initial_candidate=candidate())
    with pytest.raises(RunStoreError, match="already exists"):
        store.prepare(fingerprint=fingerprint, initial_candidate=candidate())
    with pytest.raises(RunStoreError, match="different run id"):
        store.checkpoint(state.model_copy(update={"run_id": "other"}))
    with pytest.raises(RunStoreError, match="non-empty stage_id"):
        store.write_stage(StoredResult(value="not-a-stage"))

    with pytest.raises(RunStoreError, match="incompatible.*dataset"):
        FileRunStore(directory, run_id="demo", resume="if_exists").prepare(
            fingerprint=CompatibilityFingerprint.from_dimensions({"dataset": "v2"}),
            initial_candidate=candidate(),
        )
    with pytest.raises(RunStoreError, match="does not match this run"):
        FileRunStore(directory, run_id="other", resume="if_exists").prepare(
            fingerprint=fingerprint,
            initial_candidate=candidate(),
        )

    (directory / "state.json").write_text("not json", encoding="utf-8")
    with pytest.raises(RunStoreError, match="corrupted"):
        FileRunStore(directory, run_id="demo", resume="if_exists").prepare(
            fingerprint=fingerprint,
            initial_candidate=candidate(),
        )


def test_file_run_store_fresh_mode_preserves_unowned_files_and_requires_ownership(
    tmp_path: Path,
) -> None:
    fingerprint = CompatibilityFingerprint.from_dimensions({"dataset": "v1"})
    directory = tmp_path / "run"
    store = FileRunStore(directory, run_id="demo")
    store.prepare(fingerprint=fingerprint, initial_candidate=candidate())
    store.backend_directory.mkdir()
    (directory / "notes.txt").write_text("keep", encoding="utf-8")
    (directory / ".state.json.tmp").write_text("partial", encoding="utf-8")

    fresh = FileRunStore(directory, run_id="demo", fresh=True)
    reset_state = fresh.prepare(fingerprint=fingerprint, initial_candidate=candidate("new"))
    assert reset_state.accepted_candidate == candidate("new")
    assert (directory / "notes.txt").read_text(encoding="utf-8") == "keep"
    assert not (directory / ".state.json.tmp").exists()

    (directory / ".pydantic-gepa-run").write_text("bad marker", encoding="utf-8")
    with pytest.raises(RunStoreError, match="marker.*corrupted"):
        fresh.reset()

    with pytest.raises(RunStoreError, match="is not owned"):
        FileRunStore(tmp_path / "missing", run_id="demo").reset()


def test_file_run_store_required_resume_empty_result_and_atomic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fingerprint = CompatibilityFingerprint.from_dimensions({"dataset": "v1"})
    directory = tmp_path / "run"
    store = FileRunStore(directory, run_id="demo", resume="required")
    with pytest.raises(RunStoreError, match="No resumable state"):
        store.prepare(fingerprint=fingerprint, initial_candidate=candidate())

    active = FileRunStore(tmp_path / "atomic", run_id="demo")
    active.prepare(fingerprint=fingerprint, initial_candidate=candidate())
    assert active.load_result(StoredResult) is None

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("pydantic_gepa.state.files.os.replace", fail_replace)
    with pytest.raises(RunStoreError, match="atomically write"):
        active.write_result(StoredResult(value="failure"))
    assert not (active.directory / ".result.json.tmp").exists()


def test_run_config_rejects_conflicting_or_missing_durable_run_modes() -> None:
    with pytest.raises(ValidationError, match="fresh cannot be combined"):
        RunConfig(directory=Path("run"), fresh=True, resume="if_exists")
    with pytest.raises(ValidationError, match="directory is required"):
        RunConfig(resume="required")


def test_run_state_rejects_negative_progress() -> None:
    with pytest.raises(ValidationError):
        RunState(
            run_id="demo",
            fingerprint=CompatibilityFingerprint.from_dimensions({}),
            accepted_candidate=candidate(),
            next_stage=-1,
        )


def test_plan_result_effective_score_prefers_final_rescore() -> None:
    result = PlanResult(
        initial_candidate=candidate(),
        final_candidate=candidate("final"),
        stages=(),
        score=0.5,
        final_score=0.8,
        total_metric_calls=0,
    )
    assert result.effective_score == 0.8
    assert result.model_copy(update={"final_score": None}).effective_score == 0.5


def test_rich_progress_supports_deterministic_and_interactive_modes() -> None:
    output = StringIO()
    console = Console(file=output, force_terminal=False, width=100)
    observer = RichProgress(console=console, interactive=False)
    observer(RunStarted(run_id="demo"))
    observer(
        BackendProgress(
            run_id="demo",
            stage_id="prompt",
            name="ignored",
        )
    )
    observer(StageCompleted(run_id="demo", stage_id="prompt", score=0.9))
    assert output.getvalue() == "[pydantic-gepa] stage.completed: prompt\n"

    interactive = RichProgress(console=console, interactive=True)
    interactive(RunStarted(run_id="demo"))
    interactive(StageStarted(run_id="demo", stage_id="prompt"))
    interactive(StageCompleted(run_id="demo", stage_id="unknown"))
    interactive(StageCompleted(run_id="demo", stage_id="prompt"))
    interactive(BackendProgress(run_id="demo", name="iteration"))
    interactive(RunCompleted(run_id="demo", score=0.9))
    assert "prompt" in interactive.tasks
    assert callable(rich_progress(console=console, interactive=False))
    assert RichProgress(console=console).interactive is False
    assert RichProgress(interactive=False).interactive is False


def test_logfire_autobench_and_callback_observers_receive_serialized_events() -> None:
    class Logfire:
        def __init__(self) -> None:
            self.records: list[dict[str, Any]] = []

        def info(self, message: str, **attributes: Any) -> None:
            self.records.append({"message": message, **attributes})

    class Recorder:
        def __init__(self) -> None:
            self.events: list[dict[str, JsonValue]] = []

        def record_optimization_event(self, event: Mapping[str, JsonValue]) -> None:
            self.events.append(dict(event))

    event = RunStarted(run_id="demo", seed=candidate())
    logger = Logfire()
    LogfireObserver(logfire=logger)(event)
    logfire_observer(logfire=logger)(event)
    recorder = Recorder()
    with pytest.warns(DeprecationWarning, match="native pydantic-gepa instrumentor"):
        autobench_observer(recorder)(event)
    callback_events: list[Mapping[str, JsonValue]] = []
    callback_observer(callback_events.append)(event)

    assert [record["event_kind"] for record in logger.records] == [
        "run.started",
        "run.started",
    ]
    assert recorder.events[0]["kind"] == "run.started"
    assert callback_events[0]["run_id"] == "demo"


def test_optional_observers_raise_actionable_dependency_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed = types.ModuleType("logfire")
    installed.__dict__["info"] = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "logfire", installed)
    assert LogfireObserver().logfire is installed
    monkeypatch.setitem(sys.modules, "rich.console", None)
    monkeypatch.setitem(sys.modules, "rich.progress", None)
    monkeypatch.setitem(sys.modules, "logfire", None)
    with pytest.raises(OptimizationDependencyError, match="Rich progress"):
        RichProgress()
    with pytest.raises(OptimizationDependencyError, match="Logfire events"):
        LogfireObserver()
