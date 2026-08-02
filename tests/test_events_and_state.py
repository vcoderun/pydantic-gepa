from __future__ import annotations as _annotations

import json
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError
from rich.console import Console

from pydantic_gepa import Candidate, OptimizationDependencyError, RunConfig, RunStoreError
from pydantic_gepa.events import (
    BackendProgress,
    CandidateNormalized,
    Event,
    RunCompleted,
    RunStarted,
    StageCompleted,
    StageStarted,
    compose_observers,
    event_payload,
)
from pydantic_gepa.observers import (
    LogfireObserver,
    RichProgress,
    autobench_observer,
    callback_observer,
    logfire_observer,
    rich_progress,
)
from pydantic_gepa.orchestration.models import BudgetUsage, PlanResult, StageResult
from pydantic_gepa.recorder import GEPAEventBridge
from pydantic_gepa.state import (
    CompatibilityFingerprint,
    FileRunStore,
    RunState,
    content_fingerprint,
)
from pydantic_gepa.values import JsonValue


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


def test_typed_gepa_bridge_supports_observers_and_rejects_an_empty_target() -> None:
    with pytest.raises(ValueError, match="requires on_event or recorder"):
        GEPAEventBridge()

    events: list[Event] = []
    bridge = GEPAEventBridge(run_id="typed", on_event=events.append)
    bridge.on_iteration_start({"iteration": 1})
    bridge.on_evaluation_end({"candidate_idx": 2, "scores": "invalid"})
    bridge.on_budget_updated({"metric_calls_used": True, "metric_calls_remaining": "unknown"})

    assert events[0] == BackendProgress(
        run_id="typed",
        name="iteration_start",
        metadata={"iteration": 1},
    )
    assert events[1].kind == "candidate.evaluated"
    assert events[2].kind == "budget.updated"


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
