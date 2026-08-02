from __future__ import annotations as _annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

import pydantic_gepa.orchestration.execution as execution_module
from pydantic_gepa import (
    Budget,
    Candidate,
    CandidateComponent,
    ComponentCatalog,
    Plan,
    PlanError,
    PydanticGEPAAdapter,
    PydanticGEPAOptimization,
    PydanticGEPAOptimizer,
    RunConfig,
    RunStoreError,
    ScoreObjective,
    Stage,
)
from pydantic_gepa.events import Event
from pydantic_gepa.examples import EvalCaseView, PydanticEvaluator
from pydantic_gepa.optimizer import OptimizeFn
from pydantic_gepa.orchestration import (
    BudgetUsage,
    PlanSpec,
    StageOutput,
    StageResult,
    StageRun,
)
from pydantic_gepa.results import CandidateSummary, PydanticGEPAResult
from pydantic_gepa.values import SerializableValue


def test_plan_carries_candidates_through_sequential_and_grouped_stages() -> None:
    initial = Candidate(values={"planner": "p0", "writer": "w0", "tool": "t0"})
    history = CandidateSummary(candidate_id="p1", score=0.8)

    async def planner(candidate: Candidate, limit: int) -> StageOutput:
        assert limit == 4
        return StageOutput(
            candidate=candidate.model_copy(
                update={"values": {**candidate.values, "planner": "p1"}}
            ),
            score=0.8,
            metric_calls=2,
            history=(history,),
            checkpoint="runs/planner",
        )

    def generation(candidate: Candidate, limit: int) -> PydanticGEPAResult:
        assert candidate.values["planner"] == "p1"
        assert limit == 3
        return PydanticGEPAResult(
            best_candidate=candidate.model_copy(
                update={
                    "values": {
                        **candidate.values,
                        "writer": "w1",
                        "tool": "t1",
                    }
                }
            ),
            best_score=0.9,
        )

    async def stage_rescore(candidate: Candidate) -> float:
        return 1.0 if candidate.values["planner"] == "p1" else 0.0

    plan = Plan(
        Stage(
            "planner",
            ("planner",),
            planner,
            budget=Budget(max_metric_calls=4),
            rescore=stage_rescore,
            rescore_id="planner_rescore",
        ),
        Stage(
            "generation",
            ("writer", "tool"),
            generation,
            budget=Budget(max_metric_calls=3),
        ),
        initial_candidate=initial,
        budget=Budget(max_metric_calls=8),
        aggregate="weighted_mean",
        weights={"planner": 1.0, "generation": 3.0},
        final_rescore=lambda candidate: 1.0 if candidate.values["tool"] == "t1" else 0.0,
        final_rescore_id="final",
    )

    result = plan.run()

    assert result.final_candidate.values == {
        "planner": "p1",
        "writer": "w1",
        "tool": "t1",
    }
    assert result.score == pytest.approx(0.925)
    assert result.final_score == 1.0
    assert result.total_metric_calls == 5
    assert result.budget == BudgetUsage(
        limit=8,
        used=5,
        reported=5,
        exhausted=False,
    )
    assert result.stages[0].effective_score == 1.0
    assert result.stages[0].history == (history,)
    assert result.stages[0].checkpoint == "runs/planner"
    assert result.stages[0].frozen_components == ("tool", "writer")
    assert result.stages[1].budget.reported is None
    assert result.stages[1].budget.exhausted is True


def test_plan_supports_seed_overrides_rejected_candidates_and_initial_carry() -> None:
    initial = Candidate(values={"prompt": "p0", "tool": "t0"})
    seen: list[Candidate] = []

    def reject(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        seen.append(candidate)
        return StageOutput(
            candidate=candidate.model_copy(
                update={"values": {**candidate.values, "prompt": "rejected"}}
            ),
            score=0.1,
            metric_calls=1,
            accepted=False,
        )

    def update(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        seen.append(candidate)
        return StageOutput(
            candidate=candidate.model_copy(update={"values": {**candidate.values, "tool": "t1"}}),
            score=0.7,
            metric_calls=1,
        )

    plan = Plan.sequential(
        [
            Stage(
                "prompt",
                ("prompt",),
                reject,
                seed=Candidate(values={"prompt": "seed", "tool": "t0"}),
            ),
            Stage("tool", ("tool",), update),
        ],
        initial_candidate=initial,
        aggregate="min",
    )

    result = plan.run(
        seed=initial,
    )

    assert seen[0].values["prompt"] == "seed"
    assert seen[1] == initial
    assert result.final_candidate.values == {"prompt": "p0", "tool": "t1"}
    assert result.score == 0.1

    reset_result = Plan(
        Stage("prompt", ("prompt",), reject),
        Stage("tool", ("tool",), update),
        initial_candidate=initial,
        carry_forward="initial",
    ).run()
    assert reset_result.final_candidate == initial


def test_plan_enforces_frozen_values_and_failure_stop_policies() -> None:
    initial = Candidate(values={"prompt": "p0", "tool": "t0"})

    def invalid(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        return StageOutput(
            candidate=candidate.model_copy(update={"values": {"prompt": "p1", "tool": "changed"}}),
            score=1.0,
            metric_calls=1,
        )

    def valid(candidate: Candidate, limit: int) -> StageOutput:
        return StageOutput(candidate=candidate, score=0.5, metric_calls=limit)

    stopped = Plan(
        Stage("invalid", ("prompt",), invalid, budget=Budget(max_metric_calls=2)),
        Stage("valid", ("tool",), valid),
        initial_candidate=initial,
    ).run()

    assert stopped.stop_reason == "stage_failed:invalid"
    assert len(stopped.stages) == 1
    assert stopped.stages[0].status == "failed"
    assert stopped.stages[0].error is not None
    assert stopped.stages[0].error.type == "PlanError"
    assert stopped.stages[0].budget.used == 2
    assert stopped.score is None

    continued = Plan(
        Stage("invalid", ("prompt",), invalid, budget=Budget(max_metric_calls=2)),
        Stage("valid", ("tool",), valid, budget=Budget(max_metric_calls=1)),
        initial_candidate=initial,
        stop="continue",
        aggregate="mean",
    ).run()
    assert [stage.status for stage in continued.stages] == ["failed", "completed"]
    assert continued.final_candidate == initial
    assert continued.total_metric_calls == 3
    assert continued.score == 0.5


def test_plan_stops_before_a_stage_when_shared_budget_is_exhausted() -> None:
    initial = Candidate(values={"a": "0", "b": "0"})

    def consume(candidate: Candidate, limit: int) -> StageOutput:
        return StageOutput(candidate=candidate, score=1.0, metric_calls=limit)

    result = Plan(
        Stage("a", ("a",), consume),
        Stage("b", ("b",), consume),
        initial_candidate=initial,
        budget=Budget(max_metric_calls=2),
    ).run()

    assert len(result.stages) == 1
    assert result.stop_reason == "shared_budget_exhausted"
    assert result.budget is not None
    assert result.budget.exhausted is True


def test_plan_supports_custom_aggregation_and_snapshot_reconstruction() -> None:
    initial = Candidate(values={"prompt": "p0"})

    def run(candidate: Candidate, limit: int) -> StageOutput:
        return StageOutput(candidate=candidate, score=float(limit), metric_calls=1)

    def aggregate(stages: Sequence[StageResult]) -> float:
        return sum(stage.score or 0.0 for stage in stages) + 1.0

    plan = Plan(
        Stage(
            "prompt",
            ("prompt",),
            run,
            budget=Budget(max_metric_calls=2),
            run_id="run_prompt",
        ),
        initial_candidate=initial,
        aggregate=aggregate,
        aggregate_id="aggregate_scores",
    )
    snapshot = plan.snapshot()
    restored = Plan.from_snapshot(
        PlanSpec.model_validate_json(snapshot.model_dump_json()),
        runs={"run_prompt": run},
        aggregates={"aggregate_scores": aggregate},
    )

    assert restored.run().score == 3.0
    assert restored.snapshot() == snapshot

    with pytest.raises(PlanError, match="No custom aggregation"):
        Plan.from_snapshot(snapshot, runs={"run_prompt": run})


def test_stage_snapshot_reconstructs_rescore_and_reports_missing_registrations() -> None:
    def run(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        return StageOutput(candidate=candidate, score=1.0, metric_calls=1)

    def rescore(candidate: Candidate) -> float:
        del candidate
        return 0.9

    spec = Stage(
        "prompt",
        ("prompt",),
        run,
        run_id="runner",
        rescore=rescore,
        rescore_id="rescore",
    ).snapshot()
    rebuilt = spec.build(runs={"runner": run}, rescores={"rescore": rescore})
    assert rebuilt.run_id == "runner"
    assert rebuilt.rescore is rescore
    assert run(Candidate(), 1).score == 1.0
    assert rescore(Candidate()) == 0.9

    with pytest.raises(PlanError, match="No stage runner"):
        spec.build(runs={})
    with pytest.raises(PlanError, match="No stage rescore"):
        spec.build(runs={"runner": run})

    final_spec = PlanSpec(
        stages=(spec.model_copy(update={"rescore_id": None}),),
        initial_candidate=Candidate(values={"prompt": "p0"}),
        final_rescore_id="final",
    )
    with pytest.raises(PlanError, match="No final rescore"):
        Plan.from_snapshot(final_spec, runs={"runner": run})


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda run: Stage("", ("prompt",), run), "id cannot be empty"),
        (lambda run: Stage("empty", (), run), "at least one component"),
        (
            lambda run: Stage("duplicate", ("prompt", "prompt"), run),
            "duplicate target",
        ),
        (
            lambda run: Stage("frozen", ("prompt",), run, frozen=("prompt",)),
            "targets frozen",
        ),
        (
            lambda run: Stage(
                "rescore",
                ("prompt",),
                run,
                rescore_id="missing",
            ),
            "rescore_id requires",
        ),
    ],
)
def test_stage_validates_its_definition(
    factory: Callable[[StageRun], Stage],
    message: str,
) -> None:
    def run(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        return StageOutput(candidate=candidate, score=1.0)

    with pytest.raises(PlanError, match=message):
        factory(run)
    assert run(Candidate(), 1).score == 1.0


def test_plan_validates_definition_and_weighted_aggregation() -> None:
    initial = Candidate(values={"prompt": "p0"})

    def run(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        return StageOutput(candidate=candidate, score=1.0, metric_calls=1)

    stage = Stage("prompt", ("prompt",), run)
    with pytest.raises(PlanError, match="at least one stage"):
        Plan(initial_candidate=initial)
    with pytest.raises(PlanError, match="unique"):
        Plan(stage, stage, initial_candidate=initial)
    with pytest.raises(PlanError, match="aggregate_id"):
        Plan(stage, initial_candidate=initial, aggregate=lambda stages: 1.0)
    with pytest.raises(PlanError, match="requires stage weights"):
        Plan(stage, initial_candidate=initial, aggregate="weighted_mean")
    with pytest.raises(PlanError, match="final_rescore_id requires"):
        Plan(stage, initial_candidate=initial, final_rescore_id="final")
    with pytest.raises(PlanError, match="positive total weight"):
        Plan(
            stage,
            initial_candidate=initial,
            aggregate="weighted_mean",
            weights={"prompt": 0.0},
        ).run()


def test_plan_reports_stage_seed_and_output_contract_failures() -> None:
    initial = Candidate(values={"prompt": "p0", "tool": "t0"})

    def missing(candidate: Candidate, limit: int) -> StageOutput:
        del candidate, limit
        return StageOutput(
            candidate=Candidate(values={"tool": "t0"}),
            score=1.0,
        )

    with pytest.raises(PlanError, match="seed changes frozen"):
        Plan(
            Stage(
                "prompt",
                ("prompt",),
                missing,
                seed=Candidate(values={"prompt": "seed", "tool": "changed"}),
            ),
            initial_candidate=initial,
        ).run()

    with pytest.raises(PlanError, match="no seed value"):
        Plan(
            Stage("missing", ("unknown",), missing),
            initial_candidate=initial,
        ).run()

    result = Plan(
        Stage("prompt", ("prompt",), missing),
        initial_candidate=initial,
    ).run()
    assert result.stages[0].error is not None
    assert "did not return target" in result.stages[0].error.message

    preserved_seed = Plan(
        Stage(
            "prompt",
            ("prompt",),
            lambda candidate, limit: StageOutput(
                candidate=candidate,
                score=float(limit),
                metric_calls=1,
            ),
            seed=Candidate(values={"tool": "t0"}),
        ),
        initial_candidate=initial,
    ).run()
    assert preserved_seed.final_candidate.values["prompt"] == "p0"


def test_budget_models_reject_invalid_usage() -> None:
    with pytest.raises(ValidationError):
        Budget(max_metric_calls=0)
    with pytest.raises(ValidationError, match="cannot exceed"):
        BudgetUsage(limit=1, used=2, exhausted=True)
    with pytest.raises(ValidationError, match="Reported metric calls"):
        BudgetUsage(limit=1, used=1, reported=2, exhausted=True)


def test_stage_from_optimization_uses_carried_candidate_as_gepa_seed() -> None:
    seen: list[dict[str, str]] = []

    def fake_optimize(**kwargs: SerializableValue) -> _RawResult:
        seed = kwargs["seed_candidate"]
        assert isinstance(seed, Mapping)
        values = {str(key): str(value) for key, value in seed.items()}
        seen.append(values)
        return _RawResult({**values, "prompt": "optimized"})

    component = CandidateComponent(name="prompt", initial_text="initial")
    components = ComponentCatalog.from_components([component])
    raw_adapter: PydanticGEPAAdapter[_Case, str, None] = PydanticGEPAAdapter.from_dataset(
        dataset=_Dataset(),
        task=lambda case: case.inputs,
        injections=[],
        objective=ScoreObjective(score_key="score"),
        components=components,
    )
    adapter = cast(
        "PydanticGEPAAdapter[EvalCaseView[str, str, None], str, PydanticEvaluator[str, str, None]]",
        raw_adapter,
    )
    optimizer = PydanticGEPAOptimizer(
        adapter=adapter,
        initial_candidate=components.to_candidate(),
        optimize_fn=cast("OptimizeFn", fake_optimize),
    )
    case: EvalCaseView[str, str, None] = _Case(inputs="input")
    cases: list[EvalCaseView[str, str, None]] = [case]
    optimization = PydanticGEPAOptimization(
        adapter=adapter,
        optimizer=optimizer,
        trainset=cases,
        valset=cases,
        initial_candidate=components.to_candidate(),
    )
    stage = Stage.from_optimization(
        "prompt",
        optimization,
        components=("prompt",),
        budget=Budget(max_metric_calls=2),
    )
    carried = Candidate(values={"prompt": "carried"})

    result = Plan(stage, initial_candidate=carried).run()

    assert seen == [{"prompt": "carried"}]
    assert result.stages[0].error is None
    assert result.final_candidate.values["prompt"] == "optimized"
    assert optimization.objective == ScoreObjective(score_key="score")
    assert (
        _Dataset()
        .evaluate(
            lambda case: case.inputs,
            max_concurrency=1,
            progress=False,
        )
        .cases
        == ()
    )


async def test_plan_resumes_between_stages_and_returns_completed_result_without_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def missing_version(distribution: str) -> str:
        raise PackageNotFoundError(distribution)

    monkeypatch.setattr(execution_module, "version", missing_version)
    calls: list[str] = []
    initial = Candidate(values={"prompt": "p0", "tool": "t0"})

    def prompt(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        calls.append("prompt")
        return StageOutput(
            candidate=candidate.model_copy(update={"values": {**candidate.values, "prompt": "p1"}}),
            score=0.7,
            metric_calls=1,
        )

    async def interrupted(_candidate: Candidate, _limit: int) -> StageOutput:
        calls.append("interrupted")
        raise asyncio.CancelledError

    first = Plan(
        Stage("prompt", ("prompt",), prompt, run_id="prompt"),
        Stage("tool", ("tool",), interrupted, run_id="tool"),
        initial_candidate=initial,
    )
    with pytest.raises(asyncio.CancelledError):
        await first.arun(run=RunConfig(id="resume", directory=tmp_path / "run"))

    def tool(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        calls.append("tool")
        assert candidate.values["prompt"] == "p1"
        return StageOutput(
            candidate=candidate.model_copy(update={"values": {**candidate.values, "tool": "t1"}}),
            score=0.9,
            metric_calls=1,
        )

    resumed_events: list[Event] = []
    second = Plan(
        Stage("prompt", ("prompt",), prompt, run_id="prompt"),
        Stage("tool", ("tool",), tool, run_id="tool"),
        initial_candidate=initial,
    )
    config = RunConfig(id="resume", directory=tmp_path / "run", resume="if_exists")
    result = await second.arun(run=config, on_event=[resumed_events.append])

    assert calls == ["prompt", "interrupted", "tool"]
    assert result.final_candidate.values == {"prompt": "p1", "tool": "t1"}
    assert [event.kind for event in resumed_events] == [
        "checkpoint.resumed",
        "run.started",
        "stage.started",
        "candidate.normalized",
        "candidate.evaluated",
        "candidate.accepted",
        "budget.updated",
        "stage.completed",
        "checkpoint.written",
        "checkpoint.written",
        "run.completed",
    ]

    calls.clear()
    completed_events: list[Event] = []
    restored = await second.arun(run=config, on_event=completed_events.append)
    assert restored == result
    assert calls == []
    assert [event.kind for event in completed_events] == [
        "checkpoint.resumed",
        "run.completed",
    ]


def test_plan_emits_fresh_reset_and_checkpoint_rejection_events(tmp_path: Path) -> None:
    def run(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        return StageOutput(candidate=candidate, score=0.5, metric_calls=1)

    plan = Plan(
        Stage("prompt", ("prompt",), run, run_id="prompt"),
        initial_candidate=Candidate(values={"prompt": "seed"}),
    )
    directory = tmp_path / "run"
    plan.run(run=RunConfig(id="events", directory=directory))

    fresh_events: list[Event] = []
    plan.run(
        run=RunConfig(id="events", directory=directory, fresh=True),
        on_event=fresh_events.append,
    )
    assert fresh_events[0].kind == "checkpoint.reset"

    rejected_events: list[Event] = []
    with pytest.raises(RunStoreError, match="incompatible"):
        plan.run(
            run=RunConfig(
                id="events",
                directory=directory,
                resume="if_exists",
                compatibility={"dataset": "changed"},
            ),
            on_event=rejected_events.append,
        )
    assert [event.kind for event in rejected_events] == ["checkpoint.rejected"]


class _Report:
    cases: tuple[()] = ()
    failures: tuple[()] = ()


class _RawResult:
    best_score = 0.95
    total_metric_calls = 2

    def __init__(self, best_candidate: dict[str, str]) -> None:
        self.best_candidate = best_candidate


@dataclass(frozen=True, slots=True)
class _Case:
    inputs: str
    name: str | None = None
    expected_output: str | None = None
    metadata: None = None


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


def test_plan_reconstructs_builtin_aggregation_and_final_rescore() -> None:
    def run(candidate: Candidate, limit: int) -> StageOutput:
        del limit
        return StageOutput(candidate=candidate, score=0.4, metric_calls=1)

    def final(candidate: Candidate) -> float:
        del candidate
        return 0.6

    stage = Stage("prompt", ("prompt",), run, run_id="run")
    plan = Plan(
        stage,
        initial_candidate=Candidate(values={"prompt": "p0"}),
        final_rescore=final,
        final_rescore_id="final",
    )
    restored = Plan.from_snapshot(
        plan.snapshot(),
        runs={"run": run},
        rescores={"final": final},
    )

    assert restored.run().final_score == 0.6


def test_stage_result_prefers_final_score() -> None:
    candidate = Candidate(values={"prompt": "p0"})
    result = StageResult(
        stage_id="prompt",
        status="skipped",
        input_candidate=candidate,
        output_candidate=candidate,
        target_components=("prompt",),
        frozen_components=(),
        score=0.2,
        final_score=0.9,
        budget=BudgetUsage(limit=0, used=0, exhausted=True),
    )
    assert result.effective_score == 0.9
