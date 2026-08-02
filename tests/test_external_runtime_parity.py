from __future__ import annotations as _annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic_gepa import (
    Budget,
    Candidate,
    Context,
    DataSplit,
    Evaluation,
    EvaluationConfig,
    Example,
    MetricResult,
    Plan,
    RunConfig,
    Runtime,
    Stage,
)
from pydantic_gepa.events import Event
from pydantic_gepa.orchestration import StageOutput


def test_generic_external_runtime_parity_through_public_seams(tmp_path: Path) -> None:
    active_values: dict[str, str] = {}
    restored_snapshots: list[dict[str, str]] = []

    @contextmanager
    def apply(candidate: Candidate) -> Iterator[None]:
        previous = dict(active_values)
        active_values.update(candidate.values)
        try:
            yield
        finally:
            active_values.clear()
            active_values.update(previous)
            restored_snapshots.append(dict(active_values))

    runtime = Runtime(
        lambda value: f"{active_values['planner']}:{value}",
        scope=apply,
        required_components=("planner",),
        identity="parity-runtime",
    )

    def repeated_metric(ctx: Context[str, str, dict[str, str]]) -> MetricResult:
        first = ctx.run()
        second = ctx.run_with(ctx.example.inputs.upper())
        ctx.record_trace(
            "planner",
            kind="computed-input",
            input=first,
            output=second,
            metadata={"calls": 2},
        )
        return MetricResult(
            score=1.0 if first != second else 0.0,
            feedback="Uppercase probing produced distinct evidence.",
            side_info={"first": first, "second": second},
        )

    evaluation = Evaluation.controlled(
        runtime,
        repeated_metric,
        config=EvaluationConfig[str, str, dict[str, str]](cache="memory"),
        identity="parity-metric",
    )
    examples = tuple(
        Example(
            id=f"case-{index}",
            inputs=value,
            metadata={"slice": "parity"},
        )
        for index, value in enumerate(("alpha", "beta", "gamma", "delta"))
    )
    data = DataSplit.from_sets(
        train=examples[:2],
        validation=examples[2:],
        max_validation=1,
    )
    initial = Candidate(
        values={
            "planner": "plan-v0",
            "writer": "write-v0",
            "tool.search": "search-v0",
            "router": "route-v0",
        }
    )
    metric_result = evaluation.run(initial, data.train[0], stage_id="planner")
    cached_result = evaluation.run(initial, data.train[0], stage_id="planner")

    assert metric_result.invocation_count == 2
    assert metric_result.feedback["score"]
    assert metric_result.side_info["score"]["second"] == "plan-v0:ALPHA"
    assert metric_result.traces[0].component == "planner"
    assert cached_result.cache_hit is True
    assert active_values == {}
    assert restored_snapshots == [{}, {}]
    assert len(data.validation) == 1

    stage_inputs: list[dict[str, str]] = []

    def planner(candidate: Candidate, limit: int) -> StageOutput:
        stage_inputs.append(dict(candidate.values))
        return StageOutput(
            candidate=Candidate(values={**candidate.values, "planner": "plan-v1"}),
            score=metric_result.objectives["score"],
            metric_calls=min(limit, 2),
            checkpoint="planner/checkpoint",
        )

    def generation(candidate: Candidate, limit: int) -> StageOutput:
        stage_inputs.append(dict(candidate.values))
        assert candidate.values["planner"] == "plan-v1"
        assert candidate.values["router"] == "route-v0"
        return StageOutput(
            candidate=Candidate(
                values={
                    **candidate.values,
                    "writer": "write-v1",
                    "tool.search": "search-v1",
                }
            ),
            score=0.8,
            metric_calls=min(limit, 3),
            checkpoint="generation/checkpoint",
        )

    plan = Plan(
        Stage(
            "planner",
            components=("planner",),
            frozen=("router",),
            budget=Budget(max_metric_calls=2),
            run=planner,
        ),
        Stage(
            "generation",
            components=("writer", "tool.search"),
            frozen=("router",),
            budget=Budget(max_metric_calls=3),
            run=generation,
        ),
        initial_candidate=initial,
        budget=Budget(max_metric_calls=5),
        aggregate="mean",
    )
    events: list[Event] = []
    result = plan.run(
        run=RunConfig(id="parity", directory=tmp_path / "run"),
        on_event=events.append,
    )
    resumed = plan.run(
        run=RunConfig(id="parity", directory=tmp_path / "run", resume="required"),
    )

    assert len(stage_inputs) == 2
    assert result.final_candidate.values == {
        "planner": "plan-v1",
        "writer": "write-v1",
        "tool.search": "search-v1",
        "router": "route-v0",
    }
    assert result.score == 0.9
    assert result.total_metric_calls == 5
    assert resumed == result
    assert [stage.target_components for stage in result.stages] == [
        ("planner",),
        ("writer", "tool.search"),
    ]
    assert any(event.kind == "checkpoint.written" for event in events)
