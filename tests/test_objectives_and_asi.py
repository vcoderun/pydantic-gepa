from __future__ import annotations as _annotations

import json
from dataclasses import dataclass
from typing import cast

from pydantic import BaseModel

from pydantic_gepa import (
    EvaluationBatch,
    MetricResult,
    PydanticEvalsASIBuilder,
    PydanticEvalTrajectory,
    ScoreObjective,
)
from pydantic_gepa.evaluation import ComponentTrace, Encoder
from pydantic_gepa.values import SerializableValue


def test_score_objective_extracts_named_scores_and_direction() -> None:
    metric_result = MetricResult(
        score=0.6,
        feedback="Needs better grounding.",
        side_info={"error_kind": "missing_citation"},
    )
    case = _ReportCase(
        scores={"accuracy": _Score(0.75), "valid": _Score(True), "judge": metric_result}
    )

    assert ScoreObjective(score_key="accuracy").extract(case) == 0.75
    assert ScoreObjective(score_key="valid").extract(case) == 1.0
    assert (
        ScoreObjective(score_key="latency", direction="minimize").extract(
            _ReportCase(scores={"latency": _Score(2.5)})
        )
        == -2.5
    )
    assert ScoreObjective(score_key="missing", failure_score=-1.0).extract(case) == -1.0
    assert (
        ScoreObjective(score_key="text", failure_score=0.2).extract(
            _ReportCase(scores={"text": _Score("good")})
        )
        == 0.2
    )
    assert ScoreObjective(score_key="judge").extract(case) == 0.6
    assert ScoreObjective(score_key="accuracy").extract_objective_scores(case) == {
        "accuracy": 0.75,
        "valid": 1.0,
        "judge": 0.6,
    }
    assert (
        ScoreObjective(score_key="missing").extract_objective_scores(
            _ReportCase(scores={"text": _Score("good")})
        )
        == {}
    )
    assert ScoreObjective(score_key="missing").extract(_ReportCase(scores=None)) == 0.0


def test_pydantic_evals_asi_builder_uses_trajectories_for_each_component() -> None:
    report_case = _ReportCase(
        name="case_easy",
        inputs=_Payload(value="input"),
        expected_output={"answer": "yes"},
        output={"answer": "no"},
        metadata={"difficulty": "easy"},
        scores={"accuracy": _Score(0.25), "raw": 0.5},
        assertions=[_Assertion(name="answer", passed=False, reason="Wrong answer")],
    )
    failure = _Failure(name="case_error", error_stacktrace="boom")
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[report_case.output, None],
        scores=[0.25, 0.0],
        trajectories=[
            PydanticEvalTrajectory(report_case=report_case),
            PydanticEvalTrajectory(report_case=failure),
        ],
    )

    reflective = PydanticEvalsASIBuilder(max_examples=1).build(
        candidate={"instructions": "current"},
        eval_batch=eval_batch,
        components_to_update=["instructions", "router"],
    )

    assert list(reflective) == ["instructions", "router"]
    assert reflective["instructions"] == reflective["router"]
    assert reflective["instructions"][0] == {
        "case_name": "case_easy",
        "inputs": {"value": "input"},
        "expected_output": {"answer": "yes"},
        "score": 0.25,
        "success": False,
        "failure_category": "assertion_failure",
        "metadata": {"difficulty": "easy"},
        "actual_output": {"answer": "no"},
        "scores": {"accuracy": 0.25, "raw": 0.5},
        "assertions": [{"name": "answer", "passed": False, "reason": "Wrong answer"}],
    }


def test_pydantic_evals_asi_builder_handles_failures_and_missing_traces() -> None:
    failure = _Failure(name="case_error", error_stacktrace="boom")
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None],
        scores=[0.0],
        trajectories=[PydanticEvalTrajectory(report_case=failure)],
    )

    reflective = PydanticEvalsASIBuilder().build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["instructions"],
    )
    empty = PydanticEvalsASIBuilder().build(
        candidate={},
        eval_batch=EvaluationBatch(outputs=[], scores=[]),
        components_to_update=["instructions"],
    )

    assert reflective["instructions"][0]["error"] == "boom"
    assert reflective["instructions"][0]["success"] is False
    assert reflective["instructions"][0]["failure_category"] == "error"
    assert reflective["instructions"][0]["inputs"] is None
    assert empty == {}


def test_pydantic_evals_asi_builder_supports_mapping_assertions() -> None:
    report_case = _ReportCase(
        name="case_assertions",
        assertions={"matches_expected": _AssertionResult(value=True, reason="exact match")},
    )
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None],
        scores=[1.0],
        trajectories=[PydanticEvalTrajectory(report_case=report_case)],
    )

    reflective = PydanticEvalsASIBuilder().build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["instructions"],
    )

    assert reflective["instructions"][0]["assertions"] == [
        {"name": "matches_expected", "passed": True, "reason": "exact match"}
    ]


def test_pydantic_evals_asi_builder_serializes_error_and_exception_fields() -> None:
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None, None],
        scores=[0.0, 0.0],
        trajectories=[
            PydanticEvalTrajectory(report_case=_ErrorFailure(name="case_error", error="bad")),
            PydanticEvalTrajectory(
                report_case=_ExceptionFailure(name="case_exception", exception=ValueError("worse"))
            ),
        ],
    )

    reflective = PydanticEvalsASIBuilder().build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["instructions"],
    )

    assert reflective["instructions"][0]["error"] == "bad"
    assert reflective["instructions"][1]["error"] == "worse"


def test_pydantic_evals_asi_builder_serializes_sequences_and_unknown_objects() -> None:
    report_case = _ReportCase(
        name="case_objects",
        inputs=("tuple", _Opaque()),
        expected_output=[_Opaque()],
        output=_Opaque(),
        scores={},
    )
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[report_case.output],
        scores=[0.5],
        trajectories=[PydanticEvalTrajectory(report_case=report_case)],
    )

    encoder = Encoder()
    encoder.register(_Opaque, repr)
    reflective = PydanticEvalsASIBuilder(encoder=encoder).build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["instructions"],
    )

    record = reflective["instructions"][0]
    assert record["inputs"] == ["tuple", "<opaque>"]
    assert record["expected_output"] == ["<opaque>"]
    assert record["actual_output"] == "<opaque>"


def test_pydantic_evals_asi_builder_routes_traces_and_evaluator_evidence() -> None:
    report_case = _ReportCase(name="trace-case", metadata={"component": "fallback"})
    trajectory = PydanticEvalTrajectory(
        report_case=report_case,
        traces=(
            ComponentTrace(
                id="trace-one",
                component="tool",
                kind="tool_call",
                input={"query": "Ada"},
                output={"name": "Ada"},
                metadata={"attempt": 1},
                duration_seconds=0.2,
            ),
        ),
        evidence={"judge": {"reason": "correct tool"}},
    )
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None],
        scores=[0.8],
        trajectories=[trajectory],
    )

    reflective = PydanticEvalsASIBuilder(
        component_hint_metadata_key="component",
        evaluator_evidence={
            "accuracy": lambda **_: {
                "feedback": "Keep the tool choice.",
                "details": {"field": "name"},
            }
        },
    ).build(
        candidate={"tool": "candidate", "fallback": "candidate"},
        eval_batch=eval_batch,
        components_to_update=["tool", "fallback"],
    )

    assert reflective["fallback"] == []
    record = reflective["tool"][0]
    assert record["traces"] == [
        {
            "id": "trace-one",
            "component": "tool",
            "kind": "tool_call",
            "parent_id": None,
            "input": {"query": "Ada"},
            "output": {"name": "Ada"},
            "metadata": {"attempt": 1},
            "duration_seconds": 0.2,
            "error": None,
        }
    ]
    assert record["evidence"] == {"judge": {"reason": "correct tool"}}
    assert record["evaluator_evidence"] == {
        "accuracy": {
            "details": {"field": "name"},
            "feedback": "Keep the tool choice.",
        }
    }


def test_pydantic_evals_asi_builder_selector_precedes_trace_routing() -> None:
    trajectory = PydanticEvalTrajectory(
        report_case=_ReportCase(name="selector-first"),
        traces=(ComponentTrace(id="trace", component="tool"),),
    )

    reflective = PydanticEvalsASIBuilder(
        component_selector=lambda **_: ["prompt"],
    ).build(
        candidate={"prompt": "candidate", "tool": "candidate"},
        eval_batch=EvaluationBatch(
            outputs=[None],
            scores=[0.5],
            trajectories=[trajectory],
        ),
        components_to_update=["prompt", "tool"],
    )

    assert len(reflective["prompt"]) == 1
    assert reflective["tool"] == []


def test_pydantic_evals_asi_builder_can_skip_unroutable_evidence() -> None:
    reflective = PydanticEvalsASIBuilder(unroutable_evidence="skip").build(
        candidate={"prompt": "candidate"},
        eval_batch=EvaluationBatch(
            outputs=[None],
            scores=[0.5],
            trajectories=[PydanticEvalTrajectory(report_case=_ReportCase(name="unroutable"))],
        ),
        components_to_update=["prompt"],
    )

    assert reflective == {"prompt": []}


def test_pydantic_evals_asi_builder_selects_failures_and_bounds_encoded_size() -> None:
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None, None, None],
        scores=[0.9, 0.0, 0.2],
        trajectories=[
            PydanticEvalTrajectory(report_case=_ReportCase(name="success")),
            PydanticEvalTrajectory(report_case=_Failure(name="error", error_stacktrace="boom")),
            PydanticEvalTrajectory(report_case=_ReportCase(name="low-score")),
        ],
    )
    failures = PydanticEvalsASIBuilder(
        max_examples=2,
        sample_selection="failure_first",
    ).build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["prompt"],
    )
    one_record_size = len(
        json.dumps(
            failures["prompt"][0],
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    bounded = PydanticEvalsASIBuilder(
        sample_selection="failure_first",
        max_encoded_chars=one_record_size,
    ).build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["prompt"],
    )

    assert [record["case_name"] for record in failures["prompt"]] == ["error", "low-score"]
    assert [record["case_name"] for record in bounded["prompt"]] == ["error"]


def test_pydantic_evals_asi_builder_serializes_dataclass_inputs() -> None:
    report_case = _ReportCase(inputs=_DataclassPayload(value="structured"))
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None],
        scores=[0.0],
        trajectories=[PydanticEvalTrajectory(report_case=report_case)],
    )

    reflective = PydanticEvalsASIBuilder().build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["instructions"],
    )

    assert reflective["instructions"][0]["inputs"] == {"value": "structured"}


def test_pydantic_evals_asi_builder_includes_metric_feedback_side_info_and_knobs() -> None:
    report_case = _ReportCase(
        name="case_feedback",
        expected_output={"answer": "yes"},
        metadata={"difficulty": "hard"},
        scores={
            "judge": MetricResult(
                score=0.4,
                feedback="Answer missed the constraint.",
                side_info={"constraint": "length"},
            )
        },
        assertions=[_Assertion(name="constraint", passed=False, reason="too long")],
    )
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None],
        scores=[0.4],
        trajectories=[PydanticEvalTrajectory(report_case=report_case)],
    )

    reflective = PydanticEvalsASIBuilder().build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["instructions"],
    )
    compact = PydanticEvalsASIBuilder(
        include_case_metadata=False,
        include_expected_output=False,
        include_scores=False,
        include_assertions=False,
        include_errors=False,
    ).build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["instructions"],
    )

    record = reflective["instructions"][0]
    assert record["scores"] == {"judge": 0.4}
    assert record["metric_feedback"] == {"judge": "Answer missed the constraint."}
    assert record["metric_side_info"] == {"judge": {"constraint": "length"}}
    assert compact["instructions"][0] == {
        "case_name": "case_feedback",
        "inputs": None,
        "actual_output": None,
        "score": 0.4,
        "success": False,
        "failure_category": "assertion_failure",
    }


def test_pydantic_evals_asi_builder_routes_records_by_metadata_hint() -> None:
    first = _ReportCase(name="router_case", metadata={"components": ["router"]})
    second = _ReportCase(name="instruction_case")
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None, None],
        scores=[0.0, 0.5],
        trajectories=[
            PydanticEvalTrajectory(report_case=first),
            PydanticEvalTrajectory(report_case=second),
        ],
    )

    reflective = PydanticEvalsASIBuilder(
        component_hint_metadata_key="components",
    ).build(
        candidate={"instructions": "seed", "router": "seed"},
        eval_batch=eval_batch,
        components_to_update=["instructions", "router"],
    )

    assert [record["case_name"] for record in reflective["router"]] == [
        "router_case",
        "instruction_case",
    ]
    assert [record["case_name"] for record in reflective["instructions"]] == [
        "instruction_case",
    ]


def test_pydantic_evals_asi_builder_can_select_lowest_score_examples() -> None:
    eval_batch: EvaluationBatch[PydanticEvalTrajectory, SerializableValue | None] = EvaluationBatch(
        outputs=[None, None, None],
        scores=[0.9, 0.1, 0.4],
        trajectories=[
            PydanticEvalTrajectory(report_case=_ReportCase(name="high")),
            PydanticEvalTrajectory(report_case=_ReportCase(name="low")),
            PydanticEvalTrajectory(report_case=_ReportCase(name="mid")),
        ],
    )

    reflective = PydanticEvalsASIBuilder(
        max_examples=2,
        sample_selection="lowest_score",
    ).build(
        candidate={},
        eval_batch=eval_batch,
        components_to_update=["instructions"],
    )

    assert [record["case_name"] for record in reflective["instructions"]] == [
        "low",
        "mid",
    ]


def test_pydantic_evals_asi_builder_handles_empty_components_custom_selector_and_string_hints() -> (
    None
):
    selector_report_case = _ReportCase(name="selector_case")
    metadata_report_case = _ReportCase(name="metadata_case", metadata={"component": "router"})
    weird_score_case = _ReportCase(name="weird_score")
    invalid_hint_case = _ReportCase(name="invalid_hint", metadata={"component": [1]})
    non_sequence_hint_case = _ReportCase(name="non_sequence_hint", metadata={"component": 123})

    empty = PydanticEvalsASIBuilder().build(
        candidate={},
        eval_batch=EvaluationBatch(outputs=[], scores=[]),
        components_to_update=[],
    )
    selector_reflective = PydanticEvalsASIBuilder(
        component_selector=lambda **_: ["router", "missing"],
    ).build(
        candidate={},
        eval_batch=EvaluationBatch(
            outputs=[None],
            scores=[1.0],
            trajectories=[PydanticEvalTrajectory(report_case=selector_report_case)],
        ),
        components_to_update=["instructions", "router"],
    )
    hinted_reflective = PydanticEvalsASIBuilder(
        component_selector=lambda **_: None,
        component_hint_metadata_key="component",
    ).build(
        candidate={},
        eval_batch=EvaluationBatch(
            outputs=[None, None],
            scores=[0.0, cast("float", "bad")],
            trajectories=[
                PydanticEvalTrajectory(report_case=metadata_report_case),
                PydanticEvalTrajectory(report_case=weird_score_case),
            ],
        ),
        components_to_update=["instructions", "router"],
    )
    invalid_hint_reflective = PydanticEvalsASIBuilder(
        component_hint_metadata_key="component",
        sample_selection="lowest_score",
        max_examples=1,
    ).build(
        candidate={},
        eval_batch=EvaluationBatch(
            outputs=[None],
            scores=[cast("float", "bad")],
            trajectories=[PydanticEvalTrajectory(report_case=invalid_hint_case)],
        ),
        components_to_update=["instructions", "router"],
    )
    non_sequence_hint_reflective = PydanticEvalsASIBuilder(
        component_hint_metadata_key="component",
    ).build(
        candidate={},
        eval_batch=EvaluationBatch(
            outputs=[None],
            scores=[0.2],
            trajectories=[PydanticEvalTrajectory(report_case=non_sequence_hint_case)],
        ),
        components_to_update=["instructions", "router"],
    )

    assert empty == {}
    assert [record["case_name"] for record in selector_reflective["router"]] == ["selector_case"]
    assert selector_reflective["instructions"] == []
    assert [record["case_name"] for record in hinted_reflective["router"]] == [
        "metadata_case",
        "weird_score",
    ]
    assert [record["case_name"] for record in hinted_reflective["instructions"]] == ["weird_score"]
    assert [record["case_name"] for record in invalid_hint_reflective["instructions"]] == [
        "invalid_hint"
    ]
    assert [record["case_name"] for record in invalid_hint_reflective["router"]] == ["invalid_hint"]
    assert [record["case_name"] for record in non_sequence_hint_reflective["instructions"]] == [
        "non_sequence_hint"
    ]
    assert [record["case_name"] for record in non_sequence_hint_reflective["router"]] == [
        "non_sequence_hint"
    ]


@dataclass(frozen=True)
class _Score:
    value: bool | int | float | str | None


ScoreInput = bool | int | float | str | MetricResult | None | _Score


@dataclass(frozen=True)
class _Assertion:
    name: str
    passed: bool
    reason: str | None


@dataclass(frozen=True)
class _AssertionResult:
    value: bool
    reason: str | None


@dataclass(frozen=True)
class _ReportCase:
    name: str = "case"
    inputs: SerializableValue = None
    expected_output: SerializableValue = None
    output: SerializableValue = None
    metadata: dict[str, SerializableValue] | None = None
    scores: dict[str, ScoreInput] | None = None
    assertions: dict[str, _AssertionResult] | list[_Assertion] | None = None


@dataclass(frozen=True)
class _Failure:
    name: str
    error_stacktrace: str


@dataclass(frozen=True)
class _ErrorFailure:
    name: str
    error: str


@dataclass(frozen=True)
class _ExceptionFailure:
    name: str
    exception: Exception


class _Opaque:
    def __repr__(self) -> str:
        return "<opaque>"


class _Payload(BaseModel):
    value: str


@dataclass(frozen=True)
class _DataclassPayload:
    value: str
