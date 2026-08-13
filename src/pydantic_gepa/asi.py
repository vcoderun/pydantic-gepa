from __future__ import annotations as _annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias, TypeVar, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .compat import EvaluationBatch
from .evaluation.evidence import Encoder
from .evaluation.models import CaseResult
from .evaluation.traces import ComponentTrace, ErrorInfo
from .objectives import MetricResult, ScoreInput, ScoreValueCarrier
from .values import JsonValue, SerializableValue

RolloutOutputT = TypeVar("RolloutOutputT")


@runtime_checkable
class ReportCaseName(Protocol):
    @property
    def name(self) -> str: ...


@runtime_checkable
class ReportCaseInputs(Protocol):
    @property
    def inputs(self) -> SerializableValue: ...


@runtime_checkable
class ReportCaseExpectedOutput(Protocol):
    @property
    def expected_output(self) -> SerializableValue: ...


@runtime_checkable
class ReportCaseOutput(Protocol):
    @property
    def output(self) -> SerializableValue: ...


@runtime_checkable
class ReportCaseMetadata(Protocol):
    @property
    def metadata(self) -> Mapping[str, SerializableValue] | None: ...


@runtime_checkable
class ReportCaseScores(Protocol):
    @property
    def scores(self) -> Mapping[str, ScoreInput | ScoreValueCarrier] | None: ...


@runtime_checkable
class ReportCaseAssertions(Protocol):
    @property
    def assertions(
        self,
    ) -> Mapping[str, AssertionResult] | Sequence[AssertionRecord] | None: ...


@runtime_checkable
class AssertionRecord(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def passed(self) -> bool: ...

    @property
    def reason(self) -> str | None: ...


@runtime_checkable
class AssertionResult(Protocol):
    @property
    def value(self) -> bool | int | float | str | None: ...

    @property
    def reason(self) -> str | None: ...


@runtime_checkable
class ReportCaseErrorStacktrace(Protocol):
    @property
    def error_stacktrace(self) -> str | BaseException | None: ...


@runtime_checkable
class ReportCaseError(Protocol):
    @property
    def error(self) -> str | BaseException | None: ...


@runtime_checkable
class ReportCaseException(Protocol):
    @property
    def exception(self) -> str | BaseException | None: ...


@runtime_checkable
class ReportCaseDuration(Protocol):
    @property
    def duration(self) -> float: ...


@runtime_checkable
class ScoreReason(Protocol):
    @property
    def reason(self) -> str | None: ...


ReportCaseRecord: TypeAlias = (
    ReportCaseName
    | ReportCaseInputs
    | ReportCaseExpectedOutput
    | ReportCaseOutput
    | ReportCaseMetadata
    | ReportCaseScores
    | ReportCaseAssertions
    | ReportCaseErrorStacktrace
    | ReportCaseError
    | ReportCaseException
    | ReportCaseDuration
)
SampleSelection = Literal["input_order", "lowest_score", "failure_first"]
UnroutableEvidence = Literal["shared", "skip"]


class PydanticEvalTrajectory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    report_case: ReportCaseRecord
    traces: tuple[ComponentTrace, ...] = ()
    evidence: dict[str, JsonValue] = Field(default_factory=dict)


@runtime_checkable
class ASIBuilder(Protocol):
    def build(
        self,
        *,
        candidate: Mapping[str, str],
        eval_batch: EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, SerializableValue]]]: ...


@runtime_checkable
class ComponentRecordSelector(Protocol):
    def __call__(
        self,
        *,
        candidate: Mapping[str, str],
        report_case: ReportCaseRecord,
        record: Mapping[str, SerializableValue],
        components_to_update: Sequence[str],
    ) -> Sequence[str] | None: ...


@runtime_checkable
class EvaluatorEvidence(Protocol):
    def __call__(
        self,
        *,
        candidate: Mapping[str, str],
        report_case: ReportCaseRecord,
        score: float,
    ) -> Mapping[str, SerializableValue] | None: ...


class PydanticEvalsASIBuilder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_examples: int | None = None
    include_case_metadata: bool = True
    include_expected_output: bool = True
    include_scores: bool = True
    include_assertions: bool = True
    include_errors: bool = True
    sample_selection: SampleSelection = "input_order"
    max_encoded_chars: int | None = Field(default=None, ge=1)
    component_hint_metadata_key: str | None = None
    component_selector: ComponentRecordSelector | None = None
    evaluator_evidence: dict[str, EvaluatorEvidence] = Field(default_factory=dict)
    unroutable_evidence: UnroutableEvidence = "shared"
    encoder: Encoder = Field(default_factory=Encoder)

    def build(
        self,
        *,
        candidate: Mapping[str, str],
        eval_batch: EvaluationBatch[PydanticEvalTrajectory, RolloutOutputT],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, SerializableValue]]]:
        if not components_to_update:
            return {}
        if eval_batch.trajectories is None:
            return {}

        routed_records: dict[str, list[dict[str, SerializableValue]]] = {
            component: [] for component in components_to_update
        }
        for trajectory, score in zip(
            eval_batch.trajectories,
            eval_batch.scores,
            strict=True,
        ):
            report_case = trajectory.report_case
            record = report_case_record(
                report_case,
                score=score,
                include_case_metadata=self.include_case_metadata,
                include_expected_output=self.include_expected_output,
                include_scores=self.include_scores,
                include_assertions=self.include_assertions,
                include_errors=self.include_errors,
                encoder=self.encoder,
            )
            if trajectory.traces:
                record["traces"] = [trace.model_dump(mode="json") for trace in trajectory.traces]
            if trajectory.evidence:
                record["evidence"] = trajectory.evidence
            adapted_evidence = {
                name: self.encoder.mapping(payload)
                for name, adapter in self.evaluator_evidence.items()
                if (
                    payload := adapter(
                        candidate=candidate,
                        report_case=report_case,
                        score=score,
                    )
                )
                is not None
            }
            if adapted_evidence:
                record["evaluator_evidence"] = adapted_evidence
            for component in self._targets_for_record(
                candidate=candidate,
                trajectory=trajectory,
                report_case=report_case,
                record=record,
                components_to_update=components_to_update,
            ):
                routed_records[component].append(record)

        return {
            component: _select_records(
                records,
                max_examples=self.max_examples,
                sample_selection=self.sample_selection,
                max_encoded_chars=self.max_encoded_chars,
            )
            for component, records in routed_records.items()
        }

    def _targets_for_record(
        self,
        *,
        candidate: Mapping[str, str],
        trajectory: PydanticEvalTrajectory,
        report_case: ReportCaseRecord,
        record: Mapping[str, SerializableValue],
        components_to_update: Sequence[str],
    ) -> list[str]:
        if self.component_selector is not None:
            selected = self.component_selector(
                candidate=candidate,
                report_case=report_case,
                record=record,
                components_to_update=components_to_update,
            )
            if selected is not None:
                return _filtered_components(selected, components_to_update)

        traced_components = _filtered_components(
            [trace.component for trace in trajectory.traces],
            components_to_update,
        )
        if traced_components:
            return list(dict.fromkeys(traced_components))

        if self.component_hint_metadata_key is not None:
            hinted_components = _component_hints_from_metadata(
                report_case,
                metadata_key=self.component_hint_metadata_key,
            )
            if hinted_components:
                return _filtered_components(hinted_components, components_to_update)

        if self.unroutable_evidence == "shared":
            return list(components_to_update)
        return []


def report_case_record(
    report_case: ReportCaseRecord,
    *,
    score: float,
    include_case_metadata: bool,
    include_expected_output: bool,
    include_scores: bool,
    include_assertions: bool,
    include_errors: bool,
    encoder: Encoder | None = None,
) -> dict[str, SerializableValue]:
    active_encoder = encoder or Encoder()
    record: dict[str, SerializableValue] = {
        "case_name": report_case.name if isinstance(report_case, ReportCaseName) else "unknown",
        "inputs": _inputs_payload(report_case, encoder=active_encoder),
        "score": score,
        "success": _success_payload(report_case, score=score),
        "failure_category": _failure_category(report_case, score=score),
    }

    if include_expected_output:
        record["expected_output"] = _expected_output_payload(report_case, encoder=active_encoder)
    if include_case_metadata:
        record["metadata"] = _metadata_payload(report_case, encoder=active_encoder)
    if isinstance(report_case, ReportCaseOutput):
        record["actual_output"] = active_encoder.encode(report_case.output)
    if include_scores:
        scores = _scores_payload(report_case, encoder=active_encoder)
        if scores:
            record["scores"] = scores
        metric_feedback = _metric_feedback_payload(report_case)
        if metric_feedback:
            record["metric_feedback"] = metric_feedback
        metric_side_info = _metric_side_info_payload(report_case, encoder=active_encoder)
        if metric_side_info:
            record["metric_side_info"] = metric_side_info
    if include_assertions:
        assertions = _assertions_payload(report_case)
        if assertions:
            record["assertions"] = assertions

    if include_errors:
        error = _error_payload(report_case)
        if error is not None:
            record["error"] = error

    return record


def report_case_result(
    report_case: ReportCaseRecord,
    *,
    transformed_score: float,
    objective_key: str,
    objective_scores: Mapping[str, float] | None = None,
    traces: Sequence[ComponentTrace] = (),
    encoder: Encoder | None = None,
) -> CaseResult[JsonValue]:
    active_encoder = encoder or Encoder()
    metrics: dict[str, MetricResult] = {}
    raw_scores = report_case.scores if isinstance(report_case, ReportCaseScores) else None
    for name, score in (raw_scores or {}).items():
        key = str(name)
        value = score.value if isinstance(score, ScoreValueCarrier) else score
        if isinstance(value, MetricResult):
            metrics[key] = value
        elif isinstance(value, bool | int | float):
            metrics[key] = MetricResult(
                score=float(value),
                role="objective" if key == objective_key else "diagnostic",
                feedback=score.reason if isinstance(score, ScoreReason) else None,
            )

    objectives = {name: float(value) for name, value in (objective_scores or {}).items()}
    if objective_key not in objectives and objective_key in metrics:
        objectives[objective_key] = metrics[objective_key].score
    if objective_key not in metrics:
        metrics[objective_key] = MetricResult(
            score=objectives.get(objective_key, transformed_score),
            role="objective",
        )
    elif metrics[objective_key].role != "objective":
        metrics[objective_key] = metrics[objective_key].model_copy(update={"role": "objective"})

    error = _error_payload(report_case)
    duration = report_case.duration if isinstance(report_case, ReportCaseDuration) else 0.0
    return CaseResult[JsonValue](
        output=(
            active_encoder.encode(report_case.output)
            if isinstance(report_case, ReportCaseOutput)
            else None
        ),
        metrics=metrics,
        objectives=objectives,
        feedback={
            name: metric.feedback for name, metric in metrics.items() if metric.feedback is not None
        },
        side_info={
            name: active_encoder.mapping(metric.side_info)
            for name, metric in metrics.items()
            if metric.side_info
        },
        traces=tuple(traces),
        task_error=None if error is None else ErrorInfo(kind="EvaluationError", message=error),
        duration_seconds=max(0.0, duration),
        invocation_count=1,
    )


def _select_records(
    records: list[dict[str, SerializableValue]],
    *,
    max_examples: int | None,
    sample_selection: SampleSelection,
    max_encoded_chars: int | None,
) -> list[dict[str, SerializableValue]]:
    if sample_selection == "lowest_score":
        ordered = sorted(records, key=_record_score_key)
    elif sample_selection == "failure_first":
        ordered = sorted(records, key=_record_failure_key)
    else:
        ordered = records
    selected = ordered if max_examples is None else ordered[:max_examples]
    if max_encoded_chars is None:
        return [dict(record) for record in selected]

    bounded: list[dict[str, SerializableValue]] = []
    encoded_chars = 0
    for record in selected:
        record_size = len(json.dumps(record, sort_keys=True, separators=(",", ":")))
        if encoded_chars + record_size > max_encoded_chars:
            continue
        bounded.append(dict(record))
        encoded_chars += record_size
    return bounded


def _record_score_key(record: Mapping[str, SerializableValue]) -> float:
    score = record.get("score")
    if isinstance(score, int | float):
        return float(score)
    return 0.0


def _record_failure_key(record: Mapping[str, SerializableValue]) -> tuple[bool, float]:
    return record.get("failure_category") is None, _record_score_key(record)


def _filtered_components(
    selected: Sequence[str],
    components_to_update: Sequence[str],
) -> list[str]:
    allowed = set(components_to_update)
    return [component for component in selected if component in allowed]


def _component_hints_from_metadata(
    report_case: ReportCaseRecord,
    *,
    metadata_key: str,
) -> list[str]:
    if not isinstance(report_case, ReportCaseMetadata) or report_case.metadata is None:
        return []
    raw_value = report_case.metadata.get(metadata_key)
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, str | bytes | bytearray):
        return [str(item) for item in raw_value if isinstance(item, str)]
    return []


def _inputs_payload(report_case: ReportCaseRecord, *, encoder: Encoder) -> JsonValue:
    if isinstance(report_case, ReportCaseInputs):
        return encoder.encode(report_case.inputs)
    return None


def _expected_output_payload(report_case: ReportCaseRecord, *, encoder: Encoder) -> JsonValue:
    if isinstance(report_case, ReportCaseExpectedOutput):
        return encoder.encode(report_case.expected_output)
    return None


def _metadata_payload(report_case: ReportCaseRecord, *, encoder: Encoder) -> dict[str, JsonValue]:
    if not isinstance(report_case, ReportCaseMetadata) or report_case.metadata is None:
        return {}
    return encoder.mapping(report_case.metadata)


def _scores_payload(report_case: ReportCaseRecord, *, encoder: Encoder) -> dict[str, JsonValue]:
    raw_scores = report_case.scores if isinstance(report_case, ReportCaseScores) else None
    if raw_scores is None:
        return {}
    payload: dict[str, JsonValue] = {}
    for name, score in raw_scores.items():
        value = score.value if isinstance(score, ScoreValueCarrier) else score
        if isinstance(value, MetricResult):
            payload[str(name)] = value.score
        else:
            payload[str(name)] = encoder.encode(value)
    return payload


def _metric_feedback_payload(report_case: ReportCaseRecord) -> dict[str, str]:
    raw_scores = report_case.scores if isinstance(report_case, ReportCaseScores) else None
    if raw_scores is None:
        return {}
    payload: dict[str, str] = {}
    for name, score in raw_scores.items():
        value = score.value if isinstance(score, ScoreValueCarrier) else score
        if isinstance(value, MetricResult) and value.feedback is not None:
            payload[str(name)] = value.feedback
    return payload


def _metric_side_info_payload(
    report_case: ReportCaseRecord,
    *,
    encoder: Encoder,
) -> dict[str, JsonValue]:
    raw_scores = report_case.scores if isinstance(report_case, ReportCaseScores) else None
    if raw_scores is None:
        return {}
    payload: dict[str, JsonValue] = {}
    for name, score in raw_scores.items():
        value = score.value if isinstance(score, ScoreValueCarrier) else score
        if isinstance(value, MetricResult) and value.side_info:
            payload[str(name)] = encoder.encode(value.side_info)
    return payload


def _assertions_payload(report_case: ReportCaseRecord) -> list[dict[str, SerializableValue]]:
    raw_assertions = (
        report_case.assertions if isinstance(report_case, ReportCaseAssertions) else None
    )
    if raw_assertions is None:
        return []
    if isinstance(raw_assertions, Mapping):
        payload: list[dict[str, SerializableValue]] = []
        for name, assertion in raw_assertions.items():
            passed = assertion.value if isinstance(assertion, AssertionResult) else assertion
            payload.append(
                {
                    "name": str(name),
                    "passed": bool(passed),
                    "reason": assertion.reason if isinstance(assertion, AssertionResult) else None,
                }
            )
        return payload
    return [
        {
            "name": assertion.name,
            "passed": assertion.passed,
            "reason": assertion.reason,
        }
        for assertion in raw_assertions
    ]


def _success_payload(report_case: ReportCaseRecord, *, score: float) -> bool:
    if _error_payload(report_case) is not None:
        return False
    assertions = _assertions_payload(report_case)
    if assertions and any(assertion["passed"] is False for assertion in assertions):
        return False
    numeric_score = _numeric_score(score)
    return numeric_score > 0 if numeric_score is not None else False


def _failure_category(report_case: ReportCaseRecord, *, score: float) -> str | None:
    if _error_payload(report_case) is not None:
        return "error"
    assertions = _assertions_payload(report_case)
    if assertions and any(assertion["passed"] is False for assertion in assertions):
        return "assertion_failure"
    numeric_score = _numeric_score(score)
    if numeric_score is None or numeric_score <= 0:
        return "low_score"
    return None


def _numeric_score(score: Any) -> float | None:
    if isinstance(score, int | float):
        return float(score)
    return None


def _error_payload(report_case: ReportCaseRecord) -> str | None:
    if isinstance(report_case, ReportCaseErrorStacktrace) and report_case.error_stacktrace:
        return str(report_case.error_stacktrace)
    if isinstance(report_case, ReportCaseError) and report_case.error:
        return str(report_case.error)
    if isinstance(report_case, ReportCaseException) and report_case.exception:
        return str(report_case.exception)
    return None


__all__ = (
    "ASIBuilder",
    "ComponentRecordSelector",
    "EvaluatorEvidence",
    "PydanticEvalTrajectory",
    "PydanticEvalsASIBuilder",
    "SampleSelection",
    "UnroutableEvidence",
    "report_case_record",
    "report_case_result",
)
