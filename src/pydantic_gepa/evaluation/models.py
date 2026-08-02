from __future__ import annotations as _annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..candidates import Candidate
from ..objectives import MetricResult
from ..values import JsonValue
from .evidence import Attachment, Encoder
from .traces import ComponentTrace, ErrorInfo

InputsT = TypeVar("InputsT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")
FailureAction = Literal["raise", "record"]
InvalidScoreAction = Literal["raise", "use_failure_score"]


@dataclass(frozen=True, slots=True)
class Example(Generic[InputsT, OutputT, MetadataT]):
    inputs: InputsT
    expected_output: OutputT | None = None
    name: str | None = None
    id: str | None = None
    metadata: MetadataT | None = None
    attachments: tuple[Attachment, ...] = ()

    @classmethod
    def from_pair(
        cls,
        inputs: InputsT,
        expected_output: OutputT,
        *,
        name: str | None = None,
        id: str | None = None,
        metadata: MetadataT | None = None,
        attachments: tuple[Attachment, ...] = (),
    ) -> Example[InputsT, OutputT, MetadataT]:
        return cls(
            inputs=inputs,
            expected_output=expected_output,
            name=name,
            id=id,
            metadata=metadata,
            attachments=attachments,
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[InputsT, OutputT]],
        *,
        metadata: MetadataT | None = None,
    ) -> list[Example[InputsT, OutputT, MetadataT]]:
        return [cls.from_pair(inputs, expected, metadata=metadata) for inputs, expected in pairs]

    def fingerprint(self, *, encoder: Encoder | None = None) -> str:
        active_encoder = encoder or Encoder()
        value = active_encoder.encode(
            {
                "id": self.id,
                "name": self.name,
                "inputs": self.inputs,
                "expected_output": self.expected_output,
                "metadata": self.metadata,
                "attachments": self.attachments,
            }
        )
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode()).hexdigest()

    def identity(self, *, encoder: Encoder | None = None) -> str:
        return self.id or self.name or self.fingerprint(encoder=encoder)


class CaseResult(BaseModel, Generic[OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    output: OutputT | None = None
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    objectives: dict[str, float] = Field(default_factory=dict)
    feedback: dict[str, str] = Field(default_factory=dict)
    side_info: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    traces: tuple[ComponentTrace, ...] = ()
    artifacts: tuple[Attachment, ...] = ()
    candidate_error: ErrorInfo | None = None
    task_error: ErrorInfo | None = None
    evaluator_error: ErrorInfo | None = None
    infrastructure_error: ErrorInfo | None = None
    duration_seconds: float = Field(ge=0)
    invocation_count: int = Field(ge=0)
    cache_hit: bool = False


class EvaluationConfig(BaseModel, Generic[InputsT, OutputT, MetadataT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    on_invalid_candidate: FailureAction = "raise"
    on_task_error: FailureAction = "raise"
    on_evaluator_error: FailureAction = "raise"
    on_infrastructure_error: FailureAction = "raise"
    invalid_score: InvalidScoreAction = "raise"
    failure_score: float = 0.0
    min_score: float | None = None
    max_score: float | None = None
    cache: Literal["disabled", "memory"] = "disabled"
    cache_failures: bool = False
    cache_nondeterministic: bool = False
    validate_input: Callable[[InputsT], None] | None = None
    validate_candidate: Callable[[Candidate], None] | None = None
    validate_output: Callable[[OutputT], None] | None = None
    validate_result: Callable[[CaseResult[OutputT]], None] | None = None

    @model_validator(mode="after")
    def validate_score_range(self) -> EvaluationConfig[InputsT, OutputT, MetadataT]:
        if not math.isfinite(self.failure_score):
            raise ValueError("failure_score must be finite.")
        if (
            self.min_score is not None
            and self.max_score is not None
            and self.min_score > self.max_score
        ):
            raise ValueError("min_score cannot be greater than max_score.")
        if self.min_score is not None and self.failure_score < self.min_score:
            raise ValueError("failure_score cannot be below min_score.")
        if self.max_score is not None and self.failure_score > self.max_score:
            raise ValueError("failure_score cannot be above max_score.")
        return self


__all__ = (
    "CaseResult",
    "EvaluationConfig",
    "Example",
    "FailureAction",
    "InvalidScoreAction",
)
