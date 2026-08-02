from __future__ import annotations as _annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from .values import SerializableValue

DataInstT = TypeVar("DataInstT")
TrajectoryT = TypeVar("TrajectoryT")
RolloutOutputT = TypeVar("RolloutOutputT")


@dataclass(frozen=True)
class EvaluationBatch(Generic[TrajectoryT, RolloutOutputT]):
    outputs: list[RolloutOutputT]
    scores: list[float]
    trajectories: list[TrajectoryT] | None = None
    objective_scores: list[dict[str, float]] | None = None
    num_metric_calls: int | None = None


class GEPAAdapter(Protocol[DataInstT, TrajectoryT, RolloutOutputT]):
    def evaluate(
        self,
        batch: list[DataInstT],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TrajectoryT, RolloutOutputT]: ...

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TrajectoryT, RolloutOutputT],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, SerializableValue]]]: ...


__all__ = (
    "DataInstT",
    "EvaluationBatch",
    "GEPAAdapter",
    "RolloutOutputT",
    "TrajectoryT",
)
