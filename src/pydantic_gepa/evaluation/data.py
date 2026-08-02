from __future__ import annotations as _annotations

import json
import random
from collections.abc import Sequence
from hashlib import sha256
from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from ..candidates import Candidate
from ..harness import run_awaitable_sync
from .models import CaseResult, Example
from .runner import Evaluation

InputsT = TypeVar("InputsT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")

Sampling = Literal["input_order", "random"]
FinalRescore = Literal["none", "validation", "train_validation", "all"]


class DataSplit(BaseModel, Generic[InputsT, OutputT, MetadataT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    train: tuple[Example[InputsT, OutputT, MetadataT], ...]
    validation: tuple[Example[InputsT, OutputT, MetadataT], ...]
    test: tuple[Example[InputsT, OutputT, MetadataT], ...] = ()
    seed: int = 0
    allow_train_validation_overlap: bool = False

    @model_validator(mode="after")
    def validate_sets(self) -> DataSplit[InputsT, OutputT, MetadataT]:
        if not self.train:
            raise ValueError("train cannot be empty.")
        if not self.validation:
            raise ValueError("validation cannot be empty.")
        if not self.allow_train_validation_overlap:
            train_ids = {example.identity() for example in self.train}
            validation_ids = {example.identity() for example in self.validation}
            overlap = sorted(train_ids & validation_ids)
            if overlap:
                raise ValueError(
                    "train and validation overlap; explicitly allow overlap to reuse examples."
                )
        return self

    @classmethod
    def from_sets(
        cls,
        *,
        train: Sequence[Example[InputsT, OutputT, MetadataT]],
        validation: Sequence[Example[InputsT, OutputT, MetadataT]],
        test: Sequence[Example[InputsT, OutputT, MetadataT]] = (),
        seed: int = 0,
        sampling: Sampling = "input_order",
        max_train: int | None = None,
        max_validation: int | None = None,
        max_test: int | None = None,
        allow_train_validation_overlap: bool = False,
    ) -> DataSplit[InputsT, OutputT, MetadataT]:
        generator = random.Random(seed)
        return cls(
            train=_sample(train, maximum=max_train, sampling=sampling, generator=generator),
            validation=_sample(
                validation,
                maximum=max_validation,
                sampling=sampling,
                generator=generator,
            ),
            test=_sample(test, maximum=max_test, sampling=sampling, generator=generator),
            seed=seed,
            allow_train_validation_overlap=allow_train_validation_overlap,
        )

    @classmethod
    def partition(
        cls,
        examples: Sequence[Example[InputsT, OutputT, MetadataT]],
        *,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.0,
        seed: int = 0,
        sampling: Sampling = "random",
        max_train: int | None = None,
        max_validation: int | None = None,
        max_test: int | None = None,
    ) -> DataSplit[InputsT, OutputT, MetadataT]:
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1.")
        if not 0 <= test_fraction < 1:
            raise ValueError("test_fraction must be between 0 and 1.")
        if validation_fraction + test_fraction >= 1:
            raise ValueError("validation_fraction and test_fraction must sum to less than 1.")

        ordered = list(examples)
        if sampling == "random":
            random.Random(seed).shuffle(ordered)
        validation_size = max(1, int(len(ordered) * validation_fraction))
        test_size = int(len(ordered) * test_fraction)
        train_size = len(ordered) - validation_size - test_size
        if train_size < 1:
            raise ValueError("The requested split leaves no training examples.")

        train = ordered[:train_size]
        validation = ordered[train_size : train_size + validation_size]
        test = ordered[train_size + validation_size :]
        return cls.from_sets(
            train=train,
            validation=validation,
            test=test,
            seed=seed,
            sampling="input_order",
            max_train=max_train,
            max_validation=max_validation,
            max_test=max_test,
        )

    def fingerprints(self) -> dict[str, str]:
        return {
            "train": _examples_fingerprint(self.train),
            "validation": _examples_fingerprint(self.validation),
            "test": _examples_fingerprint(self.test),
        }


class RescoreResult(BaseModel, Generic[OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    candidate_fingerprint: str
    split_fingerprints: dict[str, str]
    train: tuple[CaseResult[OutputT], ...] = ()
    validation: tuple[CaseResult[OutputT], ...] = ()
    test: tuple[CaseResult[OutputT], ...] = ()


async def arun_rescore(
    evaluation: Evaluation[InputsT, OutputT, MetadataT],
    candidate: Candidate,
    data: DataSplit[InputsT, OutputT, MetadataT],
    *,
    policy: FinalRescore = "validation",
) -> RescoreResult[OutputT]:
    train = (
        await _evaluate_set(evaluation, candidate, data.train, stage_id="final.train")
        if policy in {"train_validation", "all"}
        else ()
    )
    validation = (
        await _evaluate_set(evaluation, candidate, data.validation, stage_id="final.validation")
        if policy in {"validation", "train_validation", "all"}
        else ()
    )
    test = (
        await _evaluate_set(evaluation, candidate, data.test, stage_id="final.test")
        if policy == "all"
        else ()
    )
    return RescoreResult(
        candidate_fingerprint=candidate.fingerprint(),
        split_fingerprints=data.fingerprints(),
        train=train,
        validation=validation,
        test=test,
    )


def rescore(
    evaluation: Evaluation[InputsT, OutputT, MetadataT],
    candidate: Candidate,
    data: DataSplit[InputsT, OutputT, MetadataT],
    *,
    policy: FinalRescore = "validation",
) -> RescoreResult[OutputT]:
    return run_awaitable_sync(arun_rescore(evaluation, candidate, data, policy=policy))


def _sample(
    examples: Sequence[Example[InputsT, OutputT, MetadataT]],
    *,
    maximum: int | None,
    sampling: Sampling,
    generator: random.Random,
) -> tuple[Example[InputsT, OutputT, MetadataT], ...]:
    if maximum is not None and maximum < 1:
        raise ValueError("Maximum set sizes must be at least 1.")
    selected = list(examples)
    if sampling == "random":
        generator.shuffle(selected)
    if maximum is not None:
        selected = selected[:maximum]
    return tuple(selected)


def _examples_fingerprint(
    examples: Sequence[Example[InputsT, OutputT, MetadataT]],
) -> str:
    payload = json.dumps(
        [example.fingerprint() for example in examples],
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode()).hexdigest()


async def _evaluate_set(
    evaluation: Evaluation[InputsT, OutputT, MetadataT],
    candidate: Candidate,
    examples: Sequence[Example[InputsT, OutputT, MetadataT]],
    *,
    stage_id: str,
) -> tuple[CaseResult[OutputT], ...]:
    return tuple(
        [await evaluation.arun(candidate, example, stage_id=stage_id) for example in examples]
    )


__all__ = (
    "DataSplit",
    "FinalRescore",
    "RescoreResult",
    "Sampling",
    "arun_rescore",
    "rescore",
)
