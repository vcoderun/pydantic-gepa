from __future__ import annotations as _annotations

import pytest
from pydantic import ValidationError

from pydantic_gepa import (
    Attachment,
    Candidate,
    CaseResult,
    DataSplit,
    Evaluation,
    EvaluationConfig,
    Example,
    Runtime,
    arun_rescore,
    rescore,
)


def test_example_pair_constructors_identity_and_fingerprint() -> None:
    attachment = Attachment(kind="image", reference="image.png")
    example = Example.from_pair(
        "input",
        "output",
        id="case-id",
        name="case-name",
        metadata={"split": "train"},
        attachments=(attachment,),
    )
    pairs = Example.from_pairs(
        [("one", "ONE"), ("two", "TWO")],
        metadata={"source": "pairs"},
    )

    assert example.identity() == "case-id"
    assert example.fingerprint() == example.fingerprint()
    assert example.fingerprint() != Example.from_pair("changed", "output").fingerprint()
    assert [item.inputs for item in pairs] == ["one", "two"]
    assert pairs[0].metadata == {"source": "pairs"}
    assert Example(inputs="input", name="named").identity() == "named"
    generated_identity = Example(inputs="input").identity()
    assert len(generated_identity) == 64


def test_data_split_requires_explicit_non_overlapping_train_and_validation() -> None:
    first = Example(inputs="one", id="one")
    second = Example(inputs="two", id="two")

    split = DataSplit.from_sets(train=[first], validation=[second])
    assert split.train == (first,)
    assert split.validation == (second,)

    with pytest.raises(ValidationError, match="train cannot be empty"):
        DataSplit(train=(), validation=(second,))
    with pytest.raises(ValidationError, match="validation cannot be empty"):
        DataSplit(train=(first,), validation=())
    with pytest.raises(ValidationError, match="overlap"):
        DataSplit(train=(first,), validation=(first,))

    reused = DataSplit(
        train=(first,),
        validation=(first,),
        allow_train_validation_overlap=True,
    )
    assert reused.train == reused.validation


def test_data_split_sampling_limits_and_fingerprints_are_deterministic() -> None:
    examples = [Example(inputs=index, id=f"case-{index}") for index in range(10)]
    first = DataSplit.partition(
        examples,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )
    second = DataSplit.partition(
        examples,
        validation_fraction=0.2,
        test_fraction=0.2,
        seed=42,
    )

    assert [example.id for example in first.train] == [example.id for example in second.train]
    assert len(first.train) == 6
    assert len(first.validation) == 2
    assert len(first.test) == 2
    assert first.fingerprints() == second.fingerprints()

    limited = DataSplit.from_sets(
        train=examples[:5],
        validation=examples[5:8],
        test=examples[8:],
        sampling="random",
        seed=7,
        max_train=2,
        max_validation=1,
        max_test=1,
    )
    assert len(limited.train) == 2
    assert len(limited.validation) == 1
    assert len(limited.test) == 1

    ordered = DataSplit.partition(
        examples,
        validation_fraction=0.2,
        sampling="input_order",
    )
    assert [example.id for example in ordered.train[:2]] == ["case-0", "case-1"]


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_data_split_rejects_invalid_validation_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="validation_fraction"):
        DataSplit.partition([Example(inputs=1), Example(inputs=2)], validation_fraction=fraction)


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_data_split_rejects_invalid_test_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        DataSplit.partition(
            [Example(inputs=1), Example(inputs=2)],
            test_fraction=fraction,
        )


def test_data_split_rejects_impossible_or_invalid_limits() -> None:
    examples = [Example(inputs=1), Example(inputs=2)]
    with pytest.raises(ValueError, match="sum to less than 1"):
        DataSplit.partition(examples, validation_fraction=0.6, test_fraction=0.4)
    with pytest.raises(ValueError, match="no training"):
        DataSplit.partition([Example(inputs=1)], validation_fraction=0.5)
    with pytest.raises(ValueError, match="at least 1"):
        DataSplit.from_sets(
            train=[Example(inputs=1, id="train")],
            validation=[Example(inputs=2, id="validation")],
            max_train=0,
        )


@pytest.mark.asyncio
async def test_final_rescore_modes_use_the_accepted_candidate() -> None:
    def double(value: int) -> int:
        return value * 2

    evaluation = Evaluation.output(
        Runtime(double, identity="double"),
        lambda ctx: float(ctx.output or 0),
        identity="value-score",
    )
    data = DataSplit(
        train=(Example(inputs=1, id="train-1"), Example(inputs=2, id="train-2")),
        validation=(Example(inputs=3, id="validation-1"),),
        test=(Example(inputs=4, id="test-1"),),
    )
    candidate = Candidate(values={"prompt": "accepted"})

    none = await arun_rescore(evaluation, candidate, data, policy="none")
    validation = await arun_rescore(evaluation, candidate, data, policy="validation")
    train_validation = await arun_rescore(
        evaluation,
        candidate,
        data,
        policy="train_validation",
    )
    all_sets = await arun_rescore(evaluation, candidate, data, policy="all")

    assert none.train == none.validation == none.test == ()
    assert [result.output for result in validation.validation] == [6]
    assert [result.output for result in train_validation.train] == [2, 4]
    assert [result.output for result in all_sets.test] == [8]
    assert all_sets.candidate_fingerprint == candidate.fingerprint()
    assert all_sets.split_fingerprints == data.fingerprints()

    sync_result = rescore(evaluation, candidate, data, policy="validation")
    assert sync_result.validation[0].output == 6


def test_validation_hooks_run_with_normalized_values_and_results() -> None:
    observed: list[str] = []

    def uppercase(value: str) -> str:
        return value.upper()

    def validate_input(value: str) -> None:
        observed.append(f"input:{value}")

    def validate_candidate(candidate: Candidate) -> None:
        observed.append(f"candidate:{candidate.values['prompt']}")

    def validate_output(value: str) -> None:
        observed.append(f"output:{value}")

    def validate_result(result: CaseResult[str]) -> None:
        observed.append(f"result:{result.objectives['score']}")

    config = EvaluationConfig[str, str, None](
        cache="memory",
        validate_input=validate_input,
        validate_candidate=validate_candidate,
        validate_output=validate_output,
        validate_result=validate_result,
    )
    evaluation = Evaluation.output(
        Runtime(
            uppercase,
            normalize=lambda candidate: Candidate(
                values={"prompt": candidate.values["prompt"].strip()}
            ),
            identity="uppercase",
        ),
        lambda _: 1.0,
        config=config,
        identity="score",
    )

    first = evaluation.run(
        Candidate(values={"prompt": " stable "}),
        Example(inputs="hello"),
    )
    second = evaluation.run(
        Candidate(values={"prompt": "stable"}),
        Example(inputs="hello"),
    )

    assert first.output == "HELLO"
    assert second.cache_hit is True
    assert observed == [
        "input:hello",
        "candidate:stable",
        "output:HELLO",
        "result:1.0",
        "input:hello",
        "candidate:stable",
    ]


def test_result_validation_also_applies_to_recorded_failures() -> None:
    validated: list[str] = []

    def fail(_: str) -> str:
        raise RuntimeError("failed")

    def validate_result(result: CaseResult[str]) -> None:
        validated.append(result.task_error.kind if result.task_error is not None else "success")

    evaluation = Evaluation.output(
        Runtime(fail),
        lambda _: 1.0,
        config=EvaluationConfig[str, str, None](
            on_task_error="record",
            validate_result=validate_result,
        ),
    )
    evaluation.run(Candidate(), Example(inputs="input"))

    assert validated == ["RuntimeError"]


def test_validation_hooks_propagate_user_validation_failures() -> None:
    def reject_input(_: str) -> None:
        raise ValueError("invalid input")

    evaluation = Evaluation.output(
        Runtime(lambda value: value),
        lambda _: 1.0,
        config=EvaluationConfig[str, str, None](validate_input=reject_input),
    )

    with pytest.raises(ValueError, match="invalid input"):
        evaluation.run(Candidate(), Example(inputs="input"))
