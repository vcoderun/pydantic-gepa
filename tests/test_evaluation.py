from __future__ import annotations as _annotations

import asyncio
import math

import pytest
from pydantic import ValidationError

from pydantic_gepa import (
    Attachment,
    Candidate,
    CandidateInjectionError,
    CaseResult,
    Context,
    Evaluation,
    EvaluationConfig,
    Example,
    InfrastructureError,
    InMemoryCache,
    InvalidScoreError,
    MetricResult,
    MetricRole,
    Runtime,
)
from pydantic_gepa.evaluation import CacheStore, MetricOutput


@pytest.mark.asyncio
async def test_output_evaluation_runs_once_and_preserves_all_metric_evidence() -> None:
    task_calls = 0

    def task(value: str) -> str:
        nonlocal task_calls
        task_calls += 1
        return value.upper()

    async def score(ctx: Context[str, str, dict[str, str]]) -> MetricOutput:
        await asyncio.sleep(0)
        ctx.record_trace("judge", kind="evaluator", output={"accepted": True})
        ctx.artifact(Attachment(kind="document", reference="judge/reason.json"))
        return {
            "quality": MetricResult(
                score=0.9,
                feedback="Correct answer.",
                side_info={"rubric": {"exact": True}},
            ),
            "cost": 0.2,
            "valid": True,
        }

    roles: dict[str, MetricRole] = {"cost": "constraint", "valid": "diagnostic"}
    evaluation = Evaluation.output(
        Runtime(task, identity="uppercase-task"),
        score,
        objective="quality",
        metric_roles=roles,
        identity="quality-score",
    )
    result = await evaluation.arun(
        Candidate(values={"prompt": "uppercase"}),
        Example(inputs="hello", expected_output="HELLO", metadata={"split": "train"}),
        stage_id="generation",
    )

    assert task_calls == 1
    assert result.output == "HELLO"
    assert result.invocation_count == 1
    assert result.objectives == {"quality": 0.9}
    assert result.metrics["quality"].role == "objective"
    assert result.metrics["cost"].role == "constraint"
    assert result.metrics["valid"].role == "diagnostic"
    assert result.feedback == {"quality": "Correct answer."}
    assert result.side_info == {"quality": {"rubric": {"exact": True}}}
    assert result.traces[0].component == "judge"
    assert result.artifacts[0].reference == "judge/reason.json"
    assert result.task_error is None
    assert result.duration_seconds >= 0


@pytest.mark.asyncio
async def test_controlled_evaluator_owns_zero_or_many_runtime_calls() -> None:
    async def task(value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    async def repeated(ctx: Context[int, int, None]) -> MetricOutput:
        first = await ctx.arun()
        second = await ctx.arun_with(3)
        return {"score": float(first + second)}

    many = Evaluation.controlled(Runtime(task), repeated)
    many_result = await many.arun(Candidate(), Example(inputs=2))

    assert many_result.output == 6
    assert many_result.objectives == {"score": 10.0}
    assert many_result.invocation_count == 2

    def without_runtime(ctx: Context[int, int, None]) -> MetricOutput:
        return {"score": float(ctx.example.inputs)}

    zero = Evaluation.controlled(Runtime(task), without_runtime)
    zero_result = await zero.arun(Candidate(), Example(inputs=4))

    assert zero_result.output is None
    assert zero_result.objectives == {"score": 4.0}
    assert zero_result.invocation_count == 0


def test_sync_entrypoint_supports_async_task_and_sync_scorer() -> None:
    async def task(value: str) -> str:
        await asyncio.sleep(0)
        return value[::-1]

    def score(ctx: Context[str, str, None]) -> MetricOutput:
        return ctx.output == "cba"

    evaluation = Evaluation.output(Runtime(task), score)
    result = evaluation.run(Candidate(), Example(inputs="abc"))

    assert result.output == "cba"
    assert result.objectives == {"score": 1.0}


@pytest.mark.asyncio
async def test_sync_controlled_evaluator_can_call_async_runtime() -> None:
    async def task(value: int) -> int:
        await asyncio.sleep(0)
        return value + 1

    def evaluate(ctx: Context[int, int, None]) -> MetricOutput:
        return float(ctx.run())

    result = await Evaluation.controlled(Runtime(task), evaluate).arun(
        Candidate(),
        Example(inputs=2),
    )

    assert result.output == 3
    assert result.invocation_count == 1


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, "bad", None])
def test_invalid_scalar_values_raise(value: float | str | None) -> None:
    def score(_: Context[str, str, None]) -> MetricOutput:
        return value

    evaluation = Evaluation.output(Runtime(lambda text: text), score)

    with pytest.raises(InvalidScoreError):
        evaluation.run(Candidate(), Example(inputs="input"))


@pytest.mark.parametrize(
    ("score", "config", "message"),
    [
        (-0.1, EvaluationConfig(min_score=0.0), "below min_score"),
        (1.1, EvaluationConfig(max_score=1.0), "above max_score"),
    ],
)
def test_score_range_violations_raise(
    score: float,
    config: EvaluationConfig,
    message: str,
) -> None:
    evaluation = Evaluation.output(
        Runtime(lambda text: text),
        lambda _: score,
        config=config,
    )

    with pytest.raises(InvalidScoreError, match=message):
        evaluation.run(Candidate(), Example(inputs="input"))


def test_invalid_score_fallback_covers_missing_and_wrong_metrics() -> None:
    config = EvaluationConfig(
        invalid_score="use_failure_score",
        failure_score=0.25,
        min_score=0.0,
        max_score=1.0,
    )
    missing = Evaluation.output(
        Runtime(lambda text: text),
        lambda _: {"diagnostic": "not numeric"},
        objective="quality",
        config=config,
    ).run(Candidate(), Example(inputs="input"))

    assert missing.objectives == {"quality": 0.25}
    assert missing.metrics["diagnostic"].score == 0.25
    assert "not a numeric scalar" in (missing.metrics["diagnostic"].feedback or "")
    assert "Missing objective" in (missing.metrics["quality"].feedback or "")

    with pytest.raises(ValidationError):
        EvaluationConfig(failure_score=math.nan)
    with pytest.raises(ValidationError):
        EvaluationConfig(min_score=1.0, max_score=0.0)
    with pytest.raises(ValidationError):
        EvaluationConfig(failure_score=0.0, min_score=0.5)
    with pytest.raises(ValidationError):
        EvaluationConfig(failure_score=1.0, max_score=0.5)


def test_invalid_candidate_has_independent_failure_policy() -> None:
    evaluation = Evaluation.output(
        Runtime(lambda text: text, required_components=("prompt",)),
        lambda _: 1.0,
        config=EvaluationConfig(on_invalid_candidate="record"),
    )
    result = evaluation.run(Candidate(), Example(inputs="input"))

    assert result.candidate_error is not None
    assert result.candidate_error.kind == "CandidateInjectionError"
    assert result.task_error is None
    assert result.invocation_count == 0

    raising = Evaluation.output(
        Runtime(lambda text: text, required_components=("prompt",)),
        lambda _: 1.0,
    )
    with pytest.raises(CandidateInjectionError):
        raising.run(Candidate(), Example(inputs="input"))

    def reject_during_evaluation(_: Context[str, str, None]) -> MetricOutput:
        raise CandidateInjectionError("candidate rejected by runtime")

    runtime_rejection = Evaluation.controlled(
        Runtime(lambda text: text),
        reject_during_evaluation,
        config=EvaluationConfig(on_invalid_candidate="record"),
    ).run(Candidate(), Example(inputs="input"))
    assert runtime_rejection.candidate_error is not None
    assert runtime_rejection.candidate_error.message == "candidate rejected by runtime"

    with pytest.raises(ValueError, match="objective"):
        Evaluation.output(Runtime(lambda text: text), lambda _: 1.0, objective="")


def test_task_evaluator_and_infrastructure_failures_are_distinct() -> None:
    def task_failure(_: str) -> str:
        raise ValueError("task failed")

    task_result = Evaluation.output(
        Runtime(task_failure),
        lambda _: 1.0,
        config=EvaluationConfig(on_task_error="record"),
    ).run(Candidate(), Example(inputs="input"))
    assert task_result.task_error is not None
    assert task_result.evaluator_error is None

    def evaluator_failure(_: Context[str, str, None]) -> MetricOutput:
        raise RuntimeError("evaluator failed")

    evaluator_result = Evaluation.output(
        Runtime(lambda text: text),
        evaluator_failure,
        config=EvaluationConfig(on_evaluator_error="record"),
    ).run(Candidate(), Example(inputs="input"))
    assert evaluator_result.evaluator_error is not None
    assert evaluator_result.task_error is None

    def infrastructure_failure(_: Context[str, str, None]) -> MetricOutput:
        raise InfrastructureError("store unavailable")

    infrastructure_result = Evaluation.controlled(
        Runtime(lambda text: text),
        infrastructure_failure,
        config=EvaluationConfig(on_infrastructure_error="record"),
    ).run(Candidate(), Example(inputs="input"))
    assert infrastructure_result.infrastructure_error is not None

    with pytest.raises(ValueError, match="task failed"):
        Evaluation.output(Runtime(task_failure), lambda _: 1.0).run(
            Candidate(),
            Example(inputs="input"),
        )
    with pytest.raises(RuntimeError, match="evaluator failed"):
        Evaluation.output(Runtime(lambda text: text), evaluator_failure).run(
            Candidate(),
            Example(inputs="input"),
        )
    with pytest.raises(InfrastructureError, match="store unavailable"):
        Evaluation.controlled(Runtime(lambda text: text), infrastructure_failure).run(
            Candidate(),
            Example(inputs="input"),
        )


@pytest.mark.asyncio
async def test_evaluation_cancellation_is_never_converted_to_failure_score() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def task(_: str) -> str:
        entered.set()
        await release.wait()
        return "done"

    evaluation = Evaluation.output(
        Runtime(task),
        lambda _: 1.0,
        config=EvaluationConfig(on_task_error="record"),
    )
    running = asyncio.create_task(evaluation.arun(Candidate(), Example(inputs="input")))
    await entered.wait()
    running.cancel()

    with pytest.raises(asyncio.CancelledError):
        await running
    release.set()
    assert await task("input") == "done"


def test_candidate_aware_cache_hits_and_invalidates_relevant_inputs() -> None:
    calls = 0

    def task(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    def normalize(candidate: Candidate) -> Candidate:
        return Candidate(values={"prompt": candidate.values["prompt"].strip()})

    cache: InMemoryCache[str] = InMemoryCache()
    evaluation = Evaluation.output(
        Runtime(task, normalize=normalize, identity="task-v1"),
        lambda _: 1.0,
        cache=cache,
        identity="score-v1",
    )
    example = Example(inputs="one", name="case-one")

    first = evaluation.run(Candidate(values={"prompt": " stable "}), example, stage_id="one")
    second = evaluation.run(Candidate(values={"prompt": "stable"}), example, stage_id="one")
    evaluation.run(Candidate(values={"prompt": "changed"}), example, stage_id="one")
    evaluation.run(Candidate(values={"prompt": "stable"}), example, stage_id="two")
    evaluation.run(
        Candidate(values={"prompt": "stable"}),
        Example(inputs="two", name="case-two"),
        stage_id="one",
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert calls == 4
    assert len(cache) == 4
    assert isinstance(cache, CacheStore)
    cache.clear()
    assert len(cache) == 0


def test_cache_can_be_disabled_or_enabled_for_nondeterministic_evaluators() -> None:
    calls = 0

    def task(value: str) -> str:
        nonlocal calls
        calls += 1
        return value

    disabled = Evaluation.output(Runtime(task), lambda _: 1.0)
    disabled.run(Candidate(), Example(inputs="input"))
    disabled.run(Candidate(), Example(inputs="input"))
    assert calls == 2

    nondeterministic = Evaluation.output(
        Runtime(task, identity="task"),
        lambda _: 1.0,
        config=EvaluationConfig(cache="memory"),
        identity="score",
        deterministic=False,
    )
    nondeterministic.run(Candidate(), Example(inputs="input"))
    nondeterministic.run(Candidate(), Example(inputs="input"))
    assert calls == 4

    opted_in = Evaluation.output(
        Runtime(task, identity="task"),
        lambda _: 1.0,
        config=EvaluationConfig(cache="memory", cache_nondeterministic=True),
        identity="score",
        deterministic=False,
    )
    opted_in.run(Candidate(), Example(inputs="input"))
    cached = opted_in.run(Candidate(), Example(inputs="input"))
    assert cached.cache_hit is True
    assert calls == 5


def test_failure_cache_policy_is_explicit() -> None:
    calls = 0

    def fail(_: str) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("failed")

    uncached = Evaluation.output(
        Runtime(fail, identity="failing-task"),
        lambda _: 1.0,
        config=EvaluationConfig(cache="memory", on_task_error="record"),
        identity="score",
    )
    uncached.run(Candidate(), Example(inputs="input"))
    uncached.run(Candidate(), Example(inputs="input"))
    assert calls == 2

    cached_failures = Evaluation.output(
        Runtime(fail, identity="failing-task"),
        lambda _: 1.0,
        config=EvaluationConfig(
            cache="memory",
            cache_failures=True,
            on_task_error="record",
        ),
        identity="score",
    )
    cached_failures.run(Candidate(), Example(inputs="input"))
    cached = cached_failures.run(Candidate(), Example(inputs="input"))
    assert cached.cache_hit is True
    assert calls == 3


def test_case_result_is_a_stable_typed_result_model() -> None:
    result = CaseResult[str](duration_seconds=0.0, invocation_count=0)

    assert result.output is None
    assert result.metrics == {}
