from __future__ import annotations as _annotations

import asyncio
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping
from hashlib import sha256
from time import perf_counter
from typing import Generic, Literal, TypeAlias, TypeVar, cast

from ..candidates import Candidate
from ..errors import CandidateInjectionError, InfrastructureError, InvalidScoreError
from ..objectives import MetricResult, MetricRole, ScoreInput
from ..runtime import Runtime
from .cache import CacheStore, InMemoryCache
from .context import Context
from .evidence import Encoder
from .models import CaseResult, EvaluationConfig, Example
from .traces import ErrorInfo

InputsT = TypeVar("InputsT")
OutputT = TypeVar("OutputT")
MetadataT = TypeVar("MetadataT")
ResolvedT = TypeVar("ResolvedT")

MetricOutput: TypeAlias = ScoreInput | Mapping[str, ScoreInput]
Evaluator: TypeAlias = Callable[
    [Context[InputsT, OutputT, MetadataT]],
    MetricOutput | Awaitable[MetricOutput],
]
Strategy = Literal["output", "controlled"]


class Evaluation(Generic[InputsT, OutputT, MetadataT]):
    def __init__(
        self,
        runtime: Runtime[InputsT, OutputT],
        evaluate: Evaluator[InputsT, OutputT, MetadataT],
        *,
        strategy: Strategy,
        objective: str = "score",
        metric_roles: Mapping[str, MetricRole] | None = None,
        config: EvaluationConfig[InputsT, OutputT, MetadataT] | None = None,
        cache: CacheStore[OutputT] | None = None,
        encoder: Encoder | None = None,
        identity: str | None = None,
        deterministic: bool = True,
    ) -> None:
        if not objective:
            raise ValueError("objective cannot be empty.")
        self.runtime = runtime
        self.evaluate = evaluate
        self.strategy = strategy
        self.objective = objective
        self.metric_roles: dict[str, MetricRole] = dict(metric_roles or {})
        self.config = config or EvaluationConfig[InputsT, OutputT, MetadataT]()
        self.cache: CacheStore[OutputT] | None = (
            InMemoryCache() if cache is None and self.config.cache == "memory" else cache
        )
        self.encoder = encoder or Encoder()
        self.identity = identity or (
            f"{type(evaluate).__module__}.{type(evaluate).__qualname__}:{id(evaluate)}"
        )
        self.deterministic = deterministic

    @classmethod
    def output(
        cls,
        runtime: Runtime[InputsT, OutputT],
        score: Evaluator[InputsT, OutputT, MetadataT],
        *,
        objective: str = "score",
        metric_roles: Mapping[str, MetricRole] | None = None,
        config: EvaluationConfig[InputsT, OutputT, MetadataT] | None = None,
        cache: CacheStore[OutputT] | None = None,
        encoder: Encoder | None = None,
        identity: str | None = None,
        deterministic: bool = True,
    ) -> Evaluation[InputsT, OutputT, MetadataT]:
        return cls(
            runtime,
            score,
            strategy="output",
            objective=objective,
            metric_roles=metric_roles,
            config=config,
            cache=cache,
            encoder=encoder,
            identity=identity,
            deterministic=deterministic,
        )

    @classmethod
    def controlled(
        cls,
        runtime: Runtime[InputsT, OutputT],
        evaluate: Evaluator[InputsT, OutputT, MetadataT],
        *,
        objective: str = "score",
        metric_roles: Mapping[str, MetricRole] | None = None,
        config: EvaluationConfig[InputsT, OutputT, MetadataT] | None = None,
        cache: CacheStore[OutputT] | None = None,
        encoder: Encoder | None = None,
        identity: str | None = None,
        deterministic: bool = True,
    ) -> Evaluation[InputsT, OutputT, MetadataT]:
        return cls(
            runtime,
            evaluate,
            strategy="controlled",
            objective=objective,
            metric_roles=metric_roles,
            config=config,
            cache=cache,
            encoder=encoder,
            identity=identity,
            deterministic=deterministic,
        )

    def run(
        self,
        candidate: Candidate,
        example: Example[InputsT, OutputT, MetadataT],
        *,
        stage_id: str | None = None,
    ) -> CaseResult[OutputT]:
        from ..harness import run_awaitable_sync

        return run_awaitable_sync(self.arun(candidate, example, stage_id=stage_id))

    async def arun(
        self,
        candidate: Candidate,
        example: Example[InputsT, OutputT, MetadataT],
        *,
        stage_id: str | None = None,
    ) -> CaseResult[OutputT]:
        started = perf_counter()
        if self.config.validate_input is not None:
            self.config.validate_input(example.inputs)
        try:
            active_candidate = self.runtime.normalize_candidate(candidate)
        except CandidateInjectionError as error:
            if self.config.on_invalid_candidate == "raise":
                raise
            result = self._failure_result(
                started=started,
                error=error,
                category="candidate",
            )
            self._validate_result(result)
            return result
        if self.config.validate_candidate is not None:
            self.config.validate_candidate(active_candidate)

        cache_key = self._cache_key(active_candidate, example, stage_id=stage_id)
        cache = self.cache
        cached = cache.get(cache_key) if cache is not None and cache_key is not None else None
        if cached is not None:
            return cached.model_copy(update={"cache_hit": True})

        context = Context(
            runtime=self.runtime,
            example=example,
            candidate=active_candidate,
            stage_id=stage_id,
        )
        failure: tuple[Literal["candidate", "task", "evaluator", "infrastructure"], Exception]
        try:
            if self.strategy == "output":
                await context.arun()
            raw_metrics = await _resolve(self.evaluate(context))
        except asyncio.CancelledError:
            raise
        except CandidateInjectionError as error:
            failure = "candidate", error
        except InfrastructureError as error:
            failure = "infrastructure", error
        except Exception as error:
            failure = ("task" if context.task_failed_with(error) else "evaluator"), error
        else:
            if context.output is not None and self.config.validate_output is not None:
                self.config.validate_output(context.output)
            metrics = self._metrics(raw_metrics)
            result = self._result(context, metrics, started=started)
            self._validate_result(result)
            self._cache(cache_key, result)
            return result

        category, error = failure
        action = {
            "candidate": self.config.on_invalid_candidate,
            "task": self.config.on_task_error,
            "evaluator": self.config.on_evaluator_error,
            "infrastructure": self.config.on_infrastructure_error,
        }[category]
        if action == "raise":
            raise error
        result = self._failure_result(
            started=started,
            error=error,
            category=category,
            context=context,
        )
        self._validate_result(result)
        if self.config.cache_failures:
            self._cache(cache_key, result)
        return result

    def _metrics(self, output: MetricOutput) -> dict[str, MetricResult]:
        raw_metrics = (
            cast("Mapping[str, ScoreInput]", output)
            if isinstance(output, Mapping)
            else {self.objective: output}
        )
        metrics = {name: self._metric(name, value) for name, value in raw_metrics.items()}
        if self.objective not in metrics:
            metrics[self.objective] = self._invalid_metric(
                self.objective,
                f"Missing objective metric '{self.objective}'.",
            )
        return metrics

    def _metric(self, name: str, value: ScoreInput) -> MetricResult:
        role = self._role(name)
        if isinstance(value, MetricResult):
            score = value.score
            feedback = value.feedback
            side_info = value.side_info
        elif isinstance(value, bool):
            score = 1.0 if value else 0.0
            feedback = None
            side_info = {}
        elif isinstance(value, int | float):
            score = float(value)
            feedback = None
            side_info = {}
        else:
            return self._invalid_metric(name, f"Metric '{name}' is not a numeric scalar.")

        if not math.isfinite(score):
            return self._invalid_metric(name, f"Metric '{name}' must be finite.")
        if self.config.min_score is not None and score < self.config.min_score:
            return self._invalid_metric(name, f"Metric '{name}' is below min_score.")
        if self.config.max_score is not None and score > self.config.max_score:
            return self._invalid_metric(name, f"Metric '{name}' is above max_score.")
        return MetricResult(
            score=score,
            role=role,
            feedback=feedback,
            side_info=side_info,
        )

    def _invalid_metric(self, name: str, message: str) -> MetricResult:
        if self.config.invalid_score == "raise":
            raise InvalidScoreError(message)
        role = self._role(name)
        return MetricResult(
            score=self.config.failure_score,
            role=role,
            feedback=message,
        )

    def _role(self, name: str) -> MetricRole:
        configured = self.metric_roles.get(name)
        if configured is not None:
            return configured
        return "objective" if name == self.objective else "diagnostic"

    def _result(
        self,
        context: Context[InputsT, OutputT, MetadataT],
        metrics: dict[str, MetricResult],
        *,
        started: float,
    ) -> CaseResult[OutputT]:
        objective = metrics[self.objective]
        return CaseResult(
            output=context.output,
            metrics=metrics,
            objectives={self.objective: objective.score},
            feedback={
                name: metric.feedback
                for name, metric in metrics.items()
                if metric.feedback is not None
            },
            side_info={
                name: metric.side_info for name, metric in metrics.items() if metric.side_info
            },
            traces=context.traces,
            artifacts=context.artifacts,
            task_error=context.task_errors[-1] if context.task_errors else None,
            duration_seconds=perf_counter() - started,
            invocation_count=len(context.invocations),
        )

    def _failure_result(
        self,
        *,
        started: float,
        error: Exception,
        category: Literal["candidate", "task", "evaluator", "infrastructure"],
        context: Context[InputsT, OutputT, MetadataT] | None = None,
    ) -> CaseResult[OutputT]:
        info = ErrorInfo.from_exception(error)
        metric = MetricResult(
            score=self.config.failure_score,
            role="objective",
            feedback=f"{category} failure: {error}",
        )
        return CaseResult(
            output=context.output if context is not None else None,
            metrics={self.objective: metric},
            objectives={self.objective: metric.score},
            feedback={self.objective: metric.feedback or ""},
            traces=context.traces if context is not None else (),
            artifacts=context.artifacts if context is not None else (),
            candidate_error=info if category == "candidate" else None,
            task_error=info if category == "task" else None,
            evaluator_error=info if category == "evaluator" else None,
            infrastructure_error=info if category == "infrastructure" else None,
            duration_seconds=perf_counter() - started,
            invocation_count=len(context.invocations) if context is not None else 0,
        )

    def _cache_key(
        self,
        candidate: Candidate,
        example: Example[InputsT, OutputT, MetadataT],
        *,
        stage_id: str | None,
    ) -> str | None:
        if self.cache is None:
            return None
        if not self.deterministic and not self.config.cache_nondeterministic:
            return None
        payload = {
            "candidate": candidate.fingerprint(),
            "example": example.fingerprint(encoder=self.encoder),
            "runtime": self.runtime.identity,
            "evaluator": self.identity,
            "strategy": self.strategy,
            "objective": self.objective,
            "metric_roles": self.metric_roles,
            "stage": stage_id,
            "config": self.config.model_dump(
                mode="json",
                exclude={
                    "validate_input",
                    "validate_candidate",
                    "validate_output",
                    "validate_result",
                },
            ),
            "validators": [
                id(validator)
                for validator in (
                    self.config.validate_input,
                    self.config.validate_candidate,
                    self.config.validate_output,
                    self.config.validate_result,
                )
                if validator is not None
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha256(encoded.encode()).hexdigest()

    def _cache(self, key: str | None, result: CaseResult[OutputT]) -> None:
        if key is not None and self.cache is not None:
            self.cache.set(key, result)

    def _validate_result(self, result: CaseResult[OutputT]) -> None:
        if self.config.validate_result is not None:
            self.config.validate_result(result)


async def _resolve(value: ResolvedT | Awaitable[ResolvedT]) -> ResolvedT:
    if inspect.isawaitable(value):
        return await cast("Awaitable[ResolvedT]", value)
    return value


__all__ = (
    "Evaluation",
    "Evaluator",
    "MetricOutput",
)
