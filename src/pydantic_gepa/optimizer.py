from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from inspect import Parameter, signature
from typing import Generic, Protocol, TypeAlias, TypeVar, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from ._version import __version__
from .adapter import PydanticGEPAAdapter
from .candidates import Candidate
from .configuration import BudgetConfig, ConfigurationError, EvaluationSetConfig, GEPAConfig
from .errors import OptimizationDependencyError
from .results import (
    BudgetSummary,
    CandidateDelta,
    CandidateStatus,
    CandidateSummary,
    PydanticGEPAResult,
    ResultBackend,
    ScoreSummary,
)
from .state import CompatibilityFingerprint, FileRunStore, RunState, content_fingerprint
from .values import JsonScalar, JsonValue, Metadata, SerializableValue

CandidateValues = Mapping[str, str]
Numeric = int | float
CaseT = TypeVar("CaseT")
RolloutOutputT = TypeVar("RolloutOutputT")
EvaluatorT = TypeVar("EvaluatorT")


@runtime_checkable
class BestCandidateResult(Protocol):
    @property
    def best_candidate(self) -> Candidate | CandidateValues: ...


@runtime_checkable
class BestProgramResult(Protocol):
    @property
    def best_program(self) -> CandidateValues: ...


@runtime_checkable
class BestScoreResult(Protocol):
    @property
    def best_score(self) -> Numeric: ...


@runtime_checkable
class BestValScoreResult(Protocol):
    @property
    def best_val_score(self) -> Numeric: ...


@runtime_checkable
class ValidationScoresResult(Protocol):
    @property
    def validation_scores(self) -> Sequence[Numeric]: ...


@runtime_checkable
class CandidateHistoryResult(Protocol):
    @property
    def candidate_history(self) -> Sequence[CandidateSummary | HistoryEntry]: ...


@runtime_checkable
class ObjectiveScoresResult(Protocol):
    @property
    def objective_scores(self) -> Sequence[Mapping[str, Numeric]]: ...


@runtime_checkable
class CandidateTreeDotResult(Protocol):
    @property
    def candidate_tree_dot(self) -> str | Callable[[], str] | None: ...


@runtime_checkable
class CandidateTreeHtmlResult(Protocol):
    @property
    def candidate_tree_html(self) -> str | Callable[[], str] | None: ...


@runtime_checkable
class BestIndexResult(Protocol):
    @property
    def best_idx(self) -> int: ...


@runtime_checkable
class CandidatesResult(Protocol):
    @property
    def candidates(self) -> Sequence[CandidateValues]: ...


@runtime_checkable
class ParentsResult(Protocol):
    @property
    def parents(self) -> Sequence[Sequence[int | None]]: ...


@runtime_checkable
class ValAggregateScoresResult(Protocol):
    @property
    def val_aggregate_scores(self) -> Sequence[Numeric]: ...


@runtime_checkable
class ValAggregateSubscoresResult(Protocol):
    @property
    def val_aggregate_subscores(self) -> Sequence[Mapping[str, Numeric]] | None: ...


@runtime_checkable
class ValSubscoresResult(Protocol):
    @property
    def val_subscores(self) -> Sequence[Mapping[str, Numeric]]: ...


@runtime_checkable
class TotalMetricCallsResult(Protocol):
    @property
    def total_metric_calls(self) -> int | None: ...


@runtime_checkable
class NumFullValEvalsResult(Protocol):
    @property
    def num_full_val_evals(self) -> int | None: ...


@runtime_checkable
class RunDirResult(Protocol):
    @property
    def run_dir(self) -> str | None: ...


@runtime_checkable
class SeedResult(Protocol):
    @property
    def seed(self) -> int | None: ...


@runtime_checkable
class PerObjectiveBestCandidatesResult(Protocol):
    @property
    def per_objective_best_candidates(self) -> Mapping[str, Sequence[int]] | None: ...


@runtime_checkable
class ObjectiveParetoFrontResult(Protocol):
    @property
    def objective_pareto_front(self) -> Mapping[str, Numeric] | None: ...


@runtime_checkable
class HistoryValues(Protocol):
    @property
    def values(self) -> CandidateValues: ...


@runtime_checkable
class HistoryCandidate(Protocol):
    @property
    def candidate(self) -> CandidateValues: ...


@runtime_checkable
class HistoryScore(Protocol):
    @property
    def score(self) -> Numeric: ...


@runtime_checkable
class HistoryCandidateId(Protocol):
    @property
    def candidate_id(self) -> str: ...


@runtime_checkable
class HistoryParentIds(Protocol):
    @property
    def parent_ids(self) -> Sequence[str]: ...


@runtime_checkable
class HistoryGeneration(Protocol):
    @property
    def generation(self) -> int | None: ...


@runtime_checkable
class HistoryMetadata(Protocol):
    @property
    def metadata(self) -> Metadata: ...


HistoryEntry = (
    HistoryValues
    | HistoryCandidate
    | HistoryScore
    | HistoryCandidateId
    | HistoryParentIds
    | HistoryGeneration
    | HistoryMetadata
)
GEPAResult: TypeAlias = (
    BestCandidateResult
    | BestProgramResult
    | BestScoreResult
    | BestValScoreResult
    | ValidationScoresResult
    | CandidateHistoryResult
    | ObjectiveScoresResult
    | CandidateTreeDotResult
    | CandidateTreeHtmlResult
    | BestIndexResult
    | CandidatesResult
    | ParentsResult
    | ValAggregateScoresResult
    | ValAggregateSubscoresResult
    | ValSubscoresResult
    | TotalMetricCallsResult
    | NumFullValEvalsResult
    | RunDirResult
    | SeedResult
    | PerObjectiveBestCandidatesResult
    | ObjectiveParetoFrontResult
)
OptimizeFn = Callable[..., GEPAResult]


class PydanticGEPAOptimizer(BaseModel, Generic[CaseT, RolloutOutputT, EvaluatorT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    adapter: PydanticGEPAAdapter[CaseT, RolloutOutputT, EvaluatorT]
    initial_candidate: Candidate
    optimize_fn: OptimizeFn | None = None

    def optimize(
        self,
        *,
        trainset: Sequence[CaseT],
        valset: Sequence[CaseT] | None = None,
        initial_candidate: Candidate | None = None,
        config: GEPAConfig | None = None,
        max_metric_calls: int | None = None,
        allow_same_train_val: bool | None = None,
        **kwargs: SerializableValue,
    ) -> PydanticGEPAResult:
        active_config = _optimization_config(
            config=config,
            max_metric_calls=max_metric_calls,
            allow_same_train_val=allow_same_train_val,
            legacy_kwargs=kwargs,
        )
        active_valset = list(trainset if valset is None else valset)
        if valset is None and not active_config.evaluation_sets.allow_same_train_validation:
            raise OptimizationDependencyError(
                "valset is required unless allow_same_train_val=True."
            )

        if (
            self.optimize_fn is None
            and self.adapter.propose_new_texts is None
            and active_config.reflection.proposer is None
            and active_config.reflection.model is None
        ):
            raise ConfigurationError(
                "reflection.model is required when the adapter has no candidate proposer."
            )
        if (
            self.adapter.propose_new_texts is not None
            and active_config.reflection.proposer is not None
        ):
            raise ConfigurationError(
                "Configure a candidate proposer on either the adapter or GEPAConfig, not both."
            )

        seed_candidate = initial_candidate or self.initial_candidate
        store: FileRunStore | None = None
        run_state: RunState | None = None
        if active_config.run.directory is not None:
            store = FileRunStore(
                active_config.run.directory,
                run_id=active_config.run.id,
                resume=active_config.run.resume,
                fresh=active_config.run.fresh,
            )
            config_values = active_config.to_backend_kwargs()
            config_values["run_dir"] = None
            try:
                backend_version = version("gepa")
            except PackageNotFoundError:
                backend_version = "not-installed"
            fingerprint = CompatibilityFingerprint.from_dimensions(
                {
                    "pydantic-gepa": __version__,
                    "gepa": backend_version,
                    "candidate_schema": content_fingerprint(sorted(seed_candidate.values)),
                    "config": content_fingerprint(config_values),
                    "trainset": content_fingerprint(trainset),
                    "valset": content_fingerprint(active_valset),
                    "adapter": f"{type(self.adapter).__module__}.{type(self.adapter).__qualname__}",
                    **active_config.run.compatibility,
                }
            )
            run_state = store.prepare(
                fingerprint=fingerprint,
                initial_candidate=seed_candidate,
            )
            completed = store.load_result(PydanticGEPAResult)
            if run_state.status == "completed" and completed is not None:
                return completed
            active_config = active_config.model_copy(
                update={
                    "run": active_config.run.model_copy(
                        update={"directory": store.backend_directory}
                    )
                }
            )

        optimize_fn = self.optimize_fn or _load_gepa_optimize()
        backend_kwargs = _backend_kwargs(optimize_fn, active_config)
        try:
            raw_result = optimize_fn(
                seed_candidate=seed_candidate.to_gepa_dict(),
                trainset=list(trainset),
                valset=active_valset,
                adapter=self.adapter,
                **backend_kwargs,
            )
            result = result_from_gepa(
                raw_result,
                run_id=active_config.run.id,
                budget_limit=active_config.budget.max_metric_calls,
            ).normalize_candidates(self.adapter.normalize_candidate)
        except Exception as exc:
            if store is not None and run_state is not None:
                store.checkpoint(
                    run_state.model_copy(
                        update={
                            "status": "failed",
                            "backend_checkpoint": str(store.backend_directory),
                            "error": str(exc),
                        }
                    )
                )
            raise
        if store is not None and run_state is not None:
            store.write_candidate(result.best_candidate)
            store.write_result(result)
            store.checkpoint(
                run_state.model_copy(
                    update={
                        "status": "completed",
                        "accepted_candidate": result.best_candidate,
                        "metric_calls": result.total_metric_calls or 0,
                        "backend_checkpoint": str(store.backend_directory),
                    }
                )
            )
        return result


def _optimization_config(
    *,
    config: GEPAConfig | None,
    max_metric_calls: int | None,
    allow_same_train_val: bool | None,
    legacy_kwargs: Mapping[str, SerializableValue],
) -> GEPAConfig:
    if config is not None:
        if max_metric_calls is not None or allow_same_train_val is not None or legacy_kwargs:
            raise ConfigurationError(
                "GEPAConfig cannot be combined with max_metric_calls, "
                "allow_same_train_val, or legacy GEPA kwargs."
            )
        return config
    values = dict(legacy_kwargs)
    if max_metric_calls is not None:
        values["max_metric_calls"] = max_metric_calls
    if legacy_kwargs:
        active = GEPAConfig.from_legacy_kwargs(values)
    elif max_metric_calls is not None:
        active = GEPAConfig(budget=BudgetConfig(max_metric_calls=max_metric_calls))
    else:
        active = GEPAConfig()
    if allow_same_train_val is None:
        return active
    return active.model_copy(
        update={
            "evaluation_sets": EvaluationSetConfig(
                allow_same_train_validation=allow_same_train_val
            ),
        }
    )


def result_from_gepa(
    raw_result: GEPAResult,
    *,
    backend: ResultBackend = "gepa",
    run_id: str | None = None,
    budget_limit: int | None = None,
) -> PydanticGEPAResult:
    best_candidate = _best_candidate(raw_result)
    best_score = _best_score(raw_result)
    best_index = _best_candidate_index(raw_result)
    validation_scores = _validation_scores(raw_result)
    candidates = _candidates(raw_result)
    parents = _parent_indices(raw_result)
    history = _normalize_lineage(
        _candidate_history(raw_result),
        candidates=candidates,
        parents=parents,
        best_index=best_index,
    )
    metric_calls = _total_metric_calls(raw_result)
    run_dir = _run_dir(raw_result)
    reported: dict[str, JsonValue] = {
        "best_score": best_score,
        "validation_scores": [cast("JsonValue", score) for score in validation_scores],
        "total_metric_calls": metric_calls,
    }
    return PydanticGEPAResult(
        best_candidate=best_candidate,
        best_score=best_score,
        backend=backend,
        run_id=run_id,
        scores=ScoreSummary(
            search=tuple(validation_scores),
            validation=best_score,
            aggregate=best_score,
        ),
        budget=BudgetSummary(metric_calls=metric_calls, metric_call_limit=budget_limit),
        checkpoints=(run_dir,) if run_dir is not None else (),
        reported=reported,
        derived={
            "candidate_count": len(candidates),
            "best_candidate_fingerprint": best_candidate.fingerprint(),
        },
        best_candidate_index=best_index,
        validation_scores=validation_scores,
        candidate_history=history,
        candidates=candidates,
        parent_indices=parents,
        objective_scores=_objective_scores(raw_result),
        total_metric_calls=metric_calls,
        num_full_val_evals=_num_full_val_evals(raw_result),
        run_dir=run_dir,
        seed=_seed(raw_result),
        per_objective_best_candidates=_per_objective_best_candidates(raw_result),
        objective_pareto_front=_objective_pareto_front(raw_result),
        candidate_tree_dot=_candidate_tree_dot(raw_result),
        candidate_tree_html=_candidate_tree_html(raw_result),
        raw_gepa_result=raw_result,
    )


def _normalize_lineage(
    history: list[CandidateSummary],
    *,
    candidates: list[Candidate],
    parents: list[list[int | None]],
    best_index: int | None,
) -> list[CandidateSummary]:
    values_by_id = {item.candidate_id: item.values for item in history}
    for index, candidate in enumerate(candidates):
        values_by_id.setdefault(f"candidate_{index}", candidate.values)
    normalized: list[CandidateSummary] = []
    for index, item in enumerate(history):
        parent_values: dict[str, str] = {}
        if item.parent_ids:
            parent_values = values_by_id.get(item.parent_ids[0], {})
        elif index < len(parents) and parents[index]:
            parent_index = parents[index][0]
            if parent_index is not None and parent_index < len(candidates):
                parent_values = candidates[parent_index].values
        deltas = tuple(
            CandidateDelta(
                component=name,
                before=parent_values.get(name),
                after=item.values.get(name),
            )
            for name in sorted(set(parent_values) | set(item.values))
            if parent_values.get(name) != item.values.get(name)
        )
        accepted = item.metadata.get("accepted")
        status: CandidateStatus = "best" if index == best_index else "proposed"
        if status != "best" and accepted is True:
            status = "accepted"
        elif status != "best" and accepted is False:
            status = "rejected"
        reason = item.metadata.get("reason")
        feedback = (reason,) if isinstance(reason, str) else ()
        normalized.append(
            item.model_copy(update={"status": status, "feedback": feedback, "deltas": deltas})
        )
    return normalized


def _load_gepa_optimize() -> OptimizeFn:
    try:
        from gepa.api import optimize
    except ImportError as exc:  # pragma: no cover - depends on optional integration
        raise OptimizationDependencyError(
            "GEPA is not installed. Install pydantic-gepa[integrations]."
        ) from exc
    return cast("OptimizeFn", optimize)


def _backend_kwargs(
    optimize_fn: OptimizeFn,
    config: GEPAConfig,
) -> dict[str, SerializableValue]:
    values = config.to_backend_kwargs()
    parameters = signature(optimize_fn).parameters
    if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return values

    unsupported = set(values) - set(parameters)
    defaults = GEPAConfig().to_backend_kwargs()
    configured = sorted(key for key in unsupported if values[key] != defaults[key])
    if configured:
        names = ", ".join(configured)
        raise ConfigurationError(
            f"Installed GEPA backend does not support configured options: {names}."
        )
    return {key: value for key, value in values.items() if key in parameters}


def _best_candidate(raw_result: GEPAResult) -> Candidate:
    if isinstance(raw_result, BestCandidateResult):
        candidate = raw_result.best_candidate
        if isinstance(candidate, Candidate):
            return candidate
        return Candidate.from_gepa_dict(_string_dict(candidate))
    if isinstance(raw_result, BestProgramResult):
        return Candidate.from_gepa_dict(_string_dict(raw_result.best_program))
    return Candidate()


def _best_score(raw_result: GEPAResult) -> float:
    if isinstance(raw_result, BestScoreResult) and isinstance(raw_result.best_score, int | float):
        return float(raw_result.best_score)
    if isinstance(raw_result, BestValScoreResult) and isinstance(
        raw_result.best_val_score, int | float
    ):
        return float(raw_result.best_val_score)
    if (
        isinstance(raw_result, ValAggregateScoresResult)
        and isinstance(raw_result, BestIndexResult)
        and 0 <= raw_result.best_idx < len(raw_result.val_aggregate_scores)
    ):
        return float(raw_result.val_aggregate_scores[raw_result.best_idx])
    return 0.0


def _best_candidate_index(raw_result: GEPAResult) -> int | None:
    if isinstance(raw_result, BestIndexResult):
        return raw_result.best_idx
    return None


def _validation_scores(raw_result: GEPAResult) -> list[float]:
    if isinstance(raw_result, ValidationScoresResult):
        return [float(score) for score in raw_result.validation_scores]
    if isinstance(raw_result, ValAggregateScoresResult):
        return [float(score) for score in raw_result.val_aggregate_scores]
    return []


def _candidates(raw_result: GEPAResult) -> list[Candidate]:
    if not isinstance(raw_result, CandidatesResult):
        return []
    return [
        Candidate.from_gepa_dict(_string_dict(candidate_values), candidate_id=f"candidate_{index}")
        for index, candidate_values in enumerate(raw_result.candidates)
    ]


def _parent_indices(raw_result: GEPAResult) -> list[list[int | None]]:
    if not isinstance(raw_result, ParentsResult):
        return []
    return [list(parent_row) for parent_row in raw_result.parents]


def _candidate_history(raw_result: GEPAResult) -> list[CandidateSummary]:
    if isinstance(raw_result, CandidateHistoryResult):
        history = raw_result.candidate_history
        summaries: list[CandidateSummary] = []
        for index, item in enumerate(history):
            if isinstance(item, CandidateSummary):
                summaries.append(item)
                continue
            summaries.append(
                CandidateSummary(
                    candidate_id=_history_candidate_id(item, index=index),
                    parent_ids=_history_parent_ids(item),
                    generation=_history_generation(item),
                    score=_history_score(item),
                    values=_history_values(item),
                    metadata=_history_metadata(item),
                )
            )
        return summaries
    if not (
        isinstance(raw_result, CandidatesResult)
        and isinstance(raw_result, ParentsResult)
        and isinstance(raw_result, ValAggregateScoresResult)
    ):
        return []

    objective_scores = (
        list(raw_result.val_aggregate_subscores)
        if isinstance(raw_result, ValAggregateSubscoresResult)
        and raw_result.val_aggregate_subscores is not None
        else None
    )
    validation_subscores = (
        list(raw_result.val_subscores) if isinstance(raw_result, ValSubscoresResult) else None
    )
    summaries = []
    for index, candidate_values in enumerate(raw_result.candidates):
        parent_ids = [
            f"candidate_{parent_index}"
            for parent_index in raw_result.parents[index]
            if parent_index is not None
        ]
        summary = CandidateSummary(
            candidate_id=f"candidate_{index}",
            parent_ids=parent_ids,
            generation=None,
            score=float(raw_result.val_aggregate_scores[index]),
            values=_string_dict(candidate_values),
            validation_subscores=(
                {str(key): float(value) for key, value in validation_subscores[index].items()}
                if validation_subscores is not None and index < len(validation_subscores)
                else {}
            ),
            objective_scores=(
                {str(key): float(value) for key, value in objective_scores[index].items()}
                if objective_scores is not None and index < len(objective_scores)
                else {}
            ),
            metadata={"candidate_index": index},
        )
        summaries.append(summary)
    return summaries


def _objective_scores(raw_result: GEPAResult) -> list[dict[str, float]] | None:
    if not isinstance(raw_result, ObjectiveScoresResult):
        if not isinstance(raw_result, ValAggregateSubscoresResult):
            return None
        scores = raw_result.val_aggregate_subscores
        if scores is None:
            return None
    else:
        scores = raw_result.objective_scores
    payload: list[dict[str, float]] = []
    for score_set in scores:
        payload.append({str(key): float(value) for key, value in score_set.items()})
    return payload


def _candidate_tree_dot(raw_result: GEPAResult) -> str | None:
    if isinstance(raw_result, CandidateTreeDotResult):
        dot = raw_result.candidate_tree_dot
        if isinstance(dot, str):
            return dot
        if callable(dot):
            return dot()
    return None


def _candidate_tree_html(raw_result: GEPAResult) -> str | None:
    if isinstance(raw_result, CandidateTreeHtmlResult):
        html = raw_result.candidate_tree_html
        if isinstance(html, str):
            return html
        if callable(html):
            return html()
    return None


def _total_metric_calls(raw_result: GEPAResult) -> int | None:
    if isinstance(raw_result, TotalMetricCallsResult):
        return raw_result.total_metric_calls
    return None


def _num_full_val_evals(raw_result: GEPAResult) -> int | None:
    if isinstance(raw_result, NumFullValEvalsResult):
        return raw_result.num_full_val_evals
    return None


def _run_dir(raw_result: GEPAResult) -> str | None:
    if isinstance(raw_result, RunDirResult):
        return raw_result.run_dir
    return None


def _seed(raw_result: GEPAResult) -> int | None:
    if isinstance(raw_result, SeedResult):
        return raw_result.seed
    return None


def _per_objective_best_candidates(
    raw_result: GEPAResult,
) -> dict[str, list[int]] | None:
    if not isinstance(raw_result, PerObjectiveBestCandidatesResult):
        return None
    per_objective = raw_result.per_objective_best_candidates
    if per_objective is None:
        return None
    return {str(key): list(indices) for key, indices in per_objective.items()}


def _objective_pareto_front(raw_result: GEPAResult) -> dict[str, float] | None:
    if not isinstance(raw_result, ObjectiveParetoFrontResult):
        return None
    pareto_front = raw_result.objective_pareto_front
    if pareto_front is None:
        return None
    return {str(key): float(value) for key, value in pareto_front.items()}


def _history_values(item: HistoryEntry) -> dict[str, str]:
    values: CandidateValues | None = None
    if isinstance(item, HistoryValues):
        values = item.values
    elif isinstance(item, HistoryCandidate):
        values = item.candidate
    if values is None:
        return {}
    return _string_dict(values)


def _history_score(item: HistoryEntry) -> float:
    if isinstance(item, HistoryScore):
        return float(item.score)
    return 0.0


def _history_candidate_id(item: HistoryEntry, *, index: int) -> str:
    if isinstance(item, HistoryCandidateId):
        return item.candidate_id
    return f"candidate_{index}"


def _history_parent_ids(item: HistoryEntry) -> list[str]:
    parent_ids = item.parent_ids if isinstance(item, HistoryParentIds) else []
    return list(parent_ids)


def _history_generation(item: HistoryEntry) -> int | None:
    if isinstance(item, HistoryGeneration):
        return item.generation
    return None


def _history_metadata(item: HistoryEntry) -> dict[str, JsonScalar]:
    if not isinstance(item, HistoryMetadata):
        return {}
    return dict(item.metadata)


def _string_dict(values: CandidateValues) -> dict[str, str]:
    return dict(values)


__all__ = (
    "OptimizeFn",
    "PydanticGEPAOptimizer",
    "result_from_gepa",
)
