from __future__ import annotations as _annotations

from pydantic_gepa import (
    Budget,
    CallableReflectionModel,
    CandidateComponent,
    Component,
    GEPAConfig,
    Optimization,
    OptimizationResult,
    Plan,
    PydanticGEPAOptimization,
    PydanticGEPAResult,
    RunConfig,
    Stage,
)


def test_common_names_preserve_compatibility_types() -> None:
    assert Component is CandidateComponent
    assert Optimization is PydanticGEPAOptimization
    assert OptimizationResult is PydanticGEPAResult
    assert Optimization.run is PydanticGEPAOptimization.optimize
    assert Budget(max_metric_calls=3).max_metric_calls == 3
    assert Plan.__name__ == "Plan"
    assert Stage.__name__ == "Stage"
    assert GEPAConfig().budget.max_metric_calls == 50
    assert RunConfig().seed == 0
    assert CallableReflectionModel(lambda prompt: str(prompt))("prompt") == "prompt"
