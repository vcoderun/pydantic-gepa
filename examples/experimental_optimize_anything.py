from __future__ import annotations as _annotations

# pyright: reportMissingImports=false
from support_routing_common import (
    BASELINE_INSTRUCTIONS,
    IMPROVED_INSTRUCTIONS,
    build_agent,
    build_cases,
    build_dataset,
    run_agent,
)

from pydantic_gepa import (
    AgentInstructionsInjection,
    CallableReflectionModel,
    Candidate,
    CandidateComponent,
    PydanticGEPAAdapter,
    ScoreObjective,
)
from pydantic_gepa.configuration import (
    BudgetConfig,
    GEPAConfig,
    ProgressConfig,
    ReflectionConfig,
)
from pydantic_gepa.experimental.optimize_anything import (
    PydanticOptimizeAnythingAdapter,
    PydanticOptimizeAnythingOptimizer,
)


def main() -> None:
    agent = build_agent()
    component = CandidateComponent(name="instructions", initial_text=BASELINE_INSTRUCTIONS)
    standard = PydanticGEPAAdapter.from_dataset(
        dataset=build_dataset(),
        task=lambda ticket: run_agent(agent, ticket),
        injections=[AgentInstructionsInjection(agent=agent, candidate_component=component)],
        objective=ScoreObjective(score_key="accuracy"),
    )
    optimizer = PydanticOptimizeAnythingOptimizer(
        adapter=PydanticOptimizeAnythingAdapter(adapter=standard),
        initial_candidate=Candidate(values={"instructions": component.initial_value}),
        optimization_objective="Maximize support-routing accuracy.",
        background="The candidate is the agent instruction component.",
    )
    result = optimizer.optimize(
        trainset=build_cases(),
        valset=build_cases(),
        config=GEPAConfig(
            budget=BudgetConfig(max_metric_calls=12),
            reflection=ReflectionConfig(
                model=CallableReflectionModel(lambda prompt: f"```\n{IMPROVED_INSTRUCTIONS}\n```")
            ),
            progress=ProgressConfig(display_bar=True),
        ),
    )
    print(result.stable_dump())


if __name__ == "__main__":
    main()
