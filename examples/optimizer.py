from __future__ import annotations as _annotations

# pyright: reportMissingImports=false
from support_routing_common import (
    BASELINE_INSTRUCTIONS,
    build_agent,
    build_cases,
    build_dataset,
    build_reflective_proposer,
    run_agent,
)

from pydantic_gepa import (
    AgentInstructionsInjection,
    Candidate,
    CandidateComponent,
    PydanticGEPAAdapter,
    PydanticGEPAOptimizer,
    ScoreObjective,
)


def main() -> None:
    agent = build_agent()
    component = CandidateComponent(name="instructions", initial_text=BASELINE_INSTRUCTIONS)
    adapter = PydanticGEPAAdapter.from_dataset(
        dataset=build_dataset(),
        task=lambda ticket: run_agent(agent, ticket),
        injections=[AgentInstructionsInjection(agent=agent, candidate_component=component)],
        objective=ScoreObjective(score_key="accuracy"),
        propose_new_texts=build_reflective_proposer(component.encode),
    )
    optimizer = PydanticGEPAOptimizer(
        adapter=adapter,
        initial_candidate=Candidate(values={"instructions": component.initial_value}),
    )
    result = optimizer.optimize(
        trainset=build_cases(),
        valset=build_cases(),
        max_metric_calls=12,
        reflection_minibatch_size=3,
        module_selector="all",
    )

    print("best candidate:", result.best_candidate.values)
    print("best score:", result.best_score)
    print("validation scores:", result.validation_scores)
    print("history:")
    for item in result.candidate_history:
        print(item)


if __name__ == "__main__":
    main()
