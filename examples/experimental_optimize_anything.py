from __future__ import annotations as _annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

from gepa.oa.engine import Result
from support_routing_common import (
    BASELINE_INSTRUCTIONS,
    IMPROVED_INSTRUCTIONS,
    SupportTicket,
    build_agent,
    run_agent,
)

from pydantic_gepa import (
    AgentInstructionsInjection,
    CallableReflectionModel,
    Component,
    DataSplit,
    Example,
    Optimization,
)
from pydantic_gepa.configuration import BudgetConfig, GEPAConfig, ReflectionConfig
from pydantic_gepa.experimental.optimize_anything import (
    BestOf,
    Engine,
    EngineResult,
    EvaluationServer,
    OptimizationTask,
    OptimizeAnythingConfig,
    Pipeline,
    Single,
)


@dataclass(slots=True)
class EvaluatedCandidateEngine:
    """A deterministic custom engine that still uses the real evaluation server."""

    name: str
    candidate: str

    def run(self, task: OptimizationTask, server: EvaluationServer) -> EngineResult:
        del task
        score, evidence = server.evaluate_examples(self.candidate, split="val")
        return cast(
            "EngineResult",
            Result(
                best_candidate=self.candidate,
                best_score=score,
                metadata={"validation_evidence": evidence},
            ),
        )

    def process_result(self, result: EngineResult, output_dir: Path | None) -> None:
        del result, output_dir


def build_optimization() -> Optimization[SupportTicket, str, None]:
    agent = build_agent()
    instructions = Component(name="instructions", initial_text=BASELINE_INSTRUCTIONS)
    data = DataSplit.from_sets(
        train=(
            Example(
                name="train-refund",
                inputs=SupportTicket("I want a refund for order 123."),
                expected_output="refund_request",
            ),
            Example(
                name="train-shipping",
                inputs=SupportTicket("Where is my order?"),
                expected_output="shipping_status",
            ),
            Example(
                name="train-other",
                inputs=SupportTicket("Please resend the invoice PDF."),
                expected_output="other",
            ),
        ),
        validation=(
            Example(
                name="validation-refund",
                inputs=SupportTicket("Can I get my money back?"),
                expected_output="refund_request",
            ),
            Example(
                name="validation-shipping",
                inputs=SupportTicket("The tracking page is empty."),
                expected_output="shipping_status",
            ),
        ),
        test=(
            Example(
                name="heldout-refund",
                inputs=SupportTicket("Refund this purchase, please."),
                expected_output="refund_request",
            ),
        ),
    )
    return Optimization.from_examples(
        data=data,
        task=lambda ticket: run_agent(agent, ticket),
        score=lambda ctx: float(ctx.output == ctx.expected_output),
        components=(instructions,),
        injections=(AgentInstructionsInjection(agent=agent, candidate_component=instructions),),
        backend="optimize_anything",
        optimization_objective="Maximize support-routing accuracy.",
        background="The candidate is the agent instruction component.",
    )


def main() -> None:
    optimization = build_optimization()

    gepa_result = optimization.optimize(
        config=OptimizeAnythingConfig(
            engine=Engine.gepa(
                GEPAConfig(
                    budget=BudgetConfig(max_metric_calls=12),
                    reflection=ReflectionConfig(
                        model=CallableReflectionModel(lambda _prompt: IMPROVED_INSTRUCTIONS)
                    ),
                ),
                candidate_mode="text",
                stop_at_score=1.0,
            ),
            component="instructions",
        )
    )
    print("single GEPA:", gepa_result.best_score, gepa_result.best_candidate.values)

    baseline = Engine.custom(
        EvaluatedCandidateEngine("baseline", BASELINE_INSTRUCTIONS),
        candidate_mode="text",
        max_evals=4,
    )
    explorer = Engine.custom(
        EvaluatedCandidateEngine("explorer", IMPROVED_INSTRUCTIONS),
        candidate_mode="text",
        max_evals=4,
    )
    continuation = Engine.custom(
        EvaluatedCandidateEngine(
            "continuation",
            IMPROVED_INSTRUCTIONS + " Return only the routing label.",
        ),
        candidate_mode="text",
        max_evals=4,
    )
    omni_result = optimization.optimize(
        config=OptimizeAnythingConfig(
            composition=Pipeline(
                steps=(
                    BestOf(engines=(baseline, explorer)),
                    Single(engine=continuation),
                )
            ),
            component="instructions",
        )
    )
    print("Omni pipeline:", omni_result.best_score, omni_result.best_candidate.values)
    print("composition:", omni_result.composition)

    agent_engine = Engine.autoresearch(
        model="claude-sonnet-4-6",
        max_evals=20,
        max_token_cost=5.0,
        sandbox=True,
    )
    print(
        "AutoResearch declaration:",
        OptimizeAnythingConfig(engine=agent_engine, component="instructions").declaration(),
    )


if __name__ == "__main__":
    main()
