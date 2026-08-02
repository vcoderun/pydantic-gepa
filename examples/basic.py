from __future__ import annotations as _annotations

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
    Example,
    optimize,
)
from pydantic_gepa.configuration import BudgetConfig, GEPAConfig, ReflectionConfig


def main() -> None:
    agent = build_agent()
    instructions = Component(name="instructions", initial_text=BASELINE_INSTRUCTIONS)
    examples = [
        Example(
            name="refund",
            inputs=SupportTicket("I want a refund for order 123."),
            expected_output="refund_request",
        ),
        Example(
            name="shipping",
            inputs=SupportTicket("Where is my order? The tracking page is empty."),
            expected_output="shipping_status",
        ),
        Example(
            name="other",
            inputs=SupportTicket("Please resend the invoice PDF."),
            expected_output="other",
        ),
    ]

    result = optimize(
        train=examples,
        validation=examples,
        task=lambda ticket: run_agent(agent, ticket),
        score=lambda ctx: float(ctx.output == ctx.expected_output),
        components=[instructions],
        injections=[AgentInstructionsInjection(agent=agent, candidate_component=instructions)],
        config=GEPAConfig(
            budget=BudgetConfig(max_metric_calls=12),
            reflection=ReflectionConfig(
                model=CallableReflectionModel(lambda _prompt: IMPROVED_INSTRUCTIONS)
            ),
        ),
    )

    print("best candidate:", result.best_candidate.values)
    print("best score:", result.best_score)


if __name__ == "__main__":
    main()
