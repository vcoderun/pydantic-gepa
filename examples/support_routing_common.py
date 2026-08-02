from __future__ import annotations as _annotations

# pyright: reportMissingImports=false
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EqualsExpected, Evaluator, EvaluatorContext

from pydantic_gepa.values import SerializableValue

BASELINE_INSTRUCTIONS = "Always return other."
IMPROVED_INSTRUCTIONS = (
    "Return refund_request for refund or money back requests. "
    "Return shipping_status for order tracking or delivery questions. "
    "Return other otherwise."
)

TicketMetadata = dict[str, str]


@dataclass(frozen=True)
class SupportTicket:
    customer_message: str


class AccuracyScore(Evaluator[SupportTicket, str, TicketMetadata]):
    evaluation_name = "accuracy"

    def evaluate(self, ctx: EvaluatorContext[SupportTicket, str, TicketMetadata]) -> float:
        return 1.0 if ctx.output == ctx.expected_output else 0.0


def build_cases() -> list[Case[SupportTicket, str, TicketMetadata]]:
    return [
        Case(
            name="refund",
            inputs=SupportTicket("I want a refund for order 123."),
            expected_output="refund_request",
            metadata={"slice": "refunds"},
        ),
        Case(
            name="shipping",
            inputs=SupportTicket("Where is my order? The tracking page is empty."),
            expected_output="shipping_status",
            metadata={"slice": "fulfillment"},
        ),
        Case(
            name="other",
            inputs=SupportTicket("Please resend the invoice PDF."),
            expected_output="other",
            metadata={"slice": "general"},
        ),
    ]


def build_dataset() -> Dataset[SupportTicket, str, TicketMetadata]:
    return Dataset(
        name="support-routing",
        cases=[],
        evaluators=[AccuracyScore(), EqualsExpected()],
    )


def build_agent() -> Agent[None, str]:
    return Agent(
        FunctionModel(route_support_tickets),
        output_type=str,
        instructions=BASELINE_INSTRUCTIONS,
    )


def route_support_tickets(messages: list[ModelMessage], agent_info: AgentInfo) -> ModelResponse:
    ticket_text = latest_user_prompt(messages).lower()
    instructions = (agent_info.instructions or "").lower()

    if ("refund_request" in instructions or "refund" in instructions) and (
        "refund" in ticket_text or "money back" in ticket_text
    ):
        label = "refund_request"
    elif (
        "shipping_status" in instructions
        or "shipping" in instructions
        or "tracking" in instructions
    ) and ("where is my order" in ticket_text or "tracking" in ticket_text):
        label = "shipping_status"
    else:
        label = "other"

    return ModelResponse(parts=[TextPart(label)])


def latest_user_prompt(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        for part in reversed(message.parts):
            if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                return part.content
    raise ValueError("No user prompt found in model messages.")


def run_agent(agent: Agent[None, str], ticket: SupportTicket) -> str:
    return agent.run_sync(ticket.customer_message).output


def build_reflective_proposer(component_encoder: Callable[[str], str]):
    def propose_new_texts(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, SerializableValue]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if "instructions" not in components_to_update:
            return candidate

        records = reflective_dataset.get("instructions", ())
        saw_refund_case = any(
            "refund" in extract_customer_message(record)
            or "money back" in extract_customer_message(record)
            for record in records
        )
        saw_shipping_case = any(
            "where is my order" in extract_customer_message(record)
            or "tracking" in extract_customer_message(record)
            for record in records
        )

        instructions: list[str] = []
        if saw_refund_case:
            instructions.append("Return refund_request for refund or money back requests.")
        if saw_shipping_case:
            instructions.append("Return shipping_status for order tracking or delivery questions.")
        instructions.append("Return other otherwise.")

        return {"instructions": component_encoder(" ".join(instructions))}

    return propose_new_texts


def extract_customer_message(record: Mapping[str, SerializableValue]) -> str:
    inputs = record.get("inputs")
    if isinstance(inputs, dict):
        typed_inputs = cast("dict[str, SerializableValue]", inputs)
        value = typed_inputs.get("customer_message")
        if isinstance(value, str):
            return value.lower()
    return ""
