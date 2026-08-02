from __future__ import annotations as _annotations

# pyright: reportMissingImports=false
from collections.abc import Mapping, Sequence
from typing import TypedDict

from pydantic_evals import Case
from support_routing_common import (
    BASELINE_INSTRUCTIONS,
    IMPROVED_INSTRUCTIONS,
    SupportTicket,
    TicketMetadata,
    build_agent,
    build_cases,
    build_dataset,
    run_agent,
)

from pydantic_gepa import (
    AgentInstructionsInjection,
    CandidateComponent,
    PydanticEvalTrajectory,
    PydanticGEPAAdapter,
    ScoreObjective,
)
from pydantic_gepa.adapter import ReportCases, ReportEnvelope, ReportFailures


class BatchRecord(TypedDict):
    candidate: dict[str, str]
    batch_size: int
    case_count: int
    failure_count: int
    scores: list[float]
    score_sum: float
    trace_count: int


class InMemoryRecorder:
    def __init__(self) -> None:
        self.records: list[BatchRecord] = []

    def record_candidate_batch(
        self,
        *,
        candidate: Mapping[str, str],
        batch: Sequence[Case[SupportTicket, str, TicketMetadata]],
        report: ReportEnvelope,
        scores: Sequence[float],
        trajectories: Sequence[PydanticEvalTrajectory] | None,
    ) -> None:
        case_count = len(report.cases) if isinstance(report, ReportCases) else 0
        failure_count = len(report.failures) if isinstance(report, ReportFailures) else 0
        self.records.append(
            {
                "candidate": dict(candidate),
                "batch_size": len(batch),
                "case_count": case_count,
                "failure_count": failure_count,
                "scores": list(scores),
                "score_sum": sum(scores),
                "trace_count": 0 if trajectories is None else len(trajectories),
            }
        )


def main() -> None:
    recorder = InMemoryRecorder()
    agent = build_agent()
    component = CandidateComponent(name="instructions", initial_text=BASELINE_INSTRUCTIONS)
    adapter = PydanticGEPAAdapter.from_dataset(
        dataset=build_dataset(),
        task=lambda ticket: run_agent(agent, ticket),
        injections=[AgentInstructionsInjection(agent=agent, candidate_component=component)],
        objective=ScoreObjective(score_key="accuracy"),
        recorder=recorder,
    )

    adapter.evaluate(
        build_cases(),
        {"instructions": component.encode(IMPROVED_INSTRUCTIONS)},
        capture_traces=True,
    )

    print("recorded batch:", recorder.records[0])


if __name__ == "__main__":
    main()
