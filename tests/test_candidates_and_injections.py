from __future__ import annotations as _annotations

from contextlib import contextmanager
from typing import Any

import pytest

from pydantic_gepa import (
    AgentInstructionsInjection,
    Candidate,
    CandidateComponent,
    CandidateComponentError,
    CandidateContext,
    CandidateInjectionError,
    DerivedValueInjection,
    InstructionsCandidate,
    NoopInjection,
)


def test_candidate_roundtrips_to_gepa_dict_without_aliasing() -> None:
    candidate = Candidate.from_gepa_dict(
        {"instructions": "hello"},
        candidate_id="candidate_1",
        parent_id="candidate_0",
        generation=1,
        metadata={"source": "test"},
    )
    gepa_dict = candidate.to_gepa_dict()
    gepa_dict["instructions"] = "changed"

    assert candidate.values == {"instructions": "hello"}
    assert candidate.id == "candidate_1"
    assert candidate.parent_id == "candidate_0"
    assert candidate.generation == 1
    assert candidate.metadata == {"source": "test"}


def test_candidate_fingerprint_depends_only_on_canonical_component_values() -> None:
    first = Candidate(
        id="candidate-a",
        values={"writer": "two", "planner": "one"},
        metadata={"source": "first"},
    )
    equivalent = Candidate(
        id="candidate-b",
        values={"planner": "one", "writer": "two"},
        metadata={"source": "second"},
    )
    changed = Candidate(values={"planner": "one", "writer": "changed"})

    assert first.fingerprint() == equivalent.fingerprint()
    assert first.fingerprint() != changed.fingerprint()


def test_candidate_components_encode_and_decode_raw_and_json_string_values() -> None:
    json_component = CandidateComponent(
        name="instructions",
        initial_text="hello",
        serialization="json_string",
    )
    raw_component = CandidateComponent(
        name="system_prompt",
        initial_text="raw prompt",
        asset_ref="prompt:system@v1",
        injection_target="agent.instructions",
        coupled_components=("router.instructions",),
    )

    assert json_component.initial_value == '"hello"'
    assert json_component.decode('"hello"') == "hello"
    assert raw_component.initial_value == "raw prompt"
    assert raw_component.decode("raw prompt") == "raw prompt"
    assert raw_component.asset_ref == "prompt:system@v1"
    assert raw_component.injection_target == "agent.instructions"
    assert raw_component.coupled_components == ("router.instructions",)


def test_candidate_component_rejects_bad_json_and_non_string_json() -> None:
    component = CandidateComponent(
        name="instructions",
        initial_text="",
        serialization="json_string",
    )

    with pytest.raises(CandidateComponentError, match="not valid JSON"):
        component.decode("{")
    with pytest.raises(CandidateComponentError, match="must decode to a string"):
        component.decode("42")


def test_instructions_candidate_builds_default_candidate() -> None:
    candidate = InstructionsCandidate(instructions="Extract fields.").to_candidate()

    assert candidate.to_gepa_dict() == {"instructions": "Extract fields."}


def test_agent_instructions_injection_applies_agent_override() -> None:
    agent = _FakeAgent()
    injection = AgentInstructionsInjection(agent=agent)

    with injection.apply({"instructions": "Be precise."}):
        assert agent.active_instructions == "Be precise."

    assert agent.active_instructions is None
    assert agent.seen_instructions == ["Be precise."]


def test_agent_instructions_injection_reports_missing_or_bad_targets() -> None:
    class NoOverrideAgent:
        pass

    with pytest.raises(CandidateInjectionError, match="missing component"):
        AgentInstructionsInjection(agent=_FakeAgent()).apply({})

    with pytest.raises(CandidateInjectionError, match="override"):
        AgentInstructionsInjection.model_construct(agent=NoOverrideAgent()).apply(
            {"instructions": "x"}
        )

    with pytest.raises(CandidateInjectionError, match="context manager"):
        AgentInstructionsInjection.model_construct(agent=_BadAgent()).apply({"instructions": "x"})


def test_agent_instructions_injection_can_use_custom_component() -> None:
    agent = _FakeAgent()
    component = CandidateComponent(name="prompt", initial_text="")
    injection = AgentInstructionsInjection(
        agent=agent,
        component="instructions",
        candidate_component=component,
    )

    with injection.apply({"instructions": "Use custom component."}):
        assert agent.active_instructions == "Use custom component."

    assert injection.candidate_component.name == "instructions"


def test_candidate_context_stores_values_and_resets_after_scope() -> None:
    context = CandidateContext[str](name="output_schema")

    assert context.get() is None
    with pytest.raises(CandidateInjectionError, match="no active value"):
        context.require()

    with context.use("optimized"):
        assert context.get() == "optimized"
        assert context.require() == "optimized"

    assert context.get() is None


def test_derived_value_injection_stores_derived_value_and_validates_requirements() -> None:
    context = CandidateContext[int](name="candidate_length")
    injection = DerivedValueInjection[int](
        component="output_schema",
        context=context,
        derive_value=lambda candidate: len(candidate["instructions"]),
        required_components=("instructions",),
    )

    with injection.apply({"instructions": "prompt"}):
        assert context.require() == 6

    assert context.get() is None
    with pytest.raises(CandidateInjectionError, match="required components"):
        injection.apply({})


def test_noop_injection_validates_component_presence() -> None:
    injection = NoopInjection(component="instructions")

    with injection.apply({"instructions": "raw"}):
        pass
    with pytest.raises(CandidateInjectionError, match="missing component"):
        injection.apply({})


class _FakeAgent:
    def __init__(self) -> None:
        self.active_instructions: str | None = None
        self.seen_instructions: list[str] = []

    @contextmanager
    def override(self, *, instructions: str) -> Any:
        self.active_instructions = instructions
        self.seen_instructions.append(instructions)
        try:
            yield
        finally:
            self.active_instructions = None


class _BadAgent:
    def override(self, *, instructions: str) -> str:
        return instructions
