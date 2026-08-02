from __future__ import annotations as _annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import IntEnum, StrEnum
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from pydantic_gepa import (
    Attachment,
    Candidate,
    Context,
    EvidenceEncodingError,
    Example,
    Runtime,
)
from pydantic_gepa.evaluation import Encoder


def test_attachment_summarizes_binary_without_embedding_content() -> None:
    attachment = Attachment.from_bytes(
        b"image-content",
        kind="image",
        reference="images/customer.png",
        media_type="image/png",
    )

    evidence = attachment.as_evidence()
    assert evidence == {
        "type": "attachment",
        "kind": "image",
        "reference": "images/customer.png",
        "media_type": "image/png",
        "size_bytes": 13,
        "digest": "d2dfc251c1a7245d4eb7d95e5f815472c6dbcf7ee6690bbd7c1912f477b6c22a",
    }
    assert "image-content" not in str(evidence)

    with pytest.raises(ValidationError):
        Attachment(size_bytes=-1)


def test_encoder_handles_scalar_and_standard_values() -> None:
    encoder = Encoder(max_string_length=4)

    assert encoder.encode(None) is None
    assert encoder.encode(True) is True
    assert encoder.encode(3) == 3
    assert encoder.encode(1.5) == 1.5
    assert encoder.encode(math.nan) == {"type": "float", "value": "nan"}
    assert encoder.encode(math.inf) == {"type": "float", "value": "inf"}
    assert encoder.encode("abcdef") == "abcd...[2 chars omitted]"
    assert encoder.encode(Path("artifact.txt")) == {"type": "path", "path": "artifact.txt"}
    assert encoder.encode(date(2026, 8, 1)) == "2026-08-01"
    assert encoder.encode(datetime(2026, 8, 1, tzinfo=UTC)) == ("2026-08-01T00:00:00+00:00")
    assert encoder.encode(_Status.READY) == "ready"
    assert encoder.encode(_Priority.HIGH) == 2
    attachment = Attachment(kind="image", reference="image.png")
    assert encoder.encode(attachment) == attachment.as_evidence()


def test_encoder_handles_models_dataclasses_mappings_sequences_and_binary() -> None:
    encoder = Encoder()

    assert encoder.encode(_Model(name="Ada", count=2)) == {"count": 2, "name": "Ada"}
    assert encoder.encode(_Record(name="Ada", values=(1, 2))) == {
        "name": "Ada",
        "values": [1, 2],
    }
    assert encoder.encode({"b": 2, "a": 1}) == {"a": 1, "b": 2}
    assert encoder.encode([1, "two"]) == [1, "two"]
    assert encoder.encode(bytearray(b"abc")) == {
        "type": "attachment",
        "kind": "binary",
        "reference": None,
        "media_type": None,
        "size_bytes": 3,
        "digest": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }


def test_encoder_bounds_nested_and_collection_evidence() -> None:
    encoder = Encoder(max_depth=2, max_items=2)

    assert encoder.encode([1, 2, 3]) == {
        "type": "truncated_sequence",
        "items": [1, 2],
        "omitted": 1,
    }
    assert encoder.encode({"a": 1, "b": 2, "c": 3}) == {
        "type": "truncated_mapping",
        "items": {"a": 1, "b": 2},
        "omitted": 1,
    }
    assert encoder.encode({"outer": {"inner": {"value": 1}}}) == {
        "outer": {"inner": {"type": "truncated", "reason": "max_depth"}}
    }

    node = _Node(name="root")
    node.child = node
    assert encoder.encode(node) == {
        "child": {"type": "cycle", "python_type": "_Node"},
        "name": "root",
    }


def test_encoder_supports_exact_custom_encoders_and_rejects_unsafe_values() -> None:
    encoder = Encoder()
    encoder.register(_CustomValue, lambda value: {"custom": value.value})

    assert encoder.encode(_CustomValue("ok")) == {"custom": "ok"}
    with pytest.raises(EvidenceEncodingError, match="string keys"):
        encoder.encode({1: "invalid"})
    with pytest.raises(EvidenceEncodingError, match="No evidence encoder"):
        encoder.encode(_Unsupported())

    broken_mapping_encoder = Encoder()
    broken_mapping_encoder.register(dict, lambda _: "not-a-mapping")
    with pytest.raises(EvidenceEncodingError, match="did not encode to a mapping"):
        broken_mapping_encoder.mapping({"key": "value"})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_depth": 0}, "max_depth"),
        ({"max_items": 0}, "max_items"),
        ({"max_string_length": 0}, "max_string_length"),
    ],
)
def test_encoder_validates_bounds(kwargs: dict[str, int], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Encoder(**kwargs)


@pytest.mark.asyncio
async def test_context_runs_subject_repeatedly_and_records_invocations() -> None:
    async def run(value: str) -> str:
        await asyncio.sleep(0)
        return value.upper()

    context = Context(
        runtime=Runtime(run),
        example=Example(name="case-one", inputs="first", expected_output="FIRST"),
        candidate=Candidate(values={"prompt": "candidate"}),
        run_id="run-one",
        stage_id="stage-one",
        case_id="case-one",
        active_components=("prompt",),
        frozen_components=("tool",),
    )

    assert await context.arun() == "FIRST"
    assert await context.arun_with("second") == "SECOND"
    assert context.output == "SECOND"
    assert context.case_id == "case-one"
    assert [invocation.inputs for invocation in context.invocations] == ["first", "second"]
    assert [invocation.output for invocation in context.invocations] == ["FIRST", "SECOND"]
    assert all(invocation.error is None for invocation in context.invocations)
    assert all(invocation.duration_seconds >= 0 for invocation in context.invocations)


def test_context_sync_entrypoints_and_generated_case_id() -> None:
    def double(value: int) -> int:
        return value * 2

    context = Context(
        runtime=Runtime(double),
        example=Example(inputs=2),
        candidate=Candidate(),
    )

    assert context.output is None
    assert context.run() == 4
    assert context.run_with(3) == 6
    assert context.case_id is not None
    assert len(context.case_id) == 32

    with pytest.raises(ValueError, match="attempt"):
        Context(
            runtime=Runtime(lambda value: value),
            example=Example(inputs=1),
            candidate=Candidate(),
            attempt=0,
        )


@pytest.mark.asyncio
async def test_context_records_failures_and_cancellation() -> None:
    async def fail(_: str) -> str:
        raise RuntimeError("subject failed")

    failed = Context(
        runtime=Runtime(fail),
        example=Example(inputs="input"),
        candidate=Candidate(),
    )
    with pytest.raises(RuntimeError, match="subject failed"):
        await failed.arun()
    assert failed.invocations[0].output is None
    assert failed.invocations[0].error is not None
    assert failed.invocations[0].error.kind == "RuntimeError"

    entered = asyncio.Event()
    release = asyncio.Event()

    async def wait(_: str) -> str:
        entered.set()
        await release.wait()
        return "released"

    cancelled = Context(
        runtime=Runtime(wait),
        example=Example(inputs="input"),
        candidate=Candidate(),
    )
    task = asyncio.create_task(cancelled.arun())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.invocations[0].error is not None
    assert cancelled.invocations[0].error.kind == "CancelledError"
    release.set()
    assert await wait("input") == "released"


def test_context_records_bounded_component_traces() -> None:
    context = Context(
        runtime=Runtime(lambda value: value),
        example=Example(
            inputs="input",
            attachments=(Attachment(kind="image", reference="image://customer"),),
        ),
        candidate=Candidate(),
        encoder=Encoder(max_string_length=5),
    )
    error = ValueError("invalid tool output")
    parent = context.record_trace(
        "planner",
        input={"request": "customer-data"},
        output={"plan": "extract"},
        metadata={"attempt": 1},
        duration_seconds=0.5,
        trace_id="trace-parent",
    )
    child = context.record_trace(
        "tool",
        input=b"binary",
        error=error,
        parent_id=parent.id,
        trace_id="trace-child",
    )

    assert context.traces == (parent, child)
    assert parent.input == {"request": "custo...[8 chars omitted]"}
    assert child.parent_id == "trace-parent"
    assert child.error is not None
    assert child.error.message == "invalid tool output"
    assert child.input == Attachment.from_bytes(b"binary").as_evidence()
    assert context.example.attachments[0].reference == "image://customer"


def test_context_captures_sync_components_and_artifacts() -> None:
    context = Context(
        runtime=Runtime(lambda value: value),
        example=Example(inputs="input"),
        candidate=Candidate(),
    )
    artifact = Attachment(kind="document", reference="artifacts/result.json")

    assert context.artifact(artifact) is artifact
    assert context.artifacts == (artifact,)
    assert context.capture(
        "parser",
        lambda: {"answer": "Ada"},
        kind="output_parser",
        input={"text": "name=Ada"},
        metadata={"format": "kv"},
        parent_id="agent-trace",
    ) == {"answer": "Ada"}
    trace = context.traces[0]
    assert trace.component == "parser"
    assert trace.kind == "output_parser"
    assert trace.parent_id == "agent-trace"
    assert trace.duration_seconds is not None

    def fail() -> str:
        raise ValueError("parse failed")

    with pytest.raises(ValueError, match="parse failed"):
        context.capture("parser", fail)
    assert context.traces[1].error is not None
    assert context.traces[1].error.kind == "ValueError"


@pytest.mark.asyncio
async def test_context_captures_async_components_failures_and_cancellation() -> None:
    context = Context(
        runtime=Runtime(lambda value: value),
        example=Example(inputs="input"),
        candidate=Candidate(),
    )

    async def succeed() -> str:
        await asyncio.sleep(0)
        return "done"

    async def fail() -> str:
        await asyncio.sleep(0)
        raise RuntimeError("tool failed")

    assert await context.acapture("tool", succeed, kind="tool_call") == "done"
    with pytest.raises(RuntimeError, match="tool failed"):
        await context.acapture("tool", fail)

    entered = asyncio.Event()
    release = asyncio.Event()

    async def wait() -> str:
        entered.set()
        await release.wait()
        return "released"

    task = asyncio.create_task(context.acapture("tool", wait))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await wait() == "released"

    assert [trace.error.kind if trace.error is not None else None for trace in context.traces] == [
        None,
        "RuntimeError",
        "CancelledError",
    ]


class _Status(StrEnum):
    READY = "ready"


class _Priority(IntEnum):
    LOW = 1
    HIGH = 2


class _Model(BaseModel):
    name: str
    count: int


@dataclass(frozen=True, slots=True)
class _Record:
    name: str
    values: tuple[int, ...]


@dataclass(slots=True)
class _Node:
    name: str
    child: _Node | None = None


@dataclass(frozen=True, slots=True)
class _CustomValue:
    value: str


class _Unsupported:
    pass
