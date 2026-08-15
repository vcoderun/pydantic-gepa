from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

from ..evaluation.evidence import Attachment, AttachmentKind, Encoder, EvidenceValue
from ..reflection import (
    CallableReflectionModel,
    ReflectionFailure,
    ReflectionPrompt,
    ReflectionRecord,
    ReflectionResponse,
    ReflectionUsage,
)
from ..values import JsonValue

if TYPE_CHECKING:
    from pydantic_ai import Agent, BinaryContent
    from pydantic_ai.models import KnownModelName, Model
    from pydantic_ai.settings import ModelSettings

DepsT = TypeVar("DepsT")


@dataclass(slots=True)
class PydanticAIReflectionModel(Generic[DepsT]):
    agent: Agent[DepsT, str]
    deps: DepsT
    model_settings: ModelSettings | None = None
    max_output_tokens: int | None = None
    timeout: float | None = None
    retries: int = 0
    on_error: ReflectionFailure = "raise"
    _reflection: CallableReflectionModel = field(init=False, repr=False)

    def __post_init__(self) -> None:
        settings: ModelSettings = {} if self.model_settings is None else self.model_settings.copy()
        if self.max_output_tokens is not None:
            settings["max_tokens"] = self.max_output_tokens
        if self.timeout is not None:
            settings["timeout"] = self.timeout

        def call(prompt: ReflectionPrompt) -> ReflectionResponse:
            result = self.agent.run_sync(
                _prompt_text(prompt),
                deps=self.deps,
                model_settings=settings or None,
            )
            usage_or_getter = result.usage
            usage = usage_or_getter() if callable(usage_or_getter) else usage_or_getter
            details: dict[str, JsonValue] = dict(usage.details)
            return ReflectionResponse(
                text=result.output,
                usage=ReflectionUsage(
                    requests=usage.requests,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                ),
                metadata={"tool_calls": usage.tool_calls, "details": details},
            )

        self._reflection = CallableReflectionModel(
            call,
            retries=self.retries,
            on_error=self.on_error,
        )

    @classmethod
    def from_model(
        cls,
        model: Model | KnownModelName | str,
        *,
        model_settings: ModelSettings | None = None,
        max_output_tokens: int | None = None,
        timeout: float | None = None,
        retries: int = 0,
        on_error: ReflectionFailure = "raise",
    ) -> PydanticAIReflectionModel[None]:
        from pydantic_ai import Agent

        return PydanticAIReflectionModel(
            agent=Agent(model, output_type=str),
            deps=None,
            model_settings=model_settings,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            retries=retries,
            on_error=on_error,
        )

    @property
    def records(self) -> tuple[ReflectionRecord, ...]:
        return self._reflection.records

    @property
    def total_cost(self) -> float:
        return self._reflection.total_cost

    @property
    def total_tokens_in(self) -> int:
        return self._reflection.total_tokens_in

    @property
    def total_tokens_out(self) -> int:
        return self._reflection.total_tokens_out

    def __call__(self, prompt: ReflectionPrompt) -> str:
        return self._reflection(prompt)


def _prompt_text(prompt: ReflectionPrompt) -> str:
    if isinstance(prompt, str):
        return prompt
    return "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}" for message in prompt
    )


def register_evidence(encoder: Encoder) -> Encoder:
    from pydantic_ai import AudioUrl, BinaryContent, DocumentUrl, ImageUrl, VideoUrl

    encoder.register(
        ImageUrl,
        lambda value: Attachment(
            kind="image",
            reference=value.url,
            media_type=value.media_type,
        ).as_evidence(),
    )
    encoder.register(
        AudioUrl,
        lambda value: Attachment(
            kind="audio",
            reference=value.url,
            media_type=value.media_type,
        ).as_evidence(),
    )
    encoder.register(
        VideoUrl,
        lambda value: Attachment(
            kind="video",
            reference=value.url,
            media_type=value.media_type,
        ).as_evidence(),
    )
    encoder.register(
        DocumentUrl,
        lambda value: Attachment(
            kind="document",
            reference=value.url,
            media_type=value.media_type,
        ).as_evidence(),
    )
    encoder.register(BinaryContent, _binary_evidence)
    return encoder


def evidence_encoder(
    *,
    max_depth: int = 8,
    max_items: int = 100,
    max_string_length: int = 4_000,
) -> Encoder:
    return register_evidence(
        Encoder(
            max_depth=max_depth,
            max_items=max_items,
            max_string_length=max_string_length,
        )
    )


def _binary_evidence(content: BinaryContent) -> dict[str, EvidenceValue]:
    category = content.media_type.partition("/")[0]
    if category == "image":
        kind: AttachmentKind = "image"
    elif category == "audio":
        kind = "audio"
    elif category == "video":
        kind = "video"
    elif category in {"application", "text"}:
        kind = "document"
    else:
        kind = "binary"
    return Attachment.from_bytes(
        content.data,
        kind=kind,
        media_type=content.media_type,
    ).as_evidence()


__all__ = (
    "PydanticAIReflectionModel",
    "evidence_encoder",
    "register_evidence",
)
