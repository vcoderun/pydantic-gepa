from __future__ import annotations as _annotations

import sys
from dataclasses import dataclass
from types import ModuleType

from pytest import MonkeyPatch

from pydantic_gepa import Attachment
from pydantic_gepa.evaluation import AttachmentKind, Encoder
from pydantic_gepa.integrations.pydantic_ai import evidence_encoder, register_evidence


def test_pydantic_ai_url_references_encode_as_attachments(monkeypatch: MonkeyPatch) -> None:
    _install_fake_pydantic_ai(monkeypatch)
    encoder = evidence_encoder(max_depth=2, max_items=3, max_string_length=10)

    assert encoder.encode(_ImageUrl("image.png", "image/png")) == {
        "type": "attachment",
        "kind": "image",
        "reference": "image.png",
        "media_type": "image/png",
        "size_bytes": None,
        "digest": None,
    }
    assert (
        encoder.encode(_AudioUrl("audio.mp3", "audio/mpeg"))
        == Attachment(
            kind="audio",
            reference="audio.mp3",
            media_type="audio/mpeg",
        ).as_evidence()
    )
    assert (
        encoder.encode(_VideoUrl("video.mp4", "video/mp4"))
        == Attachment(
            kind="video",
            reference="video.mp4",
            media_type="video/mp4",
        ).as_evidence()
    )
    assert (
        encoder.encode(_DocumentUrl("report.pdf", "application/pdf"))
        == Attachment(
            kind="document",
            reference="report.pdf",
            media_type="application/pdf",
        ).as_evidence()
    )
    assert encoder.max_depth == 2
    assert encoder.max_items == 3
    assert encoder.max_string_length == 10


def test_pydantic_ai_binary_content_is_summarized_without_payload(
    monkeypatch: MonkeyPatch,
) -> None:
    _install_fake_pydantic_ai(monkeypatch)
    encoder = register_evidence(Encoder())

    cases: list[tuple[str, AttachmentKind]] = [
        ("image/png", "image"),
        ("audio/mpeg", "audio"),
        ("video/mp4", "video"),
        ("application/pdf", "document"),
        ("text/plain", "document"),
        ("chemical/x-pdb", "binary"),
    ]
    for media_type, kind in cases:
        evidence = encoder.encode(_BinaryContent(b"private-content", media_type))
        assert evidence == {
            **Attachment.from_bytes(
                b"private-content",
                kind=kind,
                media_type=media_type,
            ).as_evidence()
        }
        assert "private-content" not in str(evidence)


@dataclass(frozen=True, slots=True)
class _ImageUrl:
    url: str
    media_type: str


@dataclass(frozen=True, slots=True)
class _AudioUrl:
    url: str
    media_type: str


@dataclass(frozen=True, slots=True)
class _VideoUrl:
    url: str
    media_type: str


@dataclass(frozen=True, slots=True)
class _DocumentUrl:
    url: str
    media_type: str


@dataclass(frozen=True, slots=True)
class _BinaryContent:
    data: bytes
    media_type: str


def _install_fake_pydantic_ai(monkeypatch: MonkeyPatch) -> None:
    module = ModuleType("pydantic_ai")
    module.__dict__.update(
        {
            "ImageUrl": _ImageUrl,
            "AudioUrl": _AudioUrl,
            "VideoUrl": _VideoUrl,
            "DocumentUrl": _DocumentUrl,
            "BinaryContent": _BinaryContent,
        }
    )
    monkeypatch.setitem(sys.modules, "pydantic_ai", module)
