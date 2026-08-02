# Multimodal And Binary Evidence

`Example.attachments` carries typed references for image, audio, video,
document, and generic binary inputs. `Attachment.from_bytes(...)` stores a
digest and size rather than inserting a base64 payload into reflection data.

`Encoder` handles Pydantic models, dataclasses, mappings, sequences, paths,
dates, bounded strings, bytes, and attachments. Custom encoders can be
registered for application types. Cycles and depth/item limits produce
deterministic bounded evidence.

Pydantic AI media references are adapted lazily by the integration module, so
core imports do not require Pydantic AI.
