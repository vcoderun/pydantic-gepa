from __future__ import annotations as _annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

from .candidates import Candidate
from .configuration import GEPAConfig
from .evaluation import CaseResult
from .events import Event
from .orchestration import PlanResult, PlanSpec, StageResult
from .results import PydanticGEPAResult
from .values import JsonValue

CONTRACT_VERSION = "1.0"


class ContractSchemaGenerator(GenerateJsonSchema):
    def callable_schema(self, schema: core_schema.CallableSchema) -> JsonSchemaValue:
        return {"type": "string", "format": "python-callable-reference"}

    def is_instance_schema(self, schema: core_schema.IsInstanceSchema) -> JsonSchemaValue:
        return {"type": "string", "format": "python-object-reference"}


def contract_schemas() -> dict[str, dict[str, Any]]:
    models = {
        "candidate": Candidate,
        "case_result": CaseResult[JsonValue],
        "config": GEPAConfig,
        "event": TypeAdapter(Event),
        "plan": PlanSpec,
        "plan_result": PlanResult,
        "result": PydanticGEPAResult,
        "stage_result": StageResult,
    }
    return {
        name: adapter.json_schema(schema_generator=ContractSchemaGenerator)
        if isinstance(adapter, TypeAdapter)
        else adapter.model_json_schema(schema_generator=ContractSchemaGenerator)
        for name, adapter in models.items()
    }


def write_contract_schemas(directory: str | Path) -> tuple[Path, ...]:
    destination = Path(directory) / CONTRACT_VERSION
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, schema in contract_schemas().items():
        path = destination / f"{name}.schema.json"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
        written.append(path)
    return tuple(written)


def contract_fixture() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "candidates": {
            "text": {"values": {"instructions": "Answer precisely."}},
            "tool": {"values": {"tool.search.description": "Search verified sources."}},
            "output_field": {
                "values": {"output.answer.description": "A concise supported answer."}
            },
        },
        "multimodal_example": {
            "id": "invoice-1",
            "inputs": {"instruction": "Extract the invoice total."},
            "expected_output": {"total": 42.0},
            "attachments": [
                {
                    "kind": "image",
                    "reference": "fixtures/invoice.png",
                    "media_type": "image/png",
                }
            ],
        },
        "failed_evaluation": {
            "case_id": "invoice-2",
            "error": {"type": "TimeoutError", "message": "subject timed out"},
        },
        "resumed_run": {
            "run_id": "contract-run",
            "checkpoint": "runs/contract/backend",
            "next_stage": 1,
        },
    }


__all__ = (
    "CONTRACT_VERSION",
    "ContractSchemaGenerator",
    "contract_fixture",
    "contract_schemas",
    "write_contract_schemas",
)
