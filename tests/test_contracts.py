from __future__ import annotations as _annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from pydantic_gepa.contracts import (
    CONTRACT_VERSION,
    contract_fixture,
    contract_schemas,
    write_contract_schemas,
)
from pydantic_gepa.events import Event


def test_contract_schemas_are_versioned_and_validate_without_consumer_imports(
    tmp_path: Path,
) -> None:
    schemas = contract_schemas()
    paths = write_contract_schemas(tmp_path)

    assert set(schemas) == {
        "candidate",
        "case_result",
        "config",
        "event",
        "plan",
        "plan_result",
        "result",
        "stage_result",
    }
    assert schemas["event"]["discriminator"]["propertyName"] == "kind"
    config_schema = json.dumps(schemas["config"])
    assert "python-callable-reference" in config_schema
    assert "python-object-reference" in config_schema
    assert len(paths) == len(schemas)
    assert all(path.parent.name == CONTRACT_VERSION for path in paths)
    assert json.loads(paths[0].read_text(encoding="utf-8"))["title"]


def test_contract_fixture_covers_generic_assets_multimodal_failure_and_resume() -> None:
    fixture = contract_fixture()

    assert fixture["contract_version"] == CONTRACT_VERSION
    assert set(fixture["candidates"]) == {"text", "tool", "output_field"}
    assert fixture["multimodal_example"]["attachments"][0]["kind"] == "image"
    assert fixture["failed_evaluation"]["error"]["type"] == "TimeoutError"
    assert fixture["resumed_run"]["next_stage"] == 1
    TypeAdapter(Event).validate_python(
        {
            "kind": "candidate.accepted",
            "run_id": "contract-run",
            "candidate_id": "candidate-1",
            "score": 0.9,
        }
    )
