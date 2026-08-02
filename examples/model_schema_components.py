from __future__ import annotations as _annotations

# pyright: reportMissingImports=false
from pydantic import BaseModel, ConfigDict, Field

from pydantic_gepa import apply_model_schema_candidate, collect_model_components


class ShippingAddress(BaseModel):
    city: str = Field(description="City to ship to")
    postal_code: str


class InventorySummary(BaseModel):
    model_config = ConfigDict(json_schema_extra={"description": "Inventory summary schema"})

    warehouses: list[ShippingAddress]
    status: str = Field(description="Inventory status")


def main() -> None:
    catalog = collect_model_components(
        InventorySummary,
        model_name="InventorySummary",
    )
    candidate = {
        "output:InventorySummary:description": "Optimized inventory output schema.",
        "output:InventorySummary:param:warehouses[].city": "Warehouse city name for the shipment.",
        "output:InventorySummary:param:status": "Current warehouse availability status.",
    }
    updated = apply_model_schema_candidate(
        InventorySummary,
        candidate,
        model_name="InventorySummary",
        components=catalog,
    )

    print("components:", catalog.names())
    print("overrides:", updated.description_overrides)
    print("schema description:", updated.description)
    print("schema:", updated.json_schema)


if __name__ == "__main__":
    main()
