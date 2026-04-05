"""City topology/resource schema and loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.models import IncidentType, UnitType


class CityUnitConfig(BaseModel):
    """Unit configuration loaded from the city schema."""

    model_config = ConfigDict(extra="forbid")

    unit_id: str
    unit_type: UnitType
    base_x: float
    base_y: float
    crew_count: int = 1


class CitySchema(BaseModel):
    """City-wide schema including unit inventory and default response patterns."""

    model_config = ConfigDict(extra="forbid")

    city_name: str
    grid_size: list[int] = Field(..., min_length=2, max_length=2)
    districts: list[str] = Field(default_factory=list)
    units: list[CityUnitConfig] = Field(default_factory=list)
    unit_speeds: dict[UnitType, float] = Field(default_factory=dict)
    default_required_units: dict[IncidentType, list[UnitType]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def unit_config_by_id(self) -> dict[str, CityUnitConfig]:
        return {u.unit_id: u for u in self.units}


class CitySchemaLoader:
    """Loads and validates city schema from JSON files."""

    @staticmethod
    def load(schema_name: str) -> CitySchema:
        """Load city schema from data/{name}.json."""
        data_dir = Path(__file__).parent.parent / "data"
        file_path = data_dir / f"{schema_name}.json"
        return CitySchema.model_validate_json(file_path.read_text(encoding="utf-8"))
