"""Tests for city schema loader and validation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.city_schema import CitySchema, CitySchemaLoader

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


class CitySchemaTests(unittest.TestCase):
    def test_metro_city_fixture_loads(self) -> None:
        schema = CitySchemaLoader.load("metro_city")
        self.assertEqual(schema.city_name, "Metro City")
        self.assertGreaterEqual(len(schema.units), 1)

    def test_small_fixture_loads(self) -> None:
        schema = CitySchemaLoader.load("city_small")
        self.assertEqual(schema.city_name, "Smallville")
        self.assertEqual(schema.grid_size, [20, 20])

    def test_unit_config_by_id(self) -> None:
        schema = CitySchemaLoader.load("city_small")
        by_id = schema.unit_config_by_id()
        self.assertIn("MED-1", by_id)

    def test_extra_fields_rejected(self) -> None:
        data = json.loads((DATA_ROOT / "city_small.json").read_text(encoding="utf-8"))
        data["extra_field"] = "nope"
        with self.assertRaises(ValidationError):
            CitySchema.model_validate(data)


if __name__ == "__main__":
    unittest.main()