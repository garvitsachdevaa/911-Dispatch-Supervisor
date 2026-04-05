from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.airport_schema import (
    AirportEdge,
    AirportNode,
    AirportSchema,
    AirportSchemaLoader,
    EdgeMovement,
    NodeType,
)

DATA_ROOT = ROOT / "data"


class AirportSchemaTests(unittest.TestCase):
    def test_gatwick_fixture_loads(self) -> None:
        schema = AirportSchemaLoader.load("egkk_gatwick")
        self.assertEqual(schema.airport_code, "EGKK")

        node_types = {node.node_type for node in schema.nodes.values()}
        self.assertIn(NodeType.APPROACH_FIX, node_types)
        self.assertIn(NodeType.GLIDE_PATH, node_types)
        self.assertIn(NodeType.LANDING_THRESHOLD, node_types)
        self.assertIn(NodeType.RUNWAY_CENTER, node_types)
        self.assertIn(NodeType.STAND, node_types)
        self.assertIn(NodeType.PUSHBACK_SPOT, node_types)
        self.assertIn(NodeType.DEPARTURE_QUEUE, node_types)

    def test_dummy_fixture_loads(self) -> None:
        schema = AirportSchemaLoader.load("dummy_small")
        self.assertEqual(schema.airport_code, "DUMMY")
        self.assertGreaterEqual(len(schema.nodes), 10)
        self.assertGreaterEqual(len(schema.edges), 5)

    def test_approach_and_departure_topology_present(self) -> None:
        schema = AirportSchemaLoader.load("egkk_gatwick")
        node_types = {n.node_type for n in schema.nodes.values()}

        arrival_types = {
            NodeType.APPROACH_FIX,
            NodeType.GLIDE_PATH,
            NodeType.LANDING_THRESHOLD,
        }
        departure_types = {NodeType.PUSHBACK_SPOT, NodeType.DEPARTURE_QUEUE}

        self.assertTrue(
            arrival_types.issubset(node_types), "Missing arrival topology nodes"
        )
        self.assertTrue(
            departure_types.issubset(node_types), "Missing departure topology nodes"
        )

    def test_bad_edge_reference_rejected(self) -> None:
        schema = AirportSchemaLoader.load("dummy_small")
        bad_edge = AirportEdge(
            from_node="MISSING_NODE",
            to_node="THR",
            movement_type=EdgeMovement.TAXI,
            distance_ft=100.0,
            max_speed_kt=20.0,
        )
        schema.edges.append(bad_edge)

        errors = AirportSchemaLoader.validate_topology(schema)
        self.assertGreater(len(errors), 0)
        self.assertTrue(any("MISSING_NODE" in e for e in errors))

    def test_extra_fields_rejected(self) -> None:
        data = json.loads((DATA_ROOT / "dummy_small.json").read_text(encoding="utf-8"))
        data["extra_field"] = "should be rejected"

        with self.assertRaises(ValidationError):
            AirportSchema.model_validate(data)

    def test_node_extra_fields_rejected(self) -> None:
        data = json.loads((DATA_ROOT / "dummy_small.json").read_text(encoding="utf-8"))
        data["nodes"]["APP_FIX"]["extra"] = "rejected"

        with self.assertRaises(ValidationError):
            AirportSchema.model_validate(data)

    def test_missing_node_type_rejected(self) -> None:
        data = json.loads((DATA_ROOT / "dummy_small.json").read_text(encoding="utf-8"))
        data["nodes"]["APP_FIX"]["node_type"] = "invalid_node_type"

        with self.assertRaises(ValidationError):
            AirportSchema.model_validate(data)

    def test_missing_edge_movement_rejected(self) -> None:
        data = json.loads((DATA_ROOT / "dummy_small.json").read_text(encoding="utf-8"))
        data["edges"][0]["movement_type"] = "invalid_movement"

        with self.assertRaises(ValidationError):
            AirportSchema.model_validate(data)

    def test_validate_topology_empty_errors(self) -> None:
        schema = AirportSchemaLoader.load("dummy_small")
        errors = AirportSchemaLoader.validate_topology(schema)
        self.assertEqual(errors, [])

    def test_node_has_required_fields(self) -> None:
        node = AirportNode(
            id="TEST",
            node_type=NodeType.STAND,
            x_ft=100.0,
            y_ft=200.0,
        )
        self.assertEqual(node.id, "TEST")
        self.assertEqual(node.node_type, NodeType.STAND)
        self.assertIsNone(node.name)
        self.assertEqual(node.annotations, {})


if __name__ == "__main__":
    unittest.main()
