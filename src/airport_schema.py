"""Airport topology schema and loader."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class NodeType(StrEnum):
    """Node type enum for airport topology."""

    STAND = "stand"
    PUSHBACK_SPOT = "pushback_spot"
    TAXI_POINT = "taxi_point"
    HOLD_SHORT = "hold_short"
    RUNWAY_ENTRY = "runway_entry"
    DEPARTURE_QUEUE = "departure_queue"
    APPROACH_FIX = "approach_fix"
    GLIDE_PATH = "glide_path"
    LANDING_THRESHOLD = "landing_threshold"
    RUNWAY_CENTER = "runway_center"
    EXIT_TAXIWAY = "exit_taxiway"
    GATE = "gate"


class EdgeMovement(StrEnum):
    """Edge movement type enum."""

    PUSHBACK = "pushback"
    TAXI = "taxi"
    QUEUE_JOIN = "queue_join"
    RUNWAY_TRANSITION = "runway_transition"
    APPROACH = "approach"
    LANDING = "landing"
    TAKEOFF_RUN = "takeoff_run"
    EXIT_RUNWAY = "exit_runway"


class AirportNode(BaseModel):
    """Airport topology node."""

    model_config = ConfigDict(extra="forbid")

    id: str
    node_type: NodeType
    x_ft: float
    y_ft: float
    name: str | None = None
    annotations: dict[str, str] = Field(default_factory=dict)


class AirportEdge(BaseModel):
    """Airport topology edge."""

    model_config = ConfigDict(extra="forbid")

    from_node: str
    to_node: str
    movement_type: EdgeMovement
    distance_ft: float
    max_speed_kt: float
    one_way: bool = False


class AirportSchema(BaseModel):
    """Airport topology with metadata, nodes, edges, and annotations."""

    model_config = ConfigDict(extra="forbid")

    airport_code: str
    nodes: dict[str, AirportNode]
    edges: list[AirportEdge]
    runways: list[dict]
    gates: list[dict]
    metadata: dict[str, str] = Field(default_factory=dict)


class AirportSchemaLoader:
    """Loads and validates airport schema from JSON files."""

    @staticmethod
    def load(schema_name: str) -> AirportSchema:
        """Load airport schema from data/{code}.json."""
        data_dir = Path(__file__).parent.parent / "data"
        file_path = data_dir / f"{schema_name}.json"
        schema = AirportSchema.model_validate_json(
            file_path.read_text(encoding="utf-8")
        )
        errors = AirportSchemaLoader.validate_topology(schema)
        if errors:
            raise ValueError(f"Schema topology errors: {errors}")
        return schema

    @staticmethod
    def validate_topology(schema: AirportSchema) -> list[str]:
        """Validate schema topology, returns list of error messages (empty = valid)."""
        errors = []
        node_ids = set(schema.nodes.keys())

        for edge in schema.edges:
            if edge.from_node not in node_ids:
                errors.append(f"Edge references unknown from_node: {edge.from_node}")
            if edge.to_node not in node_ids:
                errors.append(f"Edge references unknown to_node: {edge.to_node}")

        return errors
