from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


EXECUTION_BOUNDS = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (0.2, 0.5),
}
MAX_DRONES = 3
MAX_WAYPOINTS = 36
SUPPORTED_TASK_TYPES = {
    "point_to_point",
    "multi_waypoint",
    "circle_patrol",
    "rectangle_patrol",
}


@dataclass(frozen=True)
class DronePlan:
    drone_id: str
    task_type: str
    waypoints: list[list[float]]
    return_home: bool
    altitude_min: float = 0.2
    altitude_max: float = 0.5

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DronePlan":
        return cls(
            drone_id=str(payload["drone_id"]),
            task_type=str(payload["task_type"]),
            waypoints=[[float(value) for value in point] for point in payload["waypoints"]],
            return_home=bool(payload["return_home"]),
            altitude_min=float(payload.get("altitude_min", 0.2)),
            altitude_max=float(payload.get("altitude_max", 0.5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MissionPlan:
    mission_id: str
    drone_plans: list[DronePlan]
    schema_version: str = "1.0"
    coordinate_frame: str = "local_meters"
    selected_drone_id: str = "drone-1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MissionPlan":
        mission = payload.get("mission", {})
        return cls(
            mission_id=str(payload["mission_id"]),
            drone_plans=[DronePlan.from_dict(item) for item in payload["drone_plans"]],
            schema_version=str(payload.get("schema_version", "1.0")),
            coordinate_frame=str(mission.get("coordinate_frame", payload.get("coordinate_frame", "local_meters"))),
            selected_drone_id=str(mission.get("selected_drone_id", payload.get("selected_drone_id", "drone-1"))),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mission_id": self.mission_id,
            "mission": {
                "coordinate_frame": self.coordinate_frame,
                "selected_drone_id": self.selected_drone_id,
            },
            "drone_plans": [plan.to_dict() for plan in self.drone_plans],
            "metadata": self.metadata,
        }
