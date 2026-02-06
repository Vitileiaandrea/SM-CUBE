"""Core constants and specifications for the FM 7000 Filling Machine."""

from enum import IntEnum, Enum
from dataclasses import dataclass
from typing import List


class MeatType(IntEnum):
    HIGH_QUALITY = 0
    MEDIUM_QUALITY = 1
    FAT = 2


class PlacementZone(Enum):
    CORNER_TL = "corner_top_left"
    CORNER_TR = "corner_top_right"
    CORNER_BL = "corner_bottom_left"
    CORNER_BR = "corner_bottom_right"
    EDGE_TOP = "edge_top"
    EDGE_BOTTOM = "edge_bottom"
    EDGE_LEFT = "edge_left"
    EDGE_RIGHT = "edge_right"
    CENTER = "center"


class PushDirection(Enum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    FRONT = "front"
    BACK = "back"
    LEFT_FRONT = "left_front"
    LEFT_BACK = "left_back"
    RIGHT_FRONT = "right_front"
    RIGHT_BACK = "right_back"


RECIPE_SEQUENCE: List[MeatType] = [
    MeatType.HIGH_QUALITY,
    MeatType.FAT,
    MeatType.MEDIUM_QUALITY,
    MeatType.FAT,
    MeatType.MEDIUM_QUALITY,
    MeatType.FAT,
    MeatType.MEDIUM_QUALITY,
    MeatType.FAT,
    MeatType.MEDIUM_QUALITY,
    MeatType.FAT,
    MeatType.HIGH_QUALITY,
]


@dataclass(frozen=True)
class CubeSpec:
    width_mm: float = 210.0
    length_mm: float = 210.0
    height_mm: float = 250.0
    resolution_mm: float = 5.0
    arrosticini_per_cube: int = 225
    layer_coverage_threshold: float = 0.95
    layer_compression_ratio: float = 0.90

    @property
    def w_voxels(self) -> int:
        return int(self.width_mm / self.resolution_mm)

    @property
    def l_voxels(self) -> int:
        return int(self.length_mm / self.resolution_mm)

    @property
    def h_voxels(self) -> int:
        return int(self.height_mm / self.resolution_mm)


@dataclass(frozen=True)
class GripperSpec:
    rows: int = 4
    cols: int = 4
    cup_diameter_mm: float = 30.0
    external_interaxis_mm: float = 180.0
    cup_spacing_mm: float = 40.0
    push_safety_margin_mm: float = 10.0

    @property
    def grid_width_mm(self) -> float:
        return self.external_interaxis_mm

    @property
    def grid_length_mm(self) -> float:
        return self.external_interaxis_mm

    @property
    def gap_to_wall_mm(self) -> float:
        return (210.0 - self.external_interaxis_mm) / 2.0


@dataclass(frozen=True)
class RobotSpec:
    model: str = "KUKA KR 3 DELTA D1200 HM"
    payload_kg: float = 3.0
    max_payload_kg: float = 6.0
    reach_diameter_mm: float = 1200.0
    vertical_workspace_mm: float = 250.0
    wrist_rotation_deg: float = 360.0
    adept_cycle_sec: float = 0.5
    realistic_cycle_sec: float = 2.5
    ip_rating_body: str = "IP67"
    ip_rating_axis4: str = "IP69K"


@dataclass(frozen=True)
class SliceConstraints:
    min_width_mm: float = 50.0
    max_width_mm: float = 200.0
    min_length_mm: float = 50.0
    max_length_mm: float = 200.0
    min_thickness_mm: float = 5.0
    max_thickness_mm: float = 40.0


@dataclass(frozen=True)
class PushToWallSpec:
    push_threshold_mm: float = 30.0
    wall_compression_mm: float = 25.0
    corner_compression_mm: float = 10.0


@dataclass(frozen=True)
class PerformanceTargets:
    cubes_per_hour: int = 31
    arrosticini_per_hour: int = 7000
    arrosticini_per_cube: int = 225
    estimated_slices_per_cube_min: int = 60
    estimated_slices_per_cube_max: int = 80


CUBE = CubeSpec()
GRIPPER = GripperSpec()
ROBOT = RobotSpec()
SLICE_CONSTRAINTS = SliceConstraints()
PUSH_TO_WALL = PushToWallSpec()
PERFORMANCE = PerformanceTargets()
