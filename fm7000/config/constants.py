"""Core constants and specifications for the FM 7000 Filling Machine."""

from dataclasses import dataclass
from enum import Enum, IntEnum


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


RECIPE_SEQUENCE: list[MeatType] = [
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
    # fette solo appoggiate: nessun piatto premente, nessuna compattazione
    layer_compression_ratio: float = 1.00
    max_overlap_ratio: float = 0.35
    # oltre questa sovrapposizione conviene chiudere lo strato e ripartire
    layer_close_overlap_ratio: float = 0.60
    max_layer_step_mm: float = 15.0
    slice_compliance: float = 0.5

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
    external_envelope_mm: float = 180.0
    push_safety_margin_mm: float = 10.0

    @property
    def external_interaxis_mm(self) -> float:
        """Interasse tra i centri delle ventose esterne: 180 - 30 = 150 mm."""
        return self.external_envelope_mm - self.cup_diameter_mm

    @property
    def cup_spacing_mm(self) -> float:
        """Passo tra ventose: 150 mm di interasse su 3 intervalli = 50 mm."""
        return self.external_interaxis_mm / (self.cols - 1)

    @property
    def grid_width_mm(self) -> float:
        return self.external_interaxis_mm

    @property
    def grid_length_mm(self) -> float:
        return self.external_interaxis_mm

    @property
    def envelope_mm(self) -> float:
        """Ingombro fisico della mano, labbri delle ventose compresi."""
        return self.external_envelope_mm

    @property
    def gap_to_wall_mm(self) -> float:
        return (210.0 - self.external_envelope_mm) / 2.0

    def hand_play_mm(self, cube_side_mm: float) -> float:
        """Corsa laterale della mano dentro il cubo, per lato.

        Ingombro mano 180 mm in un cubo da 210: 15 mm di corsa per lato, quindi
        la mano puo' avvicinarsi alle pareti e il resto lo fa l'offset presa.
        """
        return max(0.0, (cube_side_mm - self.envelope_mm) / 2.0)


@dataclass(frozen=True)
class RobotSpec:
    model: str = "KUKA KR 3 DELTA D1200 HM"
    payload_kg: float = 3.0
    max_payload_kg: float = 6.0
    reach_diameter_mm: float = 1200.0
    vertical_workspace_mm: float = 250.0
    # asse 4 non continuo: i tubi aria limitano la corsa a +/-180 gradi
    # (fine corsa meccanico ~185 gradi, si lavora con 180)
    wrist_limit_deg: float = 180.0
    adept_cycle_sec: float = 0.5
    realistic_cycle_sec: float = 2.8
    wrist_speed_deg_per_sec: float = 720.0
    max_speed_mm_per_sec: float = 8000.0
    max_accel_mm_per_sec2: float = 40000.0
    move_to_pick_sec: float = 0.45
    pick_sec: float = 0.20
    traverse_sec: float = 0.55
    cube_descent_sec: float = 0.40
    release_sec: float = 0.10
    push_to_wall_sec: float = 0.60
    ascend_return_sec: float = 0.50
    ip_rating_body: str = "IP67"
    ip_rating_axis4: str = "IP69K"


@dataclass(frozen=True)
class SliceConstraints:
    min_width_mm: float = 50.0
    max_width_mm: float = 200.0
    min_length_mm: float = 50.0
    max_length_mm: float = 200.0
    min_thickness_mm: float = 20.0
    max_thickness_mm: float = 40.0


@dataclass(frozen=True)
class PushToWallSpec:
    push_threshold_mm: float = 30.0
    # corsa di spinta contro la parete: 10 mm di bordo che si flette
    wall_compression_mm: float = 10.0
    corner_compression_mm: float = 10.0


@dataclass(frozen=True)
class PlacementSearchSpec:
    """
    Discretizzazione della ricerca di posizione e rotazione.

    La mano deve entrare e uscire dal cubo perpendicolare alle pareti: la
    rotazione asse 4 al deposito e' ammessa solo a multipli di 90 gradi.
    L'orientamento fine arriva da come la fetta e' girata sul nastro.
    """

    perimeter_step_voxels: int = 2
    interior_step_voxels: int = 4
    # il deposito e' vincolato ai multipli di 90 (mano quadra alle pareti):
    # l'angolo fine viene dalla presa, che puo' essere girata di 1 grado
    deposit_step_deg: int = 90
    rotation_step_deg: int = 15
    max_candidates: int = 24
    conveyor_frames_per_pick: int = 6
    # quante attese a vuoto sul nastro prima di dichiarare il cubo chiuso
    max_empty_conveyor_cycles: int = 25


@dataclass(frozen=True)
class PerformanceTargets:
    cubes_per_hour: int = 31
    arrosticini_per_hour: int = 7000
    arrosticini_per_cube: int = 225
    estimated_slices_per_cube_min: int = 28
    estimated_slices_per_cube_max: int = 35
    target_cycle_sec: float = 2.8
    target_cubes_per_hour: int = 40


CUBE = CubeSpec()
GRIPPER = GripperSpec()
ROBOT = RobotSpec()
SLICE_CONSTRAINTS = SliceConstraints()
PUSH_TO_WALL = PushToWallSpec()
SEARCH = PlacementSearchSpec()
PERFORMANCE = PerformanceTargets()
