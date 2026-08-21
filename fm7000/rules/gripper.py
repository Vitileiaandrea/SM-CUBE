"""Gripper pattern selector - determines which vacuum cups to activate."""

from dataclasses import dataclass

import numpy as np

from fm7000.config.constants import GRIPPER, GripperSpec, PlacementZone
from fm7000.cube.slice_model import MeatSlice


@dataclass
class GripperCommand:
    cup_pattern: np.ndarray
    placement_zone: PlacementZone
    wrist_rotation_deg: float
    vacuum_level: float = 0.8
    meat_margin_x_mm: float = 0.0
    meat_margin_y_mm: float = 0.0
    margin_ok: bool = True

    @property
    def active_cups(self) -> int:
        return int(np.sum(self.cup_pattern))

    @property
    def active_cup_positions(self) -> list:
        positions = []
        for i in range(self.cup_pattern.shape[0]):
            for j in range(self.cup_pattern.shape[1]):
                if self.cup_pattern[i, j] > 0:
                    positions.append((i, j))
        return positions


class GripperPatternSelector:
    """
    Selects which vacuum cups to activate based on target placement zone.

    The 4x4 grid of cups (180x180mm external) must be positioned so that:
    - The slice hangs toward the target zone in the cube (210x210mm internal)
    - At least 10mm of meat protrudes beyond the cup lip on push-to-wall sides
    - The cup pattern is decided BEFORE picking the slice from the conveyor

    Grid layout (top view, looking down at gripper):
        [0,0] [0,1] [0,2] [0,3]
        [1,0] [1,1] [1,2] [1,3]
        [2,0] [2,1] [2,2] [2,3]
        [3,0] [3,1] [3,2] [3,3]

    Rows 0-3: front to back (Y axis)
    Cols 0-3: left to right (X axis)
    """

    def __init__(self, spec: GripperSpec = GRIPPER) -> None:
        self.spec = spec
        self.rows = spec.rows
        self.cols = spec.cols

    def select_pattern(
        self,
        zone: PlacementZone,
        meat_slice: MeatSlice,
        target_rotation_deg: float = 0.0,
    ) -> GripperCommand:
        pattern = self._fit_pattern(zone, meat_slice)
        margin_x, margin_y = self._meat_margins(pattern, meat_slice)
        required = self.spec.cup_diameter_mm / 2.0 + self.spec.push_safety_margin_mm

        return GripperCommand(
            cup_pattern=pattern,
            placement_zone=zone,
            wrist_rotation_deg=target_rotation_deg,
            vacuum_level=self._calculate_vacuum_level(meat_slice),
            meat_margin_x_mm=margin_x,
            meat_margin_y_mm=margin_y,
            margin_ok=margin_x >= required and margin_y >= required,
        )

    def _fit_pattern(self, zone: PlacementZone, meat_slice: MeatSlice) -> np.ndarray:
        """
        Scegle quante ventose attivare per lasciare almeno 10 mm di carne libera
        oltre il labbro della ventosa: il perimetro serve al push-to-wall.
        """
        cup_radius = self.spec.cup_diameter_mm / 2.0
        required = cup_radius + self.spec.push_safety_margin_mm
        spacing = self.spec.cup_spacing_mm

        max_span_x = meat_slice.width_mm - 2.0 * required
        max_span_y = meat_slice.length_mm - 2.0 * required
        cols_n = 2 if max_span_x >= spacing else 1
        rows_n = 2 if max_span_y >= spacing else 1

        return self._anchored_pattern(zone, rows_n, cols_n)

    def _anchored_pattern(
        self, zone: PlacementZone, rows_n: int, cols_n: int
    ) -> np.ndarray:
        pattern = np.zeros((self.rows, self.cols), dtype=np.int8)
        base = self._get_base_pattern(zone)
        active_rows = np.where(np.any(base > 0, axis=1))[0]
        active_cols = np.where(np.any(base > 0, axis=0))[0]
        if active_rows.size == 0 or active_cols.size == 0:
            pattern[1, 1] = 1
            return pattern

        # mantiene le ventose piu vicine al centro del gripper
        center = (self.rows - 1) / 2.0
        rows = sorted(active_rows, key=lambda i: abs(i - center))[:rows_n]
        cols = sorted(active_cols, key=lambda j: abs(j - center))[:cols_n]
        for i in rows:
            for j in cols:
                pattern[i, j] = 1
        return pattern

    def _meat_margins(
        self, pattern: np.ndarray, meat_slice: MeatSlice
    ) -> tuple[float, float]:
        """Carne libera tra labbro ventose esterne e bordo fetta, per lato."""
        active_rows = np.where(np.any(pattern > 0, axis=1))[0]
        active_cols = np.where(np.any(pattern > 0, axis=0))[0]
        if active_rows.size == 0 or active_cols.size == 0:
            return 0.0, 0.0

        cup_radius = self.spec.cup_diameter_mm / 2.0
        span_x = (active_cols[-1] - active_cols[0]) * self.spec.cup_spacing_mm
        span_y = (active_rows[-1] - active_rows[0]) * self.spec.cup_spacing_mm
        margin_x = (meat_slice.width_mm - span_x) / 2.0 - cup_radius
        margin_y = (meat_slice.length_mm - span_y) / 2.0 - cup_radius
        return float(margin_x), float(margin_y)

    def _get_base_pattern(self, zone: PlacementZone) -> np.ndarray:
        pattern = np.zeros((self.rows, self.cols), dtype=np.int8)

        if zone == PlacementZone.CORNER_TL:
            pattern[0:2, 0:2] = 1
        elif zone == PlacementZone.CORNER_TR:
            pattern[0:2, 2:4] = 1
        elif zone == PlacementZone.CORNER_BL:
            pattern[2:4, 0:2] = 1
        elif zone == PlacementZone.CORNER_BR:
            pattern[2:4, 2:4] = 1

        elif zone == PlacementZone.EDGE_TOP:
            pattern[0:2, 1:3] = 1
        elif zone == PlacementZone.EDGE_BOTTOM:
            pattern[2:4, 1:3] = 1
        elif zone == PlacementZone.EDGE_LEFT:
            pattern[1:3, 0:2] = 1
        elif zone == PlacementZone.EDGE_RIGHT:
            pattern[1:3, 2:4] = 1

        elif zone == PlacementZone.CENTER:
            pattern[1:3, 1:3] = 1

        return pattern

    def _calculate_vacuum_level(self, meat_slice: MeatSlice) -> float:
        weight_estimate_g = meat_slice.volume_mm3 * 0.001 * 1.05
        if weight_estimate_g > 500:
            return 0.95
        elif weight_estimate_g > 200:
            return 0.85
        return 0.75

    def get_cup_center_positions_mm(
        self, pattern: np.ndarray
    ) -> list:
        positions = []
        offset_x = -(self.spec.external_interaxis_mm / 2.0)
        offset_y = -(self.spec.external_interaxis_mm / 2.0)

        for i in range(self.rows):
            for j in range(self.cols):
                if pattern[i, j] > 0:
                    cx = offset_x + j * self.spec.cup_spacing_mm
                    cy = offset_y + i * self.spec.cup_spacing_mm
                    positions.append((cx, cy))
        return positions
