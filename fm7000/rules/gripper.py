"""Gripper pattern selector - determines which vacuum cups to activate."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from fm7000.config.constants import GRIPPER, GripperSpec, PlacementZone
from fm7000.cube.slice_model import MeatSlice


@dataclass
class GripperCommand:
    cup_pattern: np.ndarray
    placement_zone: PlacementZone
    wrist_rotation_deg: float
    vacuum_level: float = 0.8

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
        pattern = self._get_base_pattern(zone)
        pattern = self._validate_overhang(pattern, zone, meat_slice)

        return GripperCommand(
            cup_pattern=pattern,
            placement_zone=zone,
            wrist_rotation_deg=target_rotation_deg,
            vacuum_level=self._calculate_vacuum_level(meat_slice),
        )

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

    def _validate_overhang(
        self,
        pattern: np.ndarray,
        zone: PlacementZone,
        meat_slice: MeatSlice,
    ) -> np.ndarray:
        min_overhang_mm = self.spec.push_safety_margin_mm
        cup_radius_mm = self.spec.cup_diameter_mm / 2.0
        required_reach_mm = cup_radius_mm + min_overhang_mm

        active_rows = np.where(np.any(pattern > 0, axis=1))[0]
        active_cols = np.where(np.any(pattern > 0, axis=0))[0]

        if len(active_rows) == 0 or len(active_cols) == 0:
            return pattern

        sw_mm = meat_slice.width_mm
        sl_mm = meat_slice.length_mm

        cup_span_x = (active_cols[-1] - active_cols[0]) * self.spec.cup_spacing_mm
        cup_span_y = (active_rows[-1] - active_rows[0]) * self.spec.cup_spacing_mm

        overhang_x = (sw_mm - cup_span_x) / 2.0
        overhang_y = (sl_mm - cup_span_y) / 2.0

        if overhang_x < required_reach_mm or overhang_y < required_reach_mm:
            pattern = self._compact_pattern(pattern, zone)

        return pattern

    def _compact_pattern(
        self, pattern: np.ndarray, zone: PlacementZone
    ) -> np.ndarray:
        if np.sum(pattern) <= 1:
            return pattern

        compact = np.zeros_like(pattern)
        active_positions = []
        for i in range(self.rows):
            for j in range(self.cols):
                if pattern[i, j] > 0:
                    active_positions.append((i, j))

        if zone in (PlacementZone.CORNER_TL, PlacementZone.EDGE_LEFT, PlacementZone.EDGE_TOP):
            for i, j in active_positions:
                ni = min(i + 1, self.rows - 1)
                nj = min(j + 1, self.cols - 1)
                compact[ni, nj] = 1
        elif zone in (PlacementZone.CORNER_TR, PlacementZone.EDGE_RIGHT):
            for i, j in active_positions:
                ni = min(i + 1, self.rows - 1)
                nj = max(j - 1, 0)
                compact[ni, nj] = 1
        elif zone in (PlacementZone.CORNER_BL, PlacementZone.EDGE_BOTTOM):
            for i, j in active_positions:
                ni = max(i - 1, 0)
                nj = min(j + 1, self.cols - 1)
                compact[ni, nj] = 1
        elif zone == PlacementZone.CORNER_BR:
            for i, j in active_positions:
                ni = max(i - 1, 0)
                nj = max(j - 1, 0)
                compact[ni, nj] = 1
        else:
            compact = pattern.copy()

        if np.sum(compact) == 0:
            return pattern

        return compact

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
