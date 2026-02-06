"""Placement engine - perimeter-first strategy, push-to-wall, wedge matching."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from fm7000.config.constants import (
    CUBE,
    PUSH_TO_WALL,
    CubeSpec,
    PlacementZone,
    PushDirection,
    PushToWallSpec,
)
from fm7000.cube.state import CubeState
from fm7000.cube.slice_model import MeatSlice


@dataclass
class PlacementCandidate:
    x: int
    y: int
    zone: PlacementZone
    rotation_deg: int
    push_direction: PushDirection
    push_x_mm: float
    push_y_mm: float
    score: float
    wedge_match_score: float = 0.0
    fat_overlap_score: float = 0.0


class PlacementEngine:
    """
    Deterministic placement rules for the FM 7000.

    Priority: corners -> edges -> center (perimeter-first).
    Each placement includes push-to-wall calculation and wedge matching.
    """

    def __init__(
        self,
        cube_spec: Optional[CubeSpec] = None,
        push_spec: Optional[PushToWallSpec] = None,
    ) -> None:
        self.cube = cube_spec or CUBE
        self.push = push_spec or PUSH_TO_WALL
        self.w = self.cube.w_voxels
        self.l = self.cube.l_voxels

    def find_candidates(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
        max_candidates: int = 20,
    ) -> List[PlacementCandidate]:
        candidates: List[PlacementCandidate] = []
        rotations = [0, 90, 180, 270]

        for rotation in rotations:
            rotated = meat_slice.rotate(rotation)
            sw, sl = rotated.shape_mask.shape

            corners = self._get_corner_positions(sw, sl)
            for x, y, zone in corners:
                candidate = self._evaluate_position(
                    cube_state, rotated, x, y, zone, rotation
                )
                if candidate is not None:
                    candidates.append(candidate)

            edges = self._get_edge_positions(sw, sl)
            for x, y, zone in edges:
                candidate = self._evaluate_position(
                    cube_state, rotated, x, y, zone, rotation
                )
                if candidate is not None:
                    candidates.append(candidate)

            center_positions = self._get_center_positions(sw, sl)
            for x, y, zone in center_positions:
                candidate = self._evaluate_position(
                    cube_state, rotated, x, y, zone, rotation
                )
                if candidate is not None:
                    candidates.append(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:max_candidates]

    def find_best_placement(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
    ) -> Optional[PlacementCandidate]:
        candidates = self.find_candidates(cube_state, meat_slice)
        if not candidates:
            return None
        return candidates[0]

    def _get_corner_positions(
        self, sw: int, sl: int
    ) -> List[Tuple[int, int, PlacementZone]]:
        positions = []
        if sw <= self.w and sl <= self.l:
            positions.append((0, 0, PlacementZone.CORNER_TL))
        if sw <= self.w and sl <= self.l:
            positions.append((self.w - sw, 0, PlacementZone.CORNER_TR))
        if sw <= self.w and sl <= self.l:
            positions.append((0, self.l - sl, PlacementZone.CORNER_BL))
        if sw <= self.w and sl <= self.l:
            positions.append((self.w - sw, self.l - sl, PlacementZone.CORNER_BR))
        return positions

    def _get_edge_positions(
        self, sw: int, sl: int
    ) -> List[Tuple[int, int, PlacementZone]]:
        positions = []
        mid_x = max(0, (self.w - sw) // 2)
        mid_y = max(0, (self.l - sl) // 2)

        if sw <= self.w:
            positions.append((mid_x, 0, PlacementZone.EDGE_TOP))
            positions.append((mid_x, self.l - sl, PlacementZone.EDGE_BOTTOM))
        if sl <= self.l:
            positions.append((0, mid_y, PlacementZone.EDGE_LEFT))
            positions.append((self.w - sw, mid_y, PlacementZone.EDGE_RIGHT))
        return positions

    def _get_center_positions(
        self, sw: int, sl: int
    ) -> List[Tuple[int, int, PlacementZone]]:
        positions = []
        cx = max(0, (self.w - sw) // 2)
        cy = max(0, (self.l - sl) // 2)
        if sw <= self.w and sl <= self.l:
            positions.append((cx, cy, PlacementZone.CENTER))
            offsets = [(-2, 0), (2, 0), (0, -2), (0, 2)]
            for dx, dy in offsets:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx <= self.w - sw and 0 <= ny <= self.l - sl:
                    positions.append((nx, ny, PlacementZone.CENTER))
        return positions

    def _evaluate_position(
        self,
        cube_state: CubeState,
        rotated_slice: MeatSlice,
        x: int,
        y: int,
        zone: PlacementZone,
        rotation: int,
    ) -> Optional[PlacementCandidate]:
        if not cube_state.can_place(rotated_slice, x, y):
            return None

        push_dir = self._get_push_direction(zone)
        push_x, push_y = self._calculate_push(x, y, rotated_slice, push_dir)

        zone_score = self._zone_priority_score(zone)
        wedge_score = self._wedge_match_score(cube_state, rotated_slice, x, y)
        fat_score = self._fat_overlap_score(cube_state, rotated_slice, x, y)

        total_score = zone_score * 3.0 + wedge_score * 2.0 + fat_score * 1.5

        return PlacementCandidate(
            x=x,
            y=y,
            zone=zone,
            rotation_deg=rotation,
            push_direction=push_dir,
            push_x_mm=push_x,
            push_y_mm=push_y,
            score=total_score,
            wedge_match_score=wedge_score,
            fat_overlap_score=fat_score,
        )

    def _zone_priority_score(self, zone: PlacementZone) -> float:
        priorities = {
            PlacementZone.CORNER_TL: 1.0,
            PlacementZone.CORNER_TR: 1.0,
            PlacementZone.CORNER_BL: 1.0,
            PlacementZone.CORNER_BR: 1.0,
            PlacementZone.EDGE_TOP: 0.7,
            PlacementZone.EDGE_BOTTOM: 0.7,
            PlacementZone.EDGE_LEFT: 0.7,
            PlacementZone.EDGE_RIGHT: 0.7,
            PlacementZone.CENTER: 0.4,
        }
        return priorities.get(zone, 0.0)

    def _wedge_match_score(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> float:
        sw, sl = meat_slice.shape_mask.shape
        region_height = cube_state.height_map[x:x + sw, y:y + sl].copy()
        active = meat_slice.shape_mask > 0
        if np.sum(active) == 0:
            return 0.0

        heights_active = region_height[active]
        if heights_active.size == 0:
            return 0.5

        height_variance = float(np.std(heights_active))
        thickness_active = meat_slice.thickness_map[active]
        predicted_surface = region_height + meat_slice.thickness_map
        predicted_active = predicted_surface[active]
        flatness = 1.0 / (1.0 + float(np.std(predicted_active)))

        return flatness

    def _fat_overlap_score(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> float:
        if cube_state.total_slices_placed == 0:
            return 1.0

        sw, sl = meat_slice.shape_mask.shape
        existing_fat_density = cube_state.get_fat_density_at(x, y, sw, sl)
        slice_fat = float(np.mean(meat_slice.fat_map[meat_slice.shape_mask > 0]))

        overlap = existing_fat_density * slice_fat
        return 1.0 - min(overlap, 1.0)

    def _get_push_direction(self, zone: PlacementZone) -> PushDirection:
        push_map = {
            PlacementZone.CORNER_TL: PushDirection.LEFT_FRONT,
            PlacementZone.CORNER_TR: PushDirection.RIGHT_FRONT,
            PlacementZone.CORNER_BL: PushDirection.LEFT_BACK,
            PlacementZone.CORNER_BR: PushDirection.RIGHT_BACK,
            PlacementZone.EDGE_TOP: PushDirection.FRONT,
            PlacementZone.EDGE_BOTTOM: PushDirection.BACK,
            PlacementZone.EDGE_LEFT: PushDirection.LEFT,
            PlacementZone.EDGE_RIGHT: PushDirection.RIGHT,
            PlacementZone.CENTER: PushDirection.NONE,
        }
        return push_map.get(zone, PushDirection.NONE)

    def _calculate_push(
        self,
        x: int,
        y: int,
        meat_slice: MeatSlice,
        push_dir: PushDirection,
    ) -> Tuple[float, float]:
        if push_dir == PushDirection.NONE:
            return 0.0, 0.0

        push_x, push_y = 0.0, 0.0
        sw, sl = meat_slice.shape_mask.shape
        x_mm = x * self.cube.resolution_mm
        y_mm = y * self.cube.resolution_mm
        x_end_mm = x_mm + sw * self.cube.resolution_mm
        y_end_mm = y_mm + sl * self.cube.resolution_mm

        is_corner = push_dir in (
            PushDirection.LEFT_FRONT,
            PushDirection.LEFT_BACK,
            PushDirection.RIGHT_FRONT,
            PushDirection.RIGHT_BACK,
        )
        compression = self.push.corner_compression_mm if is_corner else self.push.wall_compression_mm

        if "LEFT" in push_dir.value.upper():
            if x_mm < self.push.push_threshold_mm:
                push_x = -(x_mm + compression)
        if "RIGHT" in push_dir.value.upper():
            gap_right = self.cube.width_mm - x_end_mm
            if gap_right < self.push.push_threshold_mm:
                push_x = gap_right + compression
        if "FRONT" in push_dir.value.upper():
            if y_mm < self.push.push_threshold_mm:
                push_y = -(y_mm + compression)
        if "BACK" in push_dir.value.upper():
            gap_back = self.cube.length_mm - y_end_mm
            if gap_back < self.push.push_threshold_mm:
                push_y = gap_back + compression

        return push_x, push_y
