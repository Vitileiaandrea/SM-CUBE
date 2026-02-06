"""Cube state management - 3D voxel grid for tracking fill state."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from fm7000.config.constants import CUBE, CubeSpec, MeatType
from fm7000.cube.slice_model import MeatSlice


@dataclass
class PlacedSlice:
    slice_data: MeatSlice
    x_voxel: int
    y_voxel: int
    z_mm: float
    layer_index: int
    rotation_deg: float = 0.0
    pushed_x_mm: float = 0.0
    pushed_y_mm: float = 0.0


class CubeState:
    """
    Manages the 3D fill state of the cube.

    Tracks:
    - height_map: 2D array with current fill height at each (x,y) position
    - fat_map_cumulative: 2D array tracking fat density vertically
    - layers: list of placed slices grouped by layer
    """

    def __init__(self, spec: Optional[CubeSpec] = None) -> None:
        self.spec = spec or CUBE
        self.w = self.spec.w_voxels
        self.l = self.spec.l_voxels
        self.res = self.spec.resolution_mm

        self.height_map = np.zeros((self.w, self.l), dtype=np.float32)
        self.fat_accumulator = np.zeros((self.w, self.l), dtype=np.float32)
        self.fat_layer_count = np.zeros((self.w, self.l), dtype=np.int32)

        self.layers: List[List[PlacedSlice]] = [[]]
        self.current_layer_index: int = 0
        self.total_slices_placed: int = 0

    @property
    def current_layer_height_mm(self) -> float:
        if self.total_slices_placed == 0:
            return 0.0
        placed = self.layers[self.current_layer_index]
        if not placed:
            return float(np.min(self.height_map[self.height_map > 0])) if np.any(self.height_map > 0) else 0.0
        first_z = placed[0].z_mm
        return first_z

    @property
    def current_layer_coverage(self) -> float:
        if self.total_slices_placed == 0:
            return 0.0
        placed = self.layers[self.current_layer_index]
        if not placed:
            return 0.0

        layer_floor = placed[0].z_mm
        above_floor = self.height_map > layer_floor + 1.0
        return float(np.sum(above_floor)) / float(self.w * self.l)

    @property
    def fill_percentage(self) -> float:
        return float(np.mean(self.height_map)) / self.spec.height_mm

    @property
    def is_full(self) -> bool:
        return float(np.mean(self.height_map)) >= self.spec.height_mm * 0.95

    def get_fat_density_at(self, x: int, y: int, w: int, l: int) -> float:
        region_count = self.fat_layer_count[x:x + w, y:y + l]
        if np.sum(region_count) == 0:
            return 0.0
        region_fat = self.fat_accumulator[x:x + w, y:y + l]
        mask = region_count > 0
        return float(np.mean(region_fat[mask] / region_count[mask]))

    def can_place(
        self,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> bool:
        sw = meat_slice.shape_mask.shape[0]
        sl = meat_slice.shape_mask.shape[1]

        if x < 0 or y < 0:
            return False
        if x + sw > self.w or y + sl > self.l:
            return False

        region_height = self.height_map[x:x + sw, y:y + sl]
        active = meat_slice.shape_mask > 0
        if np.sum(active) == 0:
            return False

        heights_at_active = region_height[active]
        height_range = float(np.max(heights_at_active) - np.min(heights_at_active))
        layer_base = float(np.min(heights_at_active))

        max_after = layer_base + meat_slice.thickness_max_mm
        if max_after > self.spec.height_mm:
            return False

        if self.current_layer_index > 0:
            max_height_diff = 15.0
            if height_range > max_height_diff:
                return False

        return True

    def place_slice(
        self,
        meat_slice: MeatSlice,
        x: int,
        y: int,
        push_x_mm: float = 0.0,
        push_y_mm: float = 0.0,
    ) -> Optional[PlacedSlice]:
        if not self.can_place(meat_slice, x, y):
            return None

        sw, sl = meat_slice.shape_mask.shape
        region_height = self.height_map[x:x + sw, y:y + sl]
        active = meat_slice.shape_mask > 0
        z_base = float(np.max(region_height[active]))

        thickness = meat_slice.thickness_map
        new_heights = np.where(active, z_base + thickness, 0)
        self.height_map[x:x + sw, y:y + sl] = np.maximum(
            region_height, new_heights
        )

        active_fat = meat_slice.fat_map * meat_slice.shape_mask
        self.fat_accumulator[x:x + sw, y:y + sl] += active_fat
        self.fat_layer_count[x:x + sw, y:y + sl] += meat_slice.shape_mask.astype(np.int32)

        placed = PlacedSlice(
            slice_data=meat_slice,
            x_voxel=x,
            y_voxel=y,
            z_mm=z_base,
            layer_index=self.current_layer_index,
            rotation_deg=meat_slice.orientation_deg,
            pushed_x_mm=push_x_mm,
            pushed_y_mm=push_y_mm,
        )
        self.layers[self.current_layer_index].append(placed)
        self.total_slices_placed += 1
        return placed

    def advance_layer(self) -> bool:
        coverage = self.current_layer_coverage
        if coverage < self.spec.layer_coverage_threshold:
            return False

        self._press_current_layer()
        self.current_layer_index += 1
        self.layers.append([])
        return True

    def force_advance_layer(self) -> None:
        self._press_current_layer()
        self.current_layer_index += 1
        self.layers.append([])

    def _press_current_layer(self) -> None:
        if not self.layers[self.current_layer_index]:
            return

        placed = self.layers[self.current_layer_index]
        layer_base = placed[0].z_mm

        for ps in placed:
            sw, sl = ps.slice_data.shape_mask.shape
            x, y = ps.x_voxel, ps.y_voxel
            region = self.height_map[x:x + sw, y:y + sl]
            active = ps.slice_data.shape_mask > 0
            above_base = region > layer_base
            compress_mask = active & above_base
            if np.any(compress_mask):
                excess = region[compress_mask] - layer_base
                compressed = layer_base + excess * self.spec.layer_compression_ratio
                region[compress_mask] = compressed
                self.height_map[x:x + sw, y:y + sl] = region

    def get_height_at(self, x: int, y: int) -> float:
        if 0 <= x < self.w and 0 <= y < self.l:
            return float(self.height_map[x, y])
        return 0.0

    def get_layer_floor(self) -> float:
        placed = self.layers[self.current_layer_index]
        if not placed:
            if self.current_layer_index == 0:
                return 0.0
            prev_placed = self.layers[self.current_layer_index - 1]
            if prev_placed:
                xs, ys = [], []
                for ps in prev_placed:
                    xs.append(ps.x_voxel)
                    ys.append(ps.y_voxel)
                return float(np.mean(self.height_map))
            return 0.0
        return placed[0].z_mm

    def get_state_summary(self) -> dict:
        return {
            "current_layer": self.current_layer_index,
            "total_slices": self.total_slices_placed,
            "fill_percentage": self.fill_percentage,
            "layer_coverage": self.current_layer_coverage,
            "mean_height_mm": float(np.mean(self.height_map)),
            "max_height_mm": float(np.max(self.height_map)),
            "min_height_mm": float(np.min(self.height_map[self.height_map > 0])) if np.any(self.height_map > 0) else 0.0,
            "is_full": self.is_full,
        }

    def reset(self) -> None:
        self.height_map = np.zeros((self.w, self.l), dtype=np.float32)
        self.fat_accumulator = np.zeros((self.w, self.l), dtype=np.float32)
        self.fat_layer_count = np.zeros((self.w, self.l), dtype=np.int32)
        self.layers = [[]]
        self.current_layer_index = 0
        self.total_slices_placed = 0
