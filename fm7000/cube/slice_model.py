"""Meat slice model with geometry, thickness map, and fat distribution."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fm7000.config.constants import MeatType, SliceConstraints, SLICE_CONSTRAINTS


@dataclass
class MeatSlice:
    """
    Represents a single meat slice with its physical properties.

    Slices are wedge-shaped (non-uniform thickness) and have a fat distribution
    map that must be considered when stacking layers.
    """

    width_mm: float
    length_mm: float
    thickness_min_mm: float = 5.0
    thickness_max_mm: float = 40.0
    meat_type: MeatType = MeatType.MEDIUM_QUALITY
    fat_percentage: float = 0.0
    slice_id: int = 0
    wedge_direction: int = 0
    shape_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    thickness_map: np.ndarray = field(default_factory=lambda: np.array([]))
    fat_map: np.ndarray = field(default_factory=lambda: np.array([]))
    orientation_deg: float = 0.0
    resolution_mm: float = 5.0

    def __post_init__(self) -> None:
        if self.shape_mask.size == 0:
            self.shape_mask = self._generate_shape()
        if self.thickness_map.size == 0:
            self.thickness_map = self._generate_thickness_map()
        if self.fat_map.size == 0:
            self.fat_map = self._generate_fat_map()

    @property
    def w_voxels(self) -> int:
        return max(1, int(self.width_mm / self.resolution_mm))

    @property
    def l_voxels(self) -> int:
        return max(1, int(self.length_mm / self.resolution_mm))

    @property
    def avg_thickness_mm(self) -> float:
        if self.thickness_map.size == 0:
            return (self.thickness_min_mm + self.thickness_max_mm) / 2
        active = self.thickness_map[self.shape_mask > 0]
        if active.size == 0:
            return self.thickness_min_mm
        return float(np.mean(active))

    @property
    def volume_mm3(self) -> float:
        voxel_area = self.resolution_mm ** 2
        return float(np.sum(self.thickness_map) * voxel_area)

    def _generate_shape(self, irregularity: float = 0.25) -> np.ndarray:
        mask = np.ones((self.w_voxels, self.l_voxels), dtype=np.float32)
        if irregularity > 0:
            center_x, center_y = self.w_voxels / 2, self.l_voxels / 2
            for i in range(self.w_voxels):
                for j in range(self.l_voxels):
                    dist_x = abs(i - center_x) / max(self.w_voxels / 2, 1)
                    dist_y = abs(j - center_y) / max(self.l_voxels / 2, 1)
                    edge_dist = max(dist_x, dist_y)
                    threshold = 1.0 - irregularity * 0.3
                    if edge_dist > threshold:
                        noise = np.random.uniform(0, irregularity * 0.5)
                        if edge_dist + noise > 1.0:
                            mask[i, j] = 0
        return mask

    def _generate_thickness_map(self) -> np.ndarray:
        h, w = self.shape_mask.shape
        if self.wedge_direction == 0:
            gradient = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
            gradient = np.broadcast_to(gradient, (h, w)).copy()
        elif self.wedge_direction == 1:
            gradient = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
            gradient = np.broadcast_to(gradient, (h, w)).copy()
        else:
            x_grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
            y_grad = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
            gradient = (x_grad + y_grad) / 2.0

        thickness_range = self.thickness_max_mm - self.thickness_min_mm
        thickness_map = self.thickness_min_mm + gradient * thickness_range
        thickness_map = thickness_map * self.shape_mask
        return thickness_map.astype(np.float32)

    def _generate_fat_map(self) -> np.ndarray:
        h, w = self.shape_mask.shape
        if self.meat_type == MeatType.FAT:
            fat_map = np.ones((h, w), dtype=np.float32) * 0.9
        else:
            base_fat = self.fat_percentage / 100.0 if self.fat_percentage > 0 else 0.15
            fat_map = np.random.uniform(
                base_fat * 0.3, base_fat * 1.7, (h, w)
            ).astype(np.float32)
            fat_map = np.clip(fat_map, 0.0, 1.0)
        fat_map = fat_map * self.shape_mask
        return fat_map

    def rotate(self, angle_deg: int) -> "MeatSlice":
        rotations = (angle_deg // 90) % 4
        if rotations == 0:
            return self

        new_mask = np.rot90(self.shape_mask, rotations).copy()
        new_thickness = np.rot90(self.thickness_map, rotations).copy()
        new_fat = np.rot90(self.fat_map, rotations).copy()
        new_width = self.length_mm if rotations % 2 else self.width_mm
        new_length = self.width_mm if rotations % 2 else self.length_mm

        active = new_thickness[new_mask > 0]
        new_min = float(active.min()) if active.size > 0 else self.thickness_min_mm
        new_max = float(active.max()) if active.size > 0 else self.thickness_max_mm

        return MeatSlice(
            width_mm=new_width,
            length_mm=new_length,
            thickness_min_mm=new_min,
            thickness_max_mm=new_max,
            meat_type=self.meat_type,
            fat_percentage=self.fat_percentage,
            slice_id=self.slice_id,
            wedge_direction=(self.wedge_direction + rotations) % 4,
            shape_mask=new_mask,
            thickness_map=new_thickness,
            fat_map=new_fat,
            orientation_deg=(self.orientation_deg + angle_deg) % 360,
            resolution_mm=self.resolution_mm,
        )

    @classmethod
    def generate_random(
        cls,
        meat_type: MeatType,
        slice_id: int = 0,
        constraints: Optional[SliceConstraints] = None,
    ) -> "MeatSlice":
        c = constraints or SLICE_CONSTRAINTS
        width = np.random.uniform(c.min_width_mm, c.max_width_mm)
        length = np.random.uniform(c.min_length_mm, c.max_length_mm)
        t_min = np.random.uniform(c.min_thickness_mm, c.max_thickness_mm * 0.5)
        t_max = np.random.uniform(t_min + 2.0, c.max_thickness_mm)
        wedge_dir = np.random.randint(0, 3)

        if meat_type == MeatType.FAT:
            fat_pct = np.random.uniform(80, 95)
        elif meat_type == MeatType.HIGH_QUALITY:
            fat_pct = np.random.uniform(5, 15)
        else:
            fat_pct = np.random.uniform(15, 35)

        return cls(
            width_mm=width,
            length_mm=length,
            thickness_min_mm=t_min,
            thickness_max_mm=t_max,
            meat_type=meat_type,
            fat_percentage=fat_pct,
            slice_id=slice_id,
            wedge_direction=wedge_dir,
        )
