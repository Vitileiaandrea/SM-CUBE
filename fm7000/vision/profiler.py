"""Laser profiler interface for 3D slice geometry and fat distribution mapping."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from fm7000.config.constants import SLICE_CONSTRAINTS, SliceConstraints
from fm7000.cube.slice_model import MeatSlice
from fm7000.config.constants import MeatType
from fm7000.vision.camera import SliceDetection


@dataclass
class ProfileScan:
    slice_id: int
    width_mm: float
    length_mm: float
    thickness_min_mm: float
    thickness_max_mm: float
    thickness_map: np.ndarray
    fat_map: np.ndarray
    shape_mask: np.ndarray
    wedge_direction: int
    volume_mm3: float
    resolution_mm: float = 5.0


class ProfilerInterface:
    """
    Interface to the laser profiler for measuring slice 3D geometry.

    The profiler scans each slice to produce:
    - A 2D thickness map (non-uniform, wedge-shaped slices)
    - A 2D fat distribution map (intrinsic fat within the slice)
    - Shape mask (actual outline of the irregular slice)
    - Wedge direction (which way the thickness gradient goes)

    In production, communicates with the profiler hardware.
    For simulation, generates synthetic profiles.
    """

    def __init__(self, resolution_mm: float = 5.0, simulation: bool = True) -> None:
        self.resolution_mm = resolution_mm
        self.simulation = simulation
        self._connected = False

    def connect(self) -> bool:
        if self.simulation:
            self._connected = True
            return True
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def scan_slice(self, detection: SliceDetection) -> Optional[ProfileScan]:
        if not self._connected:
            return None

        if self.simulation:
            return self._simulate_scan(detection)

        return None

    def scan_to_meat_slice(self, detection: SliceDetection) -> Optional[MeatSlice]:
        scan = self.scan_slice(detection)
        if scan is None:
            return None

        return MeatSlice(
            width_mm=scan.width_mm,
            length_mm=scan.length_mm,
            thickness_min_mm=scan.thickness_min_mm,
            thickness_max_mm=scan.thickness_max_mm,
            meat_type=detection.meat_type,
            fat_percentage=detection.fat_percentage,
            slice_id=scan.slice_id,
            wedge_direction=scan.wedge_direction,
            shape_mask=scan.shape_mask,
            thickness_map=scan.thickness_map,
            fat_map=scan.fat_map,
            orientation_deg=detection.orientation_deg,
            resolution_mm=scan.resolution_mm,
        )

    def _simulate_scan(self, detection: SliceDetection) -> ProfileScan:
        w_vox = max(1, int(detection.width_mm / self.resolution_mm))
        l_vox = max(1, int(detection.length_mm / self.resolution_mm))

        shape_mask = self._generate_irregular_shape(w_vox, l_vox)

        t_min = np.random.uniform(20, 30)
        t_max = np.random.uniform(t_min + 3, 40)
        wedge_dir = np.random.randint(0, 3)
        thickness_map = self._generate_thickness_map(
            w_vox, l_vox, t_min, t_max, wedge_dir, shape_mask
        )

        fat_map = self._generate_fat_map(
            w_vox, l_vox, detection.fat_percentage, detection.meat_type, shape_mask
        )

        volume = float(np.sum(thickness_map) * self.resolution_mm ** 2)

        return ProfileScan(
            slice_id=detection.slice_id,
            width_mm=detection.width_mm,
            length_mm=detection.length_mm,
            thickness_min_mm=t_min,
            thickness_max_mm=t_max,
            thickness_map=thickness_map,
            fat_map=fat_map,
            shape_mask=shape_mask,
            wedge_direction=wedge_dir,
            volume_mm3=volume,
            resolution_mm=self.resolution_mm,
        )

    def _generate_irregular_shape(self, w: int, l: int) -> np.ndarray:
        mask = np.ones((w, l), dtype=np.float32)
        center_x, center_y = w / 2.0, l / 2.0
        for i in range(w):
            for j in range(l):
                dx = abs(i - center_x) / max(center_x, 1)
                dy = abs(j - center_y) / max(center_y, 1)
                edge_dist = max(dx, dy)
                if edge_dist > 0.75:
                    if np.random.random() < 0.3 * (edge_dist - 0.75) / 0.25:
                        mask[i, j] = 0
        return mask

    def _generate_thickness_map(
        self,
        w: int,
        l: int,
        t_min: float,
        t_max: float,
        wedge_dir: int,
        mask: np.ndarray,
    ) -> np.ndarray:
        if wedge_dir == 0:
            gradient = np.linspace(0, 1, w, dtype=np.float32)[:, np.newaxis]
            gradient = np.broadcast_to(gradient, (w, l)).copy()
        elif wedge_dir == 1:
            gradient = np.linspace(0, 1, l, dtype=np.float32)[np.newaxis, :]
            gradient = np.broadcast_to(gradient, (w, l)).copy()
        else:
            gx = np.linspace(0, 1, w, dtype=np.float32)[:, np.newaxis]
            gy = np.linspace(0, 1, l, dtype=np.float32)[np.newaxis, :]
            gradient = ((gx + gy) / 2.0).astype(np.float32)

        noise = np.random.uniform(-0.05, 0.05, (w, l)).astype(np.float32)
        gradient = np.clip(gradient + noise, 0, 1)

        thickness = t_min + gradient * (t_max - t_min)
        return (thickness * mask).astype(np.float32)

    def _generate_fat_map(
        self,
        w: int,
        l: int,
        fat_pct: float,
        meat_type: MeatType,
        mask: np.ndarray,
    ) -> np.ndarray:
        if meat_type == MeatType.FAT:
            base = np.random.uniform(0.7, 0.95, (w, l)).astype(np.float32)
        else:
            base_val = fat_pct / 100.0 if fat_pct > 0 else 0.15
            base = np.random.uniform(
                base_val * 0.3, base_val * 1.7, (w, l)
            ).astype(np.float32)

            num_clusters = np.random.randint(1, 4)
            for _ in range(num_clusters):
                cx = np.random.randint(0, w)
                cy = np.random.randint(0, l)
                radius = np.random.randint(2, max(3, min(w, l) // 3))
                for i in range(max(0, cx - radius), min(w, cx + radius)):
                    for j in range(max(0, cy - radius), min(l, cy + radius)):
                        dist = ((i - cx) ** 2 + (j - cy) ** 2) ** 0.5
                        if dist < radius:
                            boost = 0.3 * (1 - dist / radius)
                            base[i, j] += boost

        base = np.clip(base, 0, 1)
        return (base * mask).astype(np.float32)
