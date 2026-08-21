"""Meat slice model with geometry, thickness map, and fat distribution."""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from fm7000.config.constants import SLICE_CONSTRAINTS, MeatType, SliceConstraints


def _crop_to_footprint(
    mask: np.ndarray, thickness: np.ndarray, fat: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Elimina le righe/colonne vuote lasciate dalla rotazione."""
    rows = np.where(np.any(mask > 0, axis=1))[0]
    cols = np.where(np.any(mask > 0, axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return mask, thickness, fat
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return (
        mask[r0:r1, c0:c1].copy(),
        thickness[r0:r1, c0:c1].copy(),
        fat[r0:r1, c0:c1].copy(),
    )


@dataclass
class MeatSlice:
    """
    Represents a single meat slice with its physical properties.

    Slices are wedge-shaped (non-uniform thickness) and have a fat distribution
    map that must be considered when stacking layers. The slice is a soft body:
    it drapes over the surface below and deforms when pushed against a wall.
    """

    width_mm: float
    length_mm: float
    thickness_min_mm: float = 20.0
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

    @property
    def area_mm2(self) -> float:
        return float(np.sum(self.shape_mask > 0)) * self.resolution_mm ** 2

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

    def rotate(self, angle_deg: float) -> "MeatSlice":
        """Ruota la geometria della fetta; il polso resta quadro alle pareti."""
        angle = float(angle_deg) % 360.0
        if angle < 1e-6:
            return self

        quarters = round(angle / 90.0)
        if abs(angle - quarters * 90.0) < 1e-6:
            k = quarters % 4
            new_mask = np.rot90(self.shape_mask, k).copy()
            new_thickness = np.rot90(self.thickness_map, k).copy()
            new_fat = np.rot90(self.fat_map, k).copy()
        else:
            rot_mask = ndimage.rotate(
                self.shape_mask, angle, order=0, reshape=True, cval=0.0
            )
            new_mask = (rot_mask > 0.5).astype(np.float32)
            new_thickness = ndimage.rotate(
                self.thickness_map, angle, order=1, reshape=True, cval=0.0
            ) * new_mask
            new_fat = np.clip(
                ndimage.rotate(self.fat_map, angle, order=1, reshape=True, cval=0.0),
                0.0,
                1.0,
            ) * new_mask
            new_mask, new_thickness, new_fat = _crop_to_footprint(
                new_mask, new_thickness, new_fat
            )

        active = new_thickness[new_mask > 0]
        return MeatSlice(
            width_mm=new_mask.shape[0] * self.resolution_mm,
            length_mm=new_mask.shape[1] * self.resolution_mm,
            thickness_min_mm=float(active.min()) if active.size else self.thickness_min_mm,
            thickness_max_mm=float(active.max()) if active.size else self.thickness_max_mm,
            meat_type=self.meat_type,
            fat_percentage=self.fat_percentage,
            slice_id=self.slice_id,
            wedge_direction=self.wedge_direction,
            shape_mask=new_mask.astype(np.float32),
            thickness_map=new_thickness.astype(np.float32),
            fat_map=new_fat.astype(np.float32),
            orientation_deg=(self.orientation_deg + angle) % 360.0,
            resolution_mm=self.resolution_mm,
        )

    def flex_against_wall(self, axis: int, sign: int, flex_mm: float) -> "MeatSlice":
        """
        Push-to-wall: il bordo spinto contro la parete si flette.

        La carne nella fascia di contatto si schiaccia e si allarga a riempire
        i vuoti del bordo; il volume della fascia resta costante.
        """
        if flex_mm <= 0.0:
            return self

        n = self.shape_mask.shape[axis]
        band = min(max(1, round(flex_mm / self.resolution_mm)), n)
        sel = slice(0, band) if sign < 0 else slice(n - band, n)
        idx = (sel, slice(None)) if axis == 0 else (slice(None), sel)

        mask = self.shape_mask.copy()
        thickness = self.thickness_map.copy()
        fat = self.fat_map.copy()

        band_mask = mask[idx]
        band_thick = thickness[idx]
        band_fat = fat[idx]
        active = band_mask > 0
        volume_before = float(np.sum(band_thick))
        if volume_before <= 0.0 or not np.any(active):
            return self

        # la carne schiacciata riempie il bordo lungo le linee che hanno materiale
        line_active = np.any(active, axis=axis)
        filled = np.zeros_like(band_mask)
        if axis == 0:
            filled[:, line_active] = 1.0
        else:
            filled[line_active, :] = 1.0

        mean_thick = float(np.mean(band_thick[active]))
        mean_fat = float(np.mean(band_fat[active]))
        new_thick = np.where(active, band_thick, mean_thick) * filled
        new_fat = np.where(active, band_fat, mean_fat) * filled

        volume_after = float(np.sum(new_thick))
        if volume_after > 0.0:
            new_thick = new_thick * (volume_before / volume_after)

        mask[idx] = filled
        thickness[idx] = new_thick
        fat[idx] = new_fat

        active_thick = thickness[mask > 0]
        return MeatSlice(
            width_mm=self.width_mm,
            length_mm=self.length_mm,
            thickness_min_mm=float(active_thick.min()) if active_thick.size else self.thickness_min_mm,
            thickness_max_mm=float(active_thick.max()) if active_thick.size else self.thickness_max_mm,
            meat_type=self.meat_type,
            fat_percentage=self.fat_percentage,
            slice_id=self.slice_id,
            wedge_direction=self.wedge_direction,
            shape_mask=mask.astype(np.float32),
            thickness_map=thickness.astype(np.float32),
            fat_map=fat.astype(np.float32),
            orientation_deg=self.orientation_deg,
            resolution_mm=self.resolution_mm,
        )

    @classmethod
    def generate_random(
        cls,
        meat_type: MeatType,
        slice_id: int = 0,
        constraints: SliceConstraints | None = None,
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
