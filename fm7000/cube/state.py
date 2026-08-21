"""Cube state management - 3D voxel grid for tracking fill state."""

from dataclasses import dataclass

import numpy as np

from fm7000.config.constants import CUBE, SLICE_CONSTRAINTS, CubeSpec
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
    - layer_occupancy: cells already covered by the current layer
    - fat_accumulator: fat density accumulated per vertical column
    """

    def __init__(self, spec: CubeSpec | None = None) -> None:
        self.spec = spec or CUBE
        self.w = self.spec.w_voxels
        self.l = self.spec.l_voxels
        self.res = self.spec.resolution_mm

        self.height_map = np.zeros((self.w, self.l), dtype=np.float32)
        self.fat_accumulator = np.zeros((self.w, self.l), dtype=np.float32)
        self.fat_layer_count = np.zeros((self.w, self.l), dtype=np.int32)

        self.layer_occupancy = np.zeros((self.w, self.l), dtype=bool)
        self.layer_floor_map = np.zeros((self.w, self.l), dtype=np.float32)

        self.layers: list[list[PlacedSlice]] = [[]]
        self.current_layer_index: int = 0
        self.total_slices_placed: int = 0
        self.overlap_tolerated: int = 0

    @property
    def current_layer_height_mm(self) -> float:
        return float(np.mean(self.layer_floor_map))

    @property
    def current_layer_coverage(self) -> float:
        """Frazione di area del cubo coperta dallo strato corrente."""
        return float(np.mean(self.layer_occupancy))

    @property
    def fill_percentage(self) -> float:
        return float(np.mean(self.height_map)) / self.spec.height_mm

    @property
    def free_height_mm(self) -> float:
        """Spazio libero sopra il punto piu alto del riempimento."""
        return self.spec.height_mm - float(np.max(self.height_map))

    @property
    def flatness_mm(self) -> float:
        """Deviazione standard della superficie: piu bassa = strati piu piani."""
        return float(np.std(self.height_map))

    @property
    def is_full(self) -> bool:
        if float(np.mean(self.height_map)) >= self.spec.height_mm * 0.95:
            return True
        return self.free_height_mm < SLICE_CONSTRAINTS.min_thickness_mm

    def get_fat_density_at(self, x: int, y: int, w: int, l: int) -> float:
        region_count = self.fat_layer_count[x:x + w, y:y + l]
        if np.sum(region_count) == 0:
            return 0.0
        region_fat = self.fat_accumulator[x:x + w, y:y + l]
        mask = region_count > 0
        return float(np.mean(region_fat[mask] / region_count[mask]))

    def get_fat_column_map(self) -> np.ndarray:
        """Grasso medio per colonna verticale (0-1): indice di colonne di grasso."""
        counts = np.maximum(self.fat_layer_count, 1)
        return self.fat_accumulator / counts

    def overlap_ratio(self, meat_slice: MeatSlice, x: int, y: int) -> float:
        """Frazione della fetta che cadrebbe sopra fette dello stesso strato."""
        sw, sl = meat_slice.shape_mask.shape
        active = meat_slice.shape_mask > 0
        n_active = int(np.sum(active))
        if n_active == 0:
            return 1.0
        occupied = self.layer_occupancy[x:x + sw, y:y + sl]
        return float(np.sum(occupied & active)) / float(n_active)

    def predict_surface(
        self,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> np.ndarray | None:
        """
        Superficie prevista dopo il deposito, modellando la fetta come corpo morbido.

        La fetta si drappeggia sul profilo sottostante (non e' una piastra rigida)
        e dove sporge viene schiacciata: il volume spostato va nei punti bassi,
        quindi la media dello spessore resta costante.
        """
        sw, sl = meat_slice.shape_mask.shape
        if x < 0 or y < 0 or x + sw > self.w or y + sl > self.l:
            return None

        active = meat_slice.shape_mask > 0
        if not np.any(active):
            return None

        floor = self.height_map[x:x + sw, y:y + sl]
        draped = floor + meat_slice.thickness_map

        vals = draped[active]
        target = float(np.mean(vals))
        compliance = float(self.spec.slice_compliance)
        squashed = vals + compliance * (target - vals)
        squashed = np.maximum(squashed, floor[active])

        surface = floor.copy()
        surface[active] = squashed
        return surface

    def layer_step_mm(self, meat_slice: MeatSlice, x: int, y: int) -> float:
        """Dislivello del piano sotto l'impronta: quanto la fetta resterebbe storta."""
        sw, sl = meat_slice.shape_mask.shape
        if x < 0 or y < 0 or x + sw > self.w or y + sl > self.l:
            return float("inf")
        active = meat_slice.shape_mask > 0
        if not np.any(active):
            return float("inf")
        floor = self.height_map[x:x + sw, y:y + sl][active]
        return float(np.max(floor) - np.min(floor))

    def can_place(
        self,
        meat_slice: MeatSlice,
        x: int,
        y: int,
        count_overlap: bool = False,
    ) -> bool:
        """
        Vincoli fisici assoluti: la fetta sta dentro l'impronta del cubo e non
        supera il bordo superiore. Sovrapposizione e dislivello NON bloccano il
        deposito (la carne e' morbida e nessuna fetta va scartata): pesano solo
        nel punteggio della posizione.
        """
        surface = self.predict_surface(meat_slice, x, y)
        if surface is None:
            return False

        if count_overlap and (
            self.overlap_ratio(meat_slice, x, y) > self.spec.max_overlap_ratio
        ):
            self.overlap_tolerated += 1

        active = meat_slice.shape_mask > 0
        return float(np.max(surface[active])) <= self.spec.height_mm

    def place_slice(
        self,
        meat_slice: MeatSlice,
        x: int,
        y: int,
        push_x_mm: float = 0.0,
        push_y_mm: float = 0.0,
    ) -> PlacedSlice | None:
        surface = self.predict_surface(meat_slice, x, y)
        if surface is None or not self.can_place(meat_slice, x, y, count_overlap=True):
            return None

        sw, sl = meat_slice.shape_mask.shape
        active = meat_slice.shape_mask > 0
        floor = self.height_map[x:x + sw, y:y + sl]
        z_base = float(np.min(floor[active]))

        self.height_map[x:x + sw, y:y + sl] = np.maximum(floor, surface)

        active_fat = meat_slice.fat_map * meat_slice.shape_mask
        self.fat_accumulator[x:x + sw, y:y + sl] += active_fat
        self.fat_layer_count[x:x + sw, y:y + sl] += active.astype(np.int32)
        self.layer_occupancy[x:x + sw, y:y + sl] |= active

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
        if self.current_layer_coverage < self.spec.layer_coverage_threshold:
            return False
        self._close_current_layer()
        return True

    def force_advance_layer(self) -> float:
        """Chiude lo strato e restituisce la copertura raggiunta."""
        return self._close_current_layer()

    def _close_current_layer(self) -> float:
        # nessuna pressatura: le fette sono solo appoggiate, la copertura dello
        # strato e' quella reale delle impronte depositate
        coverage = self.current_layer_coverage
        self.current_layer_index += 1
        self.layers.append([])
        self.layer_occupancy[:] = False
        self.layer_floor_map = self.height_map.copy()
        return coverage

    def get_height_at(self, x: int, y: int) -> float:
        if 0 <= x < self.w and 0 <= y < self.l:
            return float(self.height_map[x, y])
        return 0.0

    def get_layer_floor(self) -> float:
        return float(np.mean(self.layer_floor_map))

    def get_state_summary(self) -> dict:
        filled = self.height_map[self.height_map > 0]
        return {
            "current_layer": self.current_layer_index,
            "total_slices": self.total_slices_placed,
            "fill_percentage": self.fill_percentage,
            "layer_coverage": self.current_layer_coverage,
            "mean_height_mm": float(np.mean(self.height_map)),
            "max_height_mm": float(np.max(self.height_map)),
            "min_height_mm": float(np.min(filled)) if filled.size else 0.0,
            "free_height_mm": self.free_height_mm,
            "flatness_mm": self.flatness_mm,
            "overlap_tolerated": self.overlap_tolerated,
            "is_full": self.is_full,
        }

    def reset(self) -> None:
        self.height_map = np.zeros((self.w, self.l), dtype=np.float32)
        self.fat_accumulator = np.zeros((self.w, self.l), dtype=np.float32)
        self.fat_layer_count = np.zeros((self.w, self.l), dtype=np.int32)
        self.layer_occupancy = np.zeros((self.w, self.l), dtype=bool)
        self.layer_floor_map = np.zeros((self.w, self.l), dtype=np.float32)
        self.layers = [[]]
        self.current_layer_index = 0
        self.total_slices_placed = 0
        self.overlap_rejections = 0
