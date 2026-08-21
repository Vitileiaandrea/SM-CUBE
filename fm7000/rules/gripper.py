"""Gripper pattern selector - determines which vacuum cups to activate."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from fm7000.config.constants import GRIPPER, ROBOT, GripperSpec, PlacementZone
from fm7000.cube.slice_model import MeatSlice


@dataclass
class GripperCommand:
    cup_pattern: np.ndarray
    placement_zone: PlacementZone
    wrist_rotation_deg: float
    vacuum_level: float = 0.8
    meat_margin_x_mm: float = 0.0
    meat_margin_y_mm: float = 0.0
    min_clearance_mm: float = 0.0
    margin_ok: bool = True
    # presa libera: dove cade il centro della griglia rispetto al baricentro
    # della fetta e di quanto la fetta e' girata sotto la pinza dritta
    pick_offset_x_mm: float = 0.0
    pick_offset_y_mm: float = 0.0
    slice_to_gripper_rotation_deg: float = 0.0

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
        slice_angle_deg: float = 0.0,
        pick_shift_x_mm: float = 0.0,
        pick_shift_y_mm: float = 0.0,
    ) -> GripperCommand:
        needed = self.spec.push_safety_margin_mm
        clearances, off_x, off_y = self._anchored_clearances(
            meat_slice, zone, needed, pick_shift_x_mm, pick_shift_y_mm
        )
        pattern = self._max_support_pattern(clearances, needed)
        margin_x, margin_y = self._meat_margins(pattern, meat_slice)
        clearance = self._pattern_clearance(pattern, clearances)

        return GripperCommand(
            cup_pattern=pattern,
            placement_zone=zone,
            wrist_rotation_deg=self._deposit_angle(target_rotation_deg),
            vacuum_level=self._calculate_vacuum_level(meat_slice),
            meat_margin_x_mm=margin_x,
            meat_margin_y_mm=margin_y,
            min_clearance_mm=clearance,
            # il vincolo fisico e' la carne libera sotto ogni ventosa, misurata
            # sul contorno reale: i margini sull'ingombro sono solo informativi
            margin_ok=clearance >= needed,
            pick_offset_x_mm=off_x,
            pick_offset_y_mm=off_y,
            slice_to_gripper_rotation_deg=slice_angle_deg,
        )

    def _deposit_angle(self, rotation_deg: float) -> float:
        """Deposito vincolato: la mano scende quadra alle pareti.

        L'angolo fine della fetta lo da' la presa, che e' libera.
        """
        wrist = 90.0 * round(rotation_deg / 90.0)
        limit = ROBOT.wrist_limit_deg
        return float(max(-limit, min(limit, wrist)))

    def _max_support_pattern(self, clearances: np.ndarray, needed: float) -> np.ndarray:
        """Piu' ventose possibile: si attiva ogni ventosa che appoggia sulla carne
        lasciando `needed` mm di carne libera oltre il labbro, sul contorno reale.

        La griglia 4x4 non e' centrata sul baricentro: viene appoggiata dove
        regge piu' carne, sul lato interno della fetta, cosi' il bordo sporge
        verso spigolo e pareti. Se nessuna ventosa e' valida si tiene comunque
        la migliore: la fetta non va mai scartata.
        """
        pattern = np.zeros((self.rows, self.cols), dtype=np.int8)
        if clearances.size == 0:
            pattern[1, 1] = 1
            return pattern

        cup_radius = self.spec.cup_diameter_mm / 2.0
        valid = clearances >= (cup_radius + needed)
        if np.any(valid):
            pattern[valid] = 1
            return pattern

        i, j = np.unravel_index(int(np.argmax(clearances)), clearances.shape)
        pattern[i, j] = 1
        return pattern

    def _anchored_clearances(
        self,
        meat_slice: MeatSlice,
        zone: PlacementZone,
        needed: float,
        shift_x_mm: float = 0.0,
        shift_y_mm: float = 0.0,
    ) -> tuple[np.ndarray, float, float]:
        """Griglia ventose spostata sul lato interno della fetta.

        La mano ha 210 mm di ingombro in un cubo da 210: non puo' avvicinarsi
        alla parete, quindi allo spigolo la griglia va appoggiata sul lato della
        fetta che guarda il centro del cubo e la carne sporge verso spigolo e
        pareti, dove si flette col push. La presa e' libera: la fetta puo' stare
        girata di qualunque angolo sotto la pinza dritta.

        `shift_x/y_mm` e' lo spostamento che la mano non puo' fare: la griglia
        deve cadere di altrettanto nel verso opposto.

        Ritorna le distanze dal bordo sotto ogni ventosa e l'offset in mm del
        centro griglia rispetto al baricentro della fetta.
        """
        mask = meat_slice.shape_mask
        if mask.size == 0 or not np.any(mask > 0):
            return np.zeros((0, 0)), 0.0, 0.0

        res = meat_slice.resolution_mm
        dist_mm = distance_transform_edt(mask > 0) * res
        cells = np.argwhere(mask > 0)
        ci, cj = cells[:, 0].mean(), cells[:, 1].mean()
        limit = self.spec.cup_diameter_mm / 2.0 + needed
        # verso opposto alla destinazione: la mano resta dentro, la carne sporge
        dir_i, dir_j = self._zone_direction(zone)
        dir_i, dir_j = -dir_i, -dir_j
        target_i, target_j = -float(shift_x_mm), -float(shift_y_mm)
        span_i = max(1, round(mask.shape[0] / 2))
        span_j = max(1, round(mask.shape[1] / 2))
        coarse = max(1, round(10.0 / res))

        best = self._best_offset(
            dist_mm, ci, cj, res, limit,
            (dir_i, dir_j), (target_i, target_j),
            range(-span_i, span_i + 1, coarse),
            range(-span_j, span_j + 1, coarse),
        )
        # raffinamento fine attorno al miglior appoggio trovato
        bi, bj = best[2], best[3]
        best = self._best_offset(
            dist_mm, ci, cj, res, limit,
            (dir_i, dir_j), (target_i, target_j),
            range(bi - coarse, bi + coarse + 1),
            range(bj - coarse, bj + coarse + 1),
            current=best,
        )
        return best[1], float(best[2] * res), float(best[3] * res)

    def _best_offset(
        self,
        dist_mm: np.ndarray,
        ci: float,
        cj: float,
        res: float,
        limit: float,
        direction: tuple[float, float],
        target: tuple[float, float],
        range_i: range,
        range_j: range,
        current: tuple[float, np.ndarray, int, int] | None = None,
    ) -> tuple[float, np.ndarray, int, int]:
        """Offset griglia migliore: piu' ventose valide, poi appoggio piu' saldo."""
        dir_i, dir_j = direction
        target_i, target_j = target
        best = current
        for si in range_i:
            for sj in range_j:
                grid = self._grid_clearances(dist_mm, ci + si, cj + sj, res)
                valid = grid >= limit
                cups = int(np.sum(valid))
                # a pari ventose vince la presa piu' interna alla carne
                hold = float(np.sum(grid[valid])) if cups else float(np.max(grid))
                pull = (si * dir_i + sj * dir_j) * res
                miss = abs(si * res - target_i) + abs(sj * res - target_j)
                score = cups * 1000.0 + hold * 0.5 + pull - miss * 2.0
                if best is None or score > best[0]:
                    best = (score, grid, si, sj)
        assert best is not None
        return best

    def _grid_clearances(
        self, dist_mm: np.ndarray, ci: float, cj: float, res: float
    ) -> np.ndarray:
        """Distanza dal bordo del contorno sotto ogni ventosa della griglia 4x4."""
        spacing = self.spec.cup_spacing_mm
        center = (self.rows - 1) / 2.0
        out = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                i = round(ci + (r - center) * spacing / res)
                j = round(cj + (c - center) * spacing / res)
                if 0 <= i < dist_mm.shape[0] and 0 <= j < dist_mm.shape[1]:
                    out[r, c] = float(dist_mm[i, j])
        return out

    def _zone_direction(self, zone: PlacementZone) -> tuple[float, float]:
        """Verso dello spigolo/parete di destinazione (la griglia va all'opposto)."""
        directions = {
            PlacementZone.CORNER_TL: (-1.0, -1.0),
            PlacementZone.CORNER_TR: (1.0, -1.0),
            PlacementZone.CORNER_BL: (-1.0, 1.0),
            PlacementZone.CORNER_BR: (1.0, 1.0),
            PlacementZone.EDGE_LEFT: (-1.0, 0.0),
            PlacementZone.EDGE_RIGHT: (1.0, 0.0),
            PlacementZone.EDGE_TOP: (0.0, -1.0),
            PlacementZone.EDGE_BOTTOM: (0.0, 1.0),
        }
        return directions.get(zone, (0.0, 0.0))

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

    def _pattern_clearance(self, pattern: np.ndarray, clearances: np.ndarray) -> float:
        """Carne libera oltre il labbro della ventosa piu' sfavorita del pattern."""
        if clearances.size == 0 or not np.any(pattern > 0):
            return 0.0
        cup_radius = self.spec.cup_diameter_mm / 2.0
        worst = float(np.min(clearances[pattern > 0])) - cup_radius
        return max(worst, 0.0)

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
