"""Placement engine - perimeter-first strategy, push-to-wall, wedge matching."""

from dataclasses import dataclass, field

import numpy as np

from fm7000.config.constants import (
    CUBE,
    PUSH_TO_WALL,
    ROBOT,
    SEARCH,
    CubeSpec,
    PlacementSearchSpec,
    PlacementZone,
    PushDirection,
    PushToWallSpec,
)
from fm7000.cube.slice_model import MeatSlice
from fm7000.cube.state import CubeState


@dataclass
class PlacementCandidate:
    x: int
    y: int
    zone: PlacementZone
    rotation_deg: float
    push_direction: PushDirection
    push_x_mm: float
    push_y_mm: float
    score: float
    wedge_match_score: float = 0.0
    fat_overlap_score: float = 0.0
    contact_score: float = 0.0
    overlap_ratio: float = 0.0
    prepared_slice: MeatSlice | None = field(default=None, repr=False)


class PlacementEngine:
    """
    Deterministic placement rules for the FM 7000.

    Perimeter-first: the search sweeps every position along the four walls,
    then a coarser interior grid. La mano deve entrare e uscire dal cubo
    perpendicolare alle pareti, e l'asse 4 e' limitato a +/-180 gradi per i
    tubi aria: quindi le rotazioni ammesse sono solo -90, 0, 90, 180.
    L'orientamento fine viene da come la fetta e' girata sul nastro.
    Each candidate is scored on zone priority, contact with
    walls/meat already placed, wedge matching and fat column avoidance.
    """

    def __init__(
        self,
        cube_spec: CubeSpec | None = None,
        push_spec: PushToWallSpec | None = None,
        search_spec: PlacementSearchSpec | None = None,
    ) -> None:
        self.cube = cube_spec or CUBE
        self.push = push_spec or PUSH_TO_WALL
        self.search = search_spec or SEARCH
        self.robot = ROBOT
        self.w = self.cube.w_voxels
        self.l = self.cube.l_voxels

    # ------------------------------------------------------------------ search

    def find_candidates(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
        max_candidates: int | None = None,
    ) -> list[PlacementCandidate]:
        limit = max_candidates or self.search.max_candidates
        candidates: list[PlacementCandidate] = []

        for rotation in self._rotations():
            rotated = meat_slice.rotate(rotation)
            sw, sl = rotated.shape_mask.shape
            for x, y in self._generate_positions(sw, sl):
                candidate = self._evaluate_position(
                    cube_state, rotated, x, y, rotation
                )
                if candidate is not None:
                    candidates.append(candidate)

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:limit]

    def find_best_placement(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
    ) -> PlacementCandidate | None:
        candidates = self.find_candidates(cube_state, meat_slice, max_candidates=1)
        return candidates[0] if candidates else None

    def _rotations(self) -> list[float]:
        """Angoli quadri alle pareti, entro la corsa +/-180 dell'asse 4."""
        step = max(1, int(self.search.rotation_step_deg))
        angles = []
        for a in range(0, 360, step):
            signed = float(a if a <= 180 else a - 360)
            if abs(signed) <= self.robot.wrist_limit_deg:
                angles.append(signed)
        return angles

    def _generate_positions(self, sw: int, sl: int) -> list[tuple[int, int]]:
        if sw > self.w or sl > self.l:
            return []

        x_max = self.w - sw
        y_max = self.l - sl
        step_p = max(1, self.search.perimeter_step_voxels)
        step_i = max(1, self.search.interior_step_voxels)

        positions = set()
        # scorrimento continuo lungo le quattro pareti
        for x in list(range(0, x_max + 1, step_p)) + [x_max]:
            positions.add((x, 0))
            positions.add((x, y_max))
        for y in list(range(0, y_max + 1, step_p)) + [y_max]:
            positions.add((0, y))
            positions.add((x_max, y))
        # griglia interna piu grossolana
        for x in range(0, x_max + 1, step_i):
            for y in range(0, y_max + 1, step_i):
                positions.add((x, y))
        return sorted(positions)

    # -------------------------------------------------------------- evaluation

    def _evaluate_position(
        self,
        cube_state: CubeState,
        rotated_slice: MeatSlice,
        x: int,
        y: int,
        rotation: float,
    ) -> PlacementCandidate | None:
        prepared, px, py, push_x, push_y, push_dir = self._apply_push(
            rotated_slice, x, y
        )
        if not cube_state.can_place(prepared, px, py):
            return None

        zone = self._zone_for(prepared, px, py)
        overlap = cube_state.overlap_ratio(prepared, px, py)
        zone_score = self._zone_priority_score(zone)
        contact = self._contact_score(cube_state, prepared, px, py)
        wedge = self._wedge_match_score(cube_state, prepared, px, py)
        fat = self._fat_overlap_score(cube_state, prepared, px, py)

        # nessuna fetta viene scartata: sovrapposizione e dislivello sotto
        # l'impronta sono penalita, non divieti
        overlap_penalty = overlap
        if overlap > self.cube.max_overlap_ratio:
            # oltre soglia la posizione resta ammessa ma vince solo se non
            # esiste nessun vuoto libero dove appoggiare la fetta
            overlap_penalty += (overlap - self.cube.max_overlap_ratio) * 50.0
        step = cube_state.layer_step_mm(prepared, px, py)
        step_penalty = max(0.0, step - self.cube.max_layer_step_mm) / max(
            self.cube.max_layer_step_mm, 1.0
        )

        # preferenza per le zone basse: tiene lo strato piano invece di
        # costruire torri sopra la carne gia posata
        level = self._level_score(cube_state, prepared, px, py)

        total = (
            zone_score * 2.0
            + contact * 3.0
            + wedge * 2.0
            + fat * 1.5
            + level * 5.0
            - overlap_penalty * 2.0
            - step_penalty * 2.0
        )

        return PlacementCandidate(
            x=px,
            y=py,
            zone=zone,
            rotation_deg=rotation,
            push_direction=push_dir,
            push_x_mm=push_x,
            push_y_mm=push_y,
            score=total,
            wedge_match_score=wedge,
            fat_overlap_score=fat,
            contact_score=contact,
            overlap_ratio=overlap,
            prepared_slice=prepared,
        )

    def _zone_for(self, meat_slice: MeatSlice, x: int, y: int) -> PlacementZone:
        sw, sl = meat_slice.shape_mask.shape
        near = max(1, int(self.push.push_threshold_mm / self.cube.resolution_mm))
        left = x <= near
        right = (self.w - (x + sw)) <= near
        front = y <= near
        back = (self.l - (y + sl)) <= near

        if left and front:
            return PlacementZone.CORNER_TL
        if right and front:
            return PlacementZone.CORNER_TR
        if left and back:
            return PlacementZone.CORNER_BL
        if right and back:
            return PlacementZone.CORNER_BR
        if front:
            return PlacementZone.EDGE_TOP
        if back:
            return PlacementZone.EDGE_BOTTOM
        if left:
            return PlacementZone.EDGE_LEFT
        if right:
            return PlacementZone.EDGE_RIGHT
        return PlacementZone.CENTER

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

    def _contact_score(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> float:
        """Quanto il bordo della fetta appoggia su pareti o carne gia posata."""
        sw, sl = meat_slice.shape_mask.shape
        active = meat_slice.shape_mask > 0

        # finestra locale con 1 cella di bordo: fuori dal cubo = parete (contatto)
        occ = np.ones((sw + 2, sl + 2), dtype=bool)
        x0, x1 = max(0, x - 1), min(self.w, x + sw + 1)
        y0, y1 = max(0, y - 1), min(self.l, y + sl + 1)
        occ[1 + (x0 - x):1 + (x1 - x), 1 + (y0 - y):1 + (y1 - y)] = (
            cube_state.layer_occupancy[x0:x1, y0:y1]
        )

        footprint = np.zeros((sw + 2, sl + 2), dtype=bool)
        footprint[1:1 + sw, 1:1 + sl] = active

        neighbors = np.zeros_like(footprint)
        neighbors[1:, :] |= footprint[:-1, :]
        neighbors[:-1, :] |= footprint[1:, :]
        neighbors[:, 1:] |= footprint[:, :-1]
        neighbors[:, :-1] |= footprint[:, 1:]
        border = neighbors & ~footprint
        n_border = int(np.sum(border))
        if n_border == 0:
            return 0.0
        return float(np.sum(border & occ)) / float(n_border)

    def _level_score(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> float:
        """1 nelle zone basse del cubo, 0 in quelle alte."""
        sw, sl = meat_slice.shape_mask.shape
        active = meat_slice.shape_mask > 0
        floor = cube_state.height_map[x:x + sw, y:y + sl]
        local = float(np.mean(floor[active]))
        mean = float(np.mean(cube_state.height_map))
        spread = float(np.std(cube_state.height_map))
        if spread < 1e-6:
            return 1.0
        return float(np.clip(0.5 + (mean - local) / (3.0 * spread), 0.0, 1.0))

    def _wedge_match_score(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> float:
        """
        Wedge matching: il gradiente di spessore della fetta deve opporsi al
        gradiente dell'altezza sottostante (lato sottile sul punto alto).
        """
        sw, sl = meat_slice.shape_mask.shape
        floor = cube_state.height_map[x:x + sw, y:y + sl]
        active = meat_slice.shape_mask > 0
        if not np.any(active):
            return 0.0

        surface = cube_state.predict_surface(meat_slice, x, y)
        if surface is None:
            return 0.0
        flatness = 1.0 / (1.0 + float(np.std(surface[active])))

        if sw < 2 or sl < 2:
            return flatness

        gh_x, gh_y = np.gradient(floor)
        gt_x, gt_y = np.gradient(meat_slice.thickness_map)
        h_vec = np.array([float(np.sum(gh_x[active])), float(np.sum(gh_y[active]))])
        t_vec = np.array([float(np.sum(gt_x[active])), float(np.sum(gt_y[active]))])

        norm = np.linalg.norm(h_vec) * np.linalg.norm(t_vec)
        if norm < 1e-6:
            return flatness

        cos_sim = float(np.dot(h_vec, t_vec) / norm)
        # gradienti opposti (cos = -1) => cuneo compensa il dislivello
        alignment = (1.0 - cos_sim) / 2.0
        return 0.5 * flatness + 0.5 * alignment

    def _fat_overlap_score(
        self,
        cube_state: CubeState,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> float:
        """Confronto cella per cella: penalizza colonne verticali di grasso."""
        if cube_state.total_slices_placed == 0:
            return 1.0

        sw, sl = meat_slice.shape_mask.shape
        active = meat_slice.shape_mask > 0
        column_fat = cube_state.get_fat_column_map()[x:x + sw, y:y + sl]
        slice_fat = meat_slice.fat_map

        product = column_fat[active] * slice_fat[active]
        if product.size == 0:
            return 1.0

        mean_overlap = float(np.mean(product))
        worst_overlap = float(np.max(product))
        penalty = 0.6 * mean_overlap + 0.4 * worst_overlap
        return float(max(0.0, 1.0 - penalty))

    # ------------------------------------------------------------ push to wall

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

    def _apply_push(
        self,
        meat_slice: MeatSlice,
        x: int,
        y: int,
    ) -> tuple[MeatSlice, int, int, float, float, PushDirection]:
        """
        Spinge la fetta contro le pareti vicine: la posizione va a contatto e il
        perimetro di carne si flette (deformazione a volume costante).
        """
        sw, sl = meat_slice.shape_mask.shape
        res = self.cube.resolution_mm
        gap_left = x * res
        gap_right = (self.w - (x + sw)) * res
        gap_front = y * res
        gap_back = (self.l - (y + sl)) * res

        prepared = meat_slice
        new_x, new_y = x, y
        push_x = push_y = 0.0
        pushes: list[str] = []

        push_x_active = min(gap_left, gap_right) <= self.push.push_threshold_mm
        push_y_active = min(gap_front, gap_back) <= self.push.push_threshold_mm
        is_corner = push_x_active and push_y_active
        compression = (
            self.push.corner_compression_mm if is_corner else self.push.wall_compression_mm
        )

        if push_x_active:
            if gap_left <= gap_right:
                prepared = prepared.flex_against_wall(0, -1, gap_left + compression)
                push_x = -(gap_left + compression)
                new_x = 0
                pushes.append("LEFT")
            else:
                prepared = prepared.flex_against_wall(0, 1, gap_right + compression)
                push_x = gap_right + compression
                new_x = self.w - sw
                pushes.append("RIGHT")

        if push_y_active:
            if gap_front <= gap_back:
                prepared = prepared.flex_against_wall(1, -1, gap_front + compression)
                push_y = -(gap_front + compression)
                new_y = 0
                pushes.append("FRONT")
            else:
                prepared = prepared.flex_against_wall(1, 1, gap_back + compression)
                push_y = gap_back + compression
                new_y = self.l - sl
                pushes.append("BACK")

        direction = self._direction_from_pushes(pushes)
        return prepared, new_x, new_y, push_x, push_y, direction

    def _direction_from_pushes(self, pushes: list[str]) -> PushDirection:
        if not pushes:
            return PushDirection.NONE
        if len(pushes) == 1:
            return PushDirection(pushes[0].lower())
        x_part = "left" if "LEFT" in pushes else "right"
        y_part = "front" if "FRONT" in pushes else "back"
        return PushDirection(f"{x_part}_{y_part}")
