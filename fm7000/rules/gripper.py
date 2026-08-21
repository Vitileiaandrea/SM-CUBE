"""Gripper pattern selector - determines which vacuum cups to activate."""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

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
    # linea di ventose del perimetro mano usata per allineare la fetta e
    # parete su cui spingere per prima ('riga' o 'colonna' della griglia)
    align_line: str = ""
    support_cups: int = 0
    # quanto la carne sporge oltre il perimetro esterno delle ventose attive:
    # oltre i 10 mm la fetta non si allinea alle pareti
    overhang_mm: float = 0.0
    overhang_ok: bool = True
    # frazione di labbro appoggiata sulla carne nella peggiore ventosa attiva
    min_coverage: float = 1.0
    # ventosa primaria: quella sullo spigolo della fetta che guida l'appoggio
    primary_cup: tuple[int, int] | None = None

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
        clearances, coverage, off_x, off_y = self._anchored_clearances(
            meat_slice, zone, needed, pick_shift_x_mm, pick_shift_y_mm
        )
        dir_i, dir_j = self._zone_direction(zone)
        pattern = self._max_support_pattern(
            clearances, coverage, needed, self._edge_cups(dir_i, dir_j)
        )
        align_line = self._align_line(pattern, dir_i, dir_j)
        margin_x, margin_y = self._meat_margins(pattern, meat_slice)
        # il vincolo dei 10 mm vale sulla ventosa primaria, quella che porta la
        # fetta a parete: le altre tirano se coprono il foro centrale
        edge = self._edge_cups(dir_i, dir_j)
        primary = self._primary_cup(pattern, clearances, needed, dir_i, dir_j)
        clearance = self._primary_clearance(pattern, edge, clearances, primary)
        overhang = self._overhang_mm(meat_slice, pattern, off_x, off_y)

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
            align_line=align_line,
            support_cups=int(np.sum(pattern)),
            primary_cup=primary,
            overhang_mm=overhang,
            min_coverage=self._min_coverage(pattern, coverage),
            # tolleranza di una cella raster sul contorno discretizzato
            overhang_ok=overhang <= needed + meat_slice.resolution_mm,
        )

    def _deposit_angle(self, rotation_deg: float) -> float:
        """Deposito vincolato: la mano scende quadra alle pareti.

        L'angolo fine della fetta lo da' la presa, che e' libera.
        """
        wrist = 90.0 * round(rotation_deg / 90.0)
        limit = ROBOT.wrist_limit_deg
        return float(max(-limit, min(limit, wrist)))

    def _max_support_pattern(
        self,
        clearances: np.ndarray,
        coverage: np.ndarray,
        needed: float,
        edge_cups: np.ndarray,
    ) -> np.ndarray:
        """Ventose attive: solo il perimetro della griglia, labbro dentro.

        Si usano solo le ventose dell'anello esterno della griglia 4x4: sono
        loro che tengono il bordo della fetta e la spingono a parete, le
        quattro interne lascerebbero afflosciare il perimetro. Una ventosa si
        attiva solo col labbro rientrato di `needed` mm dal bordo della carne.
        Se nessuna ci arriva si tiene quella che appoggia meglio dell'anello:
        la fetta non va mai scartata.
        """
        pattern = np.zeros((self.rows, self.cols), dtype=np.int8)
        ring = self._ring_cups()
        if clearances.size == 0:
            pattern[0, 0] = 1
            return pattern

        cup_radius = self.spec.cup_diameter_mm / 2.0
        full = clearances >= (cup_radius + needed)
        valid = full & ring
        if np.any(valid):
            pattern[valid] = 1
            return pattern

        score = np.where(ring, coverage, -1.0)
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        pattern[i, j] = 1
        return pattern

    def _ring_cups(self) -> np.ndarray:
        """Anello esterno della griglia: le uniche ventose che si usano."""
        ring = np.ones((self.rows, self.cols), dtype=bool)
        if self.rows > 2 and self.cols > 2:
            ring[1:-1, 1:-1] = False
        return ring

    def _min_coverage(
        self, pattern: np.ndarray, coverage: np.ndarray
    ) -> float:
        """Labbro appoggiato sulla carne nella peggiore ventosa attiva."""
        if coverage.size == 0 or not np.any(pattern > 0):
            return 0.0
        return float(np.min(coverage[pattern > 0]))

    def _primary_cup(
        self,
        pattern: np.ndarray,
        clearances: np.ndarray,
        needed: float,
        dir_i: float,
        dir_j: float,
    ) -> tuple[int, int] | None:
        """Ventosa primaria: quella che comanda l'appoggio allo spigolo.

        E' la ventosa attiva piu' avanzata verso spigolo/parete tra quelle col
        labbro rientrato di `needed` mm dal bordo della carne. Sulle fette piu'
        piccole della mano non e' la ventosa d'angolo della griglia (che
        cadrebbe fuori dalla carne) ma la prima che la carne copre a regola:
        e' lei che porta la fetta a parete, le altre seguono.
        """
        if clearances.size == 0:
            return None
        cup_radius = self.spec.cup_diameter_mm / 2.0
        ok = (pattern > 0) & (clearances >= cup_radius + needed)
        if not np.any(ok):
            return None
        cells = np.argwhere(ok)
        proj = cells[:, 0] * dir_i + cells[:, 1] * dir_j
        best = cells[int(np.argmax(proj))]
        return int(best[0]), int(best[1])

    def _primary_clearance(
        self,
        pattern: np.ndarray,
        edge_cups: np.ndarray,
        clearances: np.ndarray,
        primary: tuple[int, int] | None,
    ) -> float:
        """Carne oltre il labbro su primaria e perimetro di destinazione.

        Comanda la primaria, ma tutte le ventose del perimetro attive devono
        stare `needed` mm dentro il bordo della carne: conta la peggiore. Senza
        primaria valida la presa non allinea, vale zero e il piano viene
        scartato a monte.
        """
        if clearances.size == 0 or primary is None:
            return 0.0
        cup_radius = self.spec.cup_diameter_mm / 2.0
        pool = (pattern > 0) & (edge_cups > 0)
        pool[primary] = True
        return max(float(np.min(clearances[pool])) - cup_radius, 0.0)

    def _align_line(
        self, pattern: np.ndarray, dir_i: float, dir_j: float
    ) -> str:
        """Linea di ventose del perimetro mano su cui si allinea la fetta.

        Sullo spigolo le pareti candidate sono due: si spinge per prima quella
        la cui fila esterna di ventose regge piu' carne, cosi' il bordo arriva
        allineato e sostenuto per tutta la sua lunghezza.
        """
        row = self.rows - 1 if dir_i > 0 else 0
        col = self.cols - 1 if dir_j > 0 else 0
        n_row = int(np.sum(pattern[row, :])) if abs(dir_i) > 0.5 else -1
        n_col = int(np.sum(pattern[:, col])) if abs(dir_j) > 0.5 else -1
        if n_row < 0 and n_col < 0:
            return ""
        if n_row >= n_col:
            return f"riga {row} ({n_row} ventose)"
        return f"colonna {col} ({n_col} ventose)"

    def _anchored_clearances(
        self,
        meat_slice: MeatSlice,
        zone: PlacementZone,
        needed: float,
        shift_x_mm: float = 0.0,
        shift_y_mm: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        """Griglia ventose ancorata sul lato della fetta che va appoggiato.

        La fetta si prende dal lato che va contro spigolo/parete: le ventose
        arrivano fin sul bordo che sporge (tenendo `needed` mm di carne libera
        oltre il labbro) altrimenti quel lembo si affloscia e il push non lo
        spinge. La presa e' libera: la fetta puo' stare girata di qualunque
        angolo sotto la pinza dritta.

        `shift_x/y_mm` e' lo spostamento che la mano non puo' fare (ingombro
        210 mm): la griglia si sposta di altrettanto verso quel lato.

        Ritorna le distanze dal bordo sotto ogni ventosa e l'offset in mm del
        centro griglia rispetto al baricentro della fetta.
        """
        mask = meat_slice.shape_mask
        if mask.size == 0 or not np.any(mask > 0):
            return np.zeros((0, 0)), np.zeros((0, 0)), 0.0, 0.0

        res = meat_slice.resolution_mm
        dist_mm = distance_transform_edt(mask > 0) * res
        cells = np.argwhere(mask > 0)
        ci, cj = cells[:, 0].mean(), cells[:, 1].mean()
        # una cella di guardia: la maschera e' discretizzata a `res` mm e senza
        # margine il vincolo dei 10 mm cade sotto sul contorno reale
        limit = self.spec.cup_diameter_mm / 2.0 + needed + res
        # verso il lato di appoggio: le ventose sostengono il bordo che spinge
        dir_i, dir_j = self._zone_direction(zone)
        target_i, target_j = float(shift_x_mm), float(shift_y_mm)
        span_i = max(1, round(mask.shape[0] / 2))
        span_j = max(1, round(mask.shape[1] / 2))

        # quanto sporge la carne nel verso della parete: le ventose devono
        # arrivare fin qui, meno i `needed` mm di carne libera
        edge_proj = float(
            np.max(cells[:, 0] * dir_i + cells[:, 1] * dir_j)
        ) * res

        # ancoraggio geometrico: la ventosa del perimetro si mette nel punto
        # piu' vicino allo spigolo/parete che ha ancora `needed` mm di carne
        # perpendicolari al contorno su tutti i lati
        anchor = self._anchor_offset(dist_mm, ci, cj, res, limit, dir_i, dir_j)
        # la fetta deve svilupparsi dentro l'ingombro delle ventose: se sporge
        # piu' di `needed` mm in qualunque direzione quel lembo resta senza
        # sostegno e la fetta non si allinea alle pareti
        lo_i, hi_i, lo_j, hi_j = self._containment_range(
            mask > 0, ci, cj, res, needed
        )
        free_i = range(max(-span_i, lo_i), min(span_i, hi_i) + 1)
        free_j = range(max(-span_j, lo_j), min(span_j, hi_j) + 1)
        range_i, range_j = free_i, free_j
        if anchor is not None:
            # l'ancoraggio non blocca l'offset: diventa la posizione preferita.
            # Cosi' la ricerca puo' scorrere lungo la linea di ventose e
            # portarne dentro la carne piu' di una, che e' quello che tiene il
            # bordo su tutta la parete
            ai = int(min(max(anchor[0], lo_i), hi_i))
            aj = int(min(max(anchor[1], lo_j), hi_j))
            target_i, target_j = ai * res, aj * res
        if not range_i:
            range_i = range(lo_i, lo_i + 1)
        if not range_j:
            range_j = range(lo_j, lo_j + 1)

        best = self._best_offset(
            dist_mm, ci, cj, res, limit,
            (dir_i, dir_j), (target_i, target_j), edge_proj,
            range_i, range_j,
            mask=mask > 0,
            wanted=limit,
        )
        # il contenimento della coda dentro l'ingombro non deve costare
        # ventose del perimetro: se scorrendo la griglia su tutto lo span la
        # fila ne porta dentro la carne piu' di quante ne tiene l'offset
        # contenuto, vince la presa con piu' ventose - e' quella che tiene il
        # bordo su tutta la parete invece di lasciarlo afflosciare
        span = self._best_offset(
            dist_mm, ci, cj, res, limit,
            (dir_i, dir_j), (target_i, target_j), edge_proj,
            range(-span_i, span_i + 1), range(-span_j, span_j + 1),
            mask=mask > 0,
            wanted=limit,
        )
        if self._ring_valid(span[1], limit) > self._ring_valid(best[1], limit):
            best = span
        coverage = self._grid_coverage(
            mask > 0, ci + best[2], cj + best[3], res
        )
        return best[1], coverage, float(best[2] * res), float(best[3] * res)

    def _ring_valid(self, clearances: np.ndarray, limit: float) -> int:
        """Ventose dell'anello col labbro dentro la carne a regola."""
        if clearances.size == 0:
            return 0
        return int(np.sum((clearances >= limit) & self._ring_cups()))

    def _edge_full(
        self,
        clearances: np.ndarray,
        dir_i: float,
        dir_j: float,
        needed: float,
    ) -> bool:
        """Almeno una ventosa del perimetro con il labbro dentro la carne."""
        if clearances.size == 0:
            return False
        edge = self._edge_cups(dir_i, dir_j)
        if not np.any(edge > 0):
            return True
        full = clearances >= (self.spec.cup_diameter_mm / 2.0 + needed)
        return bool(np.any(full & (edge > 0)))

    def _containment_range(
        self,
        mask: np.ndarray,
        ci: float,
        cj: float,
        res: float,
        needed: float,
    ) -> tuple[int, int, int, int]:
        """Offset griglia ammessi perche' la fetta resti dentro la mano.

        La carne puo' sporgere al massimo `needed` mm oltre il perimetro
        esterno delle ventose, su ogni direzione. Se la fetta e' piu' grande
        dell'ingombro il contenimento e' impossibile: si centra la griglia,
        che minimizza la sporgenza.
        """
        cells = np.argwhere(mask)
        i_min, i_max = cells[:, 0].min(), cells[:, 0].max()
        j_min, j_max = cells[:, 1].min(), cells[:, 1].max()
        center = (self.rows - 1) / 2.0
        half = (
            center * self.spec.cup_spacing_mm
            + self.spec.cup_diameter_mm / 2.0
            + needed
        ) / res

        lo_i, hi_i = i_max - half - ci, i_min + half - ci
        lo_j, hi_j = j_max - half - cj, j_min + half - cj
        if lo_i > hi_i:
            mid = (i_min + i_max) / 2.0 - ci
            lo_i = hi_i = mid
        if lo_j > hi_j:
            mid = (j_min + j_max) / 2.0 - cj
            lo_j = hi_j = mid
        return (
            int(np.ceil(lo_i)),
            int(np.floor(hi_i)),
            int(np.ceil(lo_j)),
            int(np.floor(hi_j)),
        )

    def _anchor_offset(
        self,
        dist_mm: np.ndarray,
        ci: float,
        cj: float,
        res: float,
        limit: float,
        dir_i: float,
        dir_j: float,
    ) -> tuple[int, int] | None:
        """Offset che porta la ventosa del perimetro sul punto giusto.

        Il punto giusto e' il piu' avanzato verso spigolo/parete tra quelli che
        hanno ancora `needed` mm di carne perpendicolari al contorno su ogni
        lato: sullo spigolo e' il punto sulla bisettrice con 10 mm di carne su
        entrambi i lati, esattamente come da disegno.
        """
        if abs(dir_i) < 0.5 and abs(dir_j) < 0.5:
            return None
        cells = np.argwhere(dist_mm >= limit)
        if cells.size == 0:
            return None

        proj = cells[:, 0] * dir_i + cells[:, 1] * dir_j
        top = cells[proj >= proj.max() - 1e-9]
        # a pari avanzamento si sta al centro della fetta sull'altro asse
        lateral = np.abs(top[:, 0] - ci) if abs(dir_i) < 0.5 else np.abs(top[:, 1] - cj)
        anchor = top[int(np.argmin(lateral))]

        spacing = self.spec.cup_spacing_mm / res
        center = (self.rows - 1) / 2.0
        row = self.rows - 1 if dir_i > 0 else 0
        col = self.cols - 1 if dir_j > 0 else 0
        si = anchor[0] - (ci + (row - center) * spacing) if abs(dir_i) > 0.5 else 0.0
        sj = anchor[1] - (cj + (col - center) * spacing) if abs(dir_j) > 0.5 else 0.0
        return round(si), round(sj)

    def _best_offset(
        self,
        dist_mm: np.ndarray,
        ci: float,
        cj: float,
        res: float,
        limit: float,
        direction: tuple[float, float],
        target: tuple[float, float],
        edge_proj: float,
        range_i: range,
        range_j: range,
        current: tuple[float, np.ndarray, int, int] | None = None,
        mask: np.ndarray | None = None,
        wanted: float = 0.0,
    ) -> tuple[float, np.ndarray, int, int]:
        """Offset griglia migliore: piu' ventose valide, poi appoggio piu' saldo.

        A pari numero di ventose vince la presa che porta la ventosa attiva
        piu' esterna vicino al bordo che va contro parete/spigolo: quel lembo
        deve arrivare sostenuto, altrimenti si affloscia e il push non spinge.
        """
        dir_i, dir_j = direction
        target_i, target_j = target
        spacing = self.spec.cup_spacing_mm
        center = (self.rows - 1) / 2.0
        rr, cc = np.meshgrid(
            np.arange(self.rows) - center,
            np.arange(self.cols) - center,
            indexing="ij",
        )
        edge_cups = self._edge_cups(dir_i, dir_j)
        corner = abs(dir_i) > 0.5 and abs(dir_j) > 0.5
        row = self.rows - 1 if dir_i > 0 else 0
        col = self.cols - 1 if dir_j > 0 else 0
        best = current
        for si in range_i:
            for sj in range_j:
                grid = self._grid_clearances(dist_mm, ci + si, cj + sj, res)
                # contano solo le ventose dell'anello: le interne non si usano
                valid = (grid >= limit) & self._ring_cups()
                cups = int(np.sum(valid))
                base = ((ci + si) * dir_i + (cj + sj) * dir_j) * res
                proj = base + (rr * dir_i + cc * dir_j) * spacing
                if cups:
                    # distanza tra la ventosa piu' esterna e il bordo di spinta
                    gap = edge_proj - float(np.max(proj[valid]))
                    hold = float(np.sum(grid[valid]))
                else:
                    gap = edge_proj
                    hold = float(np.max(grid))
                # le ventose del perimetro della griglia sul lato di
                # destinazione devono prendere la carne: sono quelle che
                # piazzano la fetta a parete/spigolo, le interne solo sostegno
                on_edge = float(np.sum(edge_cups[valid]))
                # spigolo: la ventosa d'angolo va tenuta a ~10 mm di carne su
                # entrambi i lati, cosi' prende il piu' vicino possibile al
                # perimetro invece di stare inutilmente dentro
                excess = 0.0
                if corner and valid[row, col]:
                    # carne perpendicolare al contorno sotto la ventosa
                    # d'angolo: va tenuta ai 10 mm, non di piu'
                    excess = abs(grid[row, col] - wanted)
                miss = abs(si * res - target_i) + abs(sj * res - target_j)
                # comanda il numero di ventose del perimetro che prendono
                # carne a regola: una sola ventosa lascia afflosciare il bordo
                score = (
                    cups * 20000.0
                    + on_edge * 9000.0
                    - gap * 6.0
                    - excess * 400.0
                    + hold * 0.2
                    - miss * 2.0
                )
                if best is None or score > best[0]:
                    best = (score, grid, si, sj)
        assert best is not None
        return best

    @staticmethod
    def _ray_dist_mm(
        mask: np.ndarray, ci: float, cj: float, di: float, dj: float, res: float
    ) -> float:
        """Carne davanti al centro ventosa lungo una direzione, in mm."""
        steps = 0
        while True:
            i = round(ci + di * steps)
            j = round(cj + dj * steps)
            if not (0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]):
                break
            if not mask[i, j]:
                break
            steps += 1
            if steps > max(mask.shape):
                break
        return steps * res

    def _edge_cups(self, dir_i: float, dir_j: float) -> np.ndarray:
        """Ventose del perimetro della griglia sul lato di destinazione.

        Per una parete e' la fila esterna verso quella parete, per uno spigolo
        la ventosa d'angolo piu' le due file che formano lo spigolo. Sono
        queste che devono prendere la carne: con loro la fetta si piazza a
        parete, le interne servono solo a sostenere il resto della fetta e a
        riempire il centro quando i bordi sono chiusi.
        """
        weights = np.zeros((self.rows, self.cols))
        row = self.rows - 1 if dir_i > 0 else 0
        col = self.cols - 1 if dir_j > 0 else 0
        if abs(dir_i) > 0.5:
            weights[row, :] = 1.0
        if abs(dir_j) > 0.5:
            weights[:, col] = 1.0
        if abs(dir_i) > 0.5 and abs(dir_j) > 0.5:
            # spigolo: la ventosa d'angolo e' quella che conta di piu'
            weights[row, col] = 3.0
        if not weights.any():
            # centro: nessun lato imposto, va bene qualunque ventosa
            weights[:] = 1.0
        return weights

    def _grid_clearances(
        self, dist_mm: np.ndarray, ci: float, cj: float, res: float
    ) -> np.ndarray:
        """Distanza dal bordo del contorno sotto ogni ventosa della griglia 4x4."""
        spacing = self.spec.cup_spacing_mm
        center = (self.rows - 1) / 2.0
        idx = np.arange(self.rows) - center
        jdx = np.arange(self.cols) - center
        ii = ci + idx[:, None] * spacing / res + np.zeros((1, self.cols))
        jj = cj + jdx[None, :] * spacing / res + np.zeros((self.rows, 1))
        # interpolazione: il centro ventosa cade tra le celle da 5 mm e
        # arrotondarlo sposta la clearance di mezza cella, cioe' del margine
        return map_coordinates(
            dist_mm, [ii.ravel(), jj.ravel()], order=1, mode="constant", cval=0.0
        ).reshape(self.rows, self.cols)

    def _overhang_mm(
        self,
        meat_slice: MeatSlice,
        pattern: np.ndarray,
        off_x_mm: float,
        off_y_mm: float,
    ) -> float:
        """Quanto la carne sporge oltre l'ingombro delle ventose attive.

        La fetta deve svilupparsi dentro il perimetro esterno delle ventose:
        se sporge piu' dei 10 mm del push, sui lati o sugli spigoli, quel lembo
        resta senza sostegno e non si allinea alla parete.
        """
        mask = meat_slice.shape_mask > 0
        if not np.any(mask) or not np.any(pattern > 0):
            return 0.0

        res = meat_slice.resolution_mm
        cells = np.argwhere(mask)
        ci = cells[:, 0].mean() + off_x_mm / res
        cj = cells[:, 1].mean() + off_y_mm / res
        spacing = self.spec.cup_spacing_mm / res
        center = (self.rows - 1) / 2.0
        rad = self.spec.cup_diameter_mm / 2.0 / res
        # perimetro esterno della mano: ventose d'angolo della griglia + labbro
        i0 = ci - center * spacing - rad
        i1 = ci + center * spacing + rad
        j0 = cj - center * spacing - rad
        j1 = cj + center * spacing + rad
        di = np.maximum(0.0, np.maximum(i0 - cells[:, 0], cells[:, 0] - i1))
        dj = np.maximum(0.0, np.maximum(j0 - cells[:, 1], cells[:, 1] - j1))
        return float(np.max(np.hypot(di, dj)) * res)

    def _grid_coverage(
        self, mask: np.ndarray, ci: float, cj: float, res: float
    ) -> np.ndarray:
        """Frazione del labbro di ogni ventosa che appoggia sulla carne."""
        spacing = self.spec.cup_spacing_mm
        center = (self.rows - 1) / 2.0
        rad = self.spec.cup_diameter_mm / 2.0 / res
        span = int(np.ceil(rad))
        di, dj = np.meshgrid(
            np.arange(-span, span + 1), np.arange(-span, span + 1), indexing="ij"
        )
        disc = (di**2 + dj**2) <= rad**2
        out = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                i = round(ci + (r - center) * spacing / res)
                j = round(cj + (c - center) * spacing / res)
                ii = i + di[disc]
                jj = j + dj[disc]
                ok = (
                    (ii >= 0)
                    & (ii < mask.shape[0])
                    & (jj >= 0)
                    & (jj < mask.shape[1])
                )
                if not np.any(ok):
                    continue
                out[r, c] = float(np.sum(mask[ii[ok], jj[ok]])) / float(disc.sum())
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

    def cup_layout_on_slice_mm(
        self,
        offset_x_mm: float,
        offset_y_mm: float,
        rotation_deg: float,
        meat_slice: MeatSlice | None = None,
    ) -> list[tuple[int, int, float, float]]:
        """Centri delle 16 ventose sulla fetta non ruotata, in mm dal centro.

        La presa si calcola sulla fetta girata di `rotation_deg`: qui si torna
        nel frame della fetta come arriva sul nastro, per poter disegnare le
        ventose sul contorno reale. Ritorna (riga, colonna, dx_mm, dy_mm) dove
        x e y seguono gli assi della mappa della fetta (asse 0 = x).

        Gli offset della presa sono riferiti al baricentro della carne: con
        `meat_slice` si riportano al centro dell'ingombro, che e' il
        riferimento del contorno misurato.
        """
        theta = np.radians(-float(rotation_deg))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        center = (self.rows - 1) / 2.0
        bias_x, bias_y = self._centroid_bias_mm(meat_slice)
        out: list[tuple[int, int, float, float]] = []
        for r in range(self.rows):
            for c in range(self.cols):
                di = offset_x_mm + (r - center) * self.spec.cup_spacing_mm
                dj = offset_y_mm + (c - center) * self.spec.cup_spacing_mm
                out.append((
                    r,
                    c,
                    di * cos_t - dj * sin_t + bias_x,
                    di * sin_t + dj * cos_t + bias_y,
                ))
        return out

    @staticmethod
    def _centroid_bias_mm(meat_slice: MeatSlice | None) -> tuple[float, float]:
        """Scarto tra baricentro della carne e centro dell'ingombro, in mm."""
        if meat_slice is None or meat_slice.shape_mask.size == 0:
            return 0.0, 0.0
        mask = meat_slice.shape_mask
        cells = np.argwhere(mask > 0)
        if cells.size == 0:
            return 0.0, 0.0
        res = meat_slice.resolution_mm
        bias_x = (cells[:, 0].mean() - (mask.shape[0] - 1) / 2.0) * res
        bias_y = (cells[:, 1].mean() - (mask.shape[1] - 1) / 2.0) * res
        return float(bias_x), float(bias_y)

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
