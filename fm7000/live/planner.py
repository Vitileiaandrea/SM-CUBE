"""Ponte tra le fette misurate dall'immagine e il pianificatore FM 7000.

Stessa logica del robot: destinazione scelta prima, presa derivata dopo,
ventose rientrate di 10 mm, polso quadro entro +/-180 gradi.
"""

from dataclasses import dataclass, field

import numpy as np

from fm7000.config.constants import (
    CUBE,
    GRIPPER,
    PUSH_TO_WALL,
    ROBOT,
    MeatType,
    PushDirection,
)
from fm7000.cube.slice_model import MeatSlice
from fm7000.cube.state import CubeState
from fm7000.live.segmentation import SliceDetection2D
from fm7000.rules.gripper import GripperCommand, GripperPatternSelector
from fm7000.rules.placement import PlacementCandidate, PlacementEngine
from fm7000.rules.recipe import RecipeManager


def meat_type_from_fat(fat_percentage: float) -> MeatType:
    if fat_percentage > 0.55:
        return MeatType.FAT
    if fat_percentage > 0.25:
        return MeatType.MEDIUM_QUALITY
    return MeatType.HIGH_QUALITY


def detection_to_slice(
    detection: SliceDetection2D, thickness_mm: float, slice_id: int = 0
) -> MeatSlice:
    """Fetta misurata -> MeatSlice con mappa spessori e grasso reali."""
    shape = detection.shape_mask
    thickness = shape * float(thickness_mm)
    return MeatSlice(
        width_mm=shape.shape[0] * detection.resolution_mm,
        length_mm=shape.shape[1] * detection.resolution_mm,
        thickness_min_mm=float(thickness_mm),
        thickness_max_mm=float(thickness_mm),
        meat_type=meat_type_from_fat(detection.fat_percentage),
        fat_percentage=detection.fat_percentage,
        slice_id=slice_id,
        shape_mask=shape,
        thickness_map=thickness,
        fat_map=detection.fat_map,
        resolution_mm=detection.resolution_mm,
    )


@dataclass
class LivePlan:
    """Presa + deposito proposti per una fetta, pronti per la conferma."""

    detection: SliceDetection2D
    meat_slice: MeatSlice
    candidate: PlacementCandidate
    gripper: GripperCommand
    score: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self, planner: "LivePlanner") -> dict:
        det = self.detection
        cand = self.candidate
        res = CUBE.resolution_mm
        prepared = cand.prepared_slice or self.meat_slice
        sw, sl = prepared.shape_mask.shape
        return {
            "indice": det.index,
            "punteggio": round(self.score, 2),
            "fetta": {
                "contorno_mm": det.contour_mm,
                "centro_px": list(det.center_px),
                "centro_mm": list(det.center_mm),
                "larghezza_mm": det.width_mm,
                "lunghezza_mm": det.length_mm,
                "area_mm2": det.area_mm2,
                "spessore_mm": round(self.meat_slice.avg_thickness_mm, 1),
                "grasso_pct": round(det.fat_percentage * 100, 1),
                "tipo": self.meat_slice.meat_type.name,
                "peso_g": round(self.meat_slice.volume_mm3 * 0.00105, 0),
            },
            "presa": {
                "ventose": self.gripper.cup_pattern.tolist(),
                "n_ventose": self.gripper.active_cups,
                "vuoto": self.gripper.vacuum_level,
                "margine_x_mm": round(self.gripper.meat_margin_x_mm, 1),
                "margine_y_mm": round(self.gripper.meat_margin_y_mm, 1),
                "margine_min_mm": round(self.gripper.min_clearance_mm, 1),
                "margine_ok": bool(self.gripper.margin_ok),
                "rotazione_polso_deg": cand.wrist_deg,
                "angolo_fetta_deg": round(cand.rotation_deg, 1),
                "presa_girata_deg": round(cand.pick_angle_deg, 1),
                "offset_x_mm": round(self.gripper.pick_offset_x_mm, 1),
                "offset_y_mm": round(self.gripper.pick_offset_y_mm, 1),
                # spostamento che la mano non puo' fare (ingombro 210 mm) e che
                # viene recuperato prendendo la fetta decentrata
                "sporgenza_x_mm": round(cand.pick_shift_x_mm, 1),
                "sporgenza_y_mm": round(cand.pick_shift_y_mm, 1),
                "posizioni_ventose_mm": planner.gripper_selector
                .get_cup_center_positions_mm(self.gripper.cup_pattern),
            },
            "deposito": {
                "zona": cand.zone.value,
                "x_mm": round(cand.x * res, 1),
                "y_mm": round(cand.y * res, 1),
                "larghezza_mm": round(sw * res, 1),
                "lunghezza_mm": round(sl * res, 1),
                "push": cand.push_direction.value,
                # corsa mano (limitata dall'ingombro) + 10 mm di bordo flesso
                "push_x_mm": round(cand.push_x_mm, 1),
                "push_y_mm": round(cand.push_y_mm, 1),
                "push_bordo_mm": PUSH_TO_WALL.corner_compression_mm,
                "sovrapposizione_pct": round(cand.overlap_ratio * 100, 1),
                "strato": planner.cube_state.current_layer_index,
                "contorno_impronta": planner.footprint_outline(prepared, cand.x, cand.y),
            },
            "note": self.notes,
        }


class LivePlanner:
    """Stato del cubo + regole FM 7000 per la sessione di prova dal vivo."""

    def __init__(self) -> None:
        self.cube_state = CubeState()
        self.placement = PlacementEngine()
        self.gripper_selector = GripperPatternSelector()
        self.recipe = RecipeManager()
        self.layer_thickness_mm = 25.0
        self.placed: list[dict] = []
        self.layer_coverages: list[float] = []

    # ---------------------------------------------------------------- pianifica

    def plan(self, detection: SliceDetection2D, thickness_mm: float) -> LivePlan | None:
        meat_slice = detection_to_slice(
            detection, thickness_mm, slice_id=detection.index
        )
        candidates = self.placement.find_candidates(self.cube_state, meat_slice)
        if not candidates:
            return None

        fallback: LivePlan | None = None
        for candidate in candidates:
            if abs(candidate.rotation_deg) > ROBOT.wrist_limit_deg:
                continue
            prepared = candidate.prepared_slice or meat_slice.rotate(
                candidate.rotation_deg
            )
            gripper = self.gripper_selector.select_pattern(
                candidate.zone,
                prepared,
                candidate.wrist_deg,
                candidate.pick_angle_deg,
                candidate.pick_shift_x_mm,
                candidate.pick_shift_y_mm,
            )
            notes = self._notes(meat_slice, candidate, gripper)
            plan = LivePlan(
                detection=detection,
                meat_slice=meat_slice,
                candidate=candidate,
                gripper=gripper,
                score=candidate.score,
                notes=notes,
            )
            if not gripper.margin_ok and candidate.push_direction != PushDirection.NONE:
                fallback = fallback or plan
                continue
            if candidate.overlap_ratio > CUBE.max_overlap_ratio:
                fallback = fallback or plan
                continue
            return plan
        return fallback

    def plan_all(
        self, detections: list[SliceDetection2D], thickness_mm: float
    ) -> list[LivePlan]:
        """Ordina le fette: prima quella che chiude meglio lo strato.

        Se nello strato corrente nessuna fetta ci sta, lo strato viene chiuso e
        si ripianifica: nessuna fetta viene scartata.
        """
        plans = [
            plan
            for plan in (self.plan(d, thickness_mm) for d in detections)
            if plan is not None
        ]
        if not plans and detections and not self.cube_state.is_full:
            self.close_layer()
            for detection in detections:
                plan = self.plan(detection, thickness_mm)
                if plan is not None:
                    plan.notes.insert(0, "strato chiuso: fetta spostata su quello nuovo")
                    plans.append(plan)
        plans.sort(key=lambda p: p.score, reverse=True)
        return plans

    def _notes(
        self,
        meat_slice: MeatSlice,
        candidate: PlacementCandidate,
        gripper: GripperCommand,
    ) -> list[str]:
        notes: list[str] = []
        if not gripper.margin_ok:
            worst = min(
                gripper.meat_margin_x_mm,
                gripper.meat_margin_y_mm,
                gripper.min_clearance_mm,
            )
            notes.append(
                f"carne libera oltre il labbro {worst:.0f} mm < "
                f"{GRIPPER.push_safety_margin_mm:.0f} mm: push-to-wall non ammesso"
            )
        if candidate.overlap_ratio > CUBE.max_overlap_ratio:
            notes.append(
                f"sovrapposizione {candidate.overlap_ratio * 100:.0f}%: "
                "conviene chiudere lo strato"
            )
        wanted = self.recipe.current_meat_type
        if meat_slice.meat_type != wanted:
            notes.append(
                f"ricetta: lo strato vuole {wanted.name}, "
                f"questa fetta e' {meat_slice.meat_type.name}"
            )
        if not self.recipe.fits_in_cube(
            float(np.max(self.cube_state.height_map)), meat_slice.avg_thickness_mm
        ):
            notes.append("altezza residua insufficiente: cubo da chiudere")
        return notes

    # ----------------------------------------------------------------- conferma

    def confirm(self, plan: LivePlan) -> dict:
        cand = plan.candidate
        prepared = cand.prepared_slice or plan.meat_slice.rotate(cand.rotation_deg)
        placed = self.cube_state.place_slice(
            prepared, cand.x, cand.y, cand.push_x_mm, cand.push_y_mm
        )
        if placed is None:
            return {"ok": False, "errore": "deposito non eseguibile"}

        self.layer_thickness_mm = prepared.avg_thickness_mm
        self.placed.append(
            {
                "strato": placed.layer_index,
                "x_mm": round(cand.x * CUBE.resolution_mm, 1),
                "y_mm": round(cand.y * CUBE.resolution_mm, 1),
                "z_mm": round(placed.z_mm, 1),
                "tipo": prepared.meat_type.name,
                "impronta": self.footprint_outline(prepared, cand.x, cand.y),
            }
        )

        closed = False
        if self.cube_state.current_layer_coverage >= CUBE.layer_coverage_threshold:
            self.layer_coverages.append(self.cube_state.force_advance_layer())
            self.recipe.advance_layer()
            closed = True
        return {"ok": True, "strato_chiuso": closed, "stato": self.state()}

    def close_layer(self) -> dict:
        self.layer_coverages.append(self.cube_state.force_advance_layer())
        self.recipe.advance_layer()
        return self.state()

    def reset(self) -> dict:
        self.cube_state.reset()
        self.recipe.reset()
        self.placed.clear()
        self.layer_coverages.clear()
        return self.state()

    # -------------------------------------------------------------------- stato

    def footprint_outline(self, meat_slice: MeatSlice, x: int, y: int) -> list[list[int]]:
        """Impronta della fetta nel cubo come celle occupate (per il disegno)."""
        active = np.argwhere(meat_slice.shape_mask > 0)
        return [[int(x + i), int(y + j)] for i, j in active]

    def state(self) -> dict:
        cube = self.cube_state
        return {
            "strato": cube.current_layer_index,
            "copertura_strato_pct": round(cube.current_layer_coverage * 100, 1),
            "copertura_strati": [round(c * 100, 1) for c in self.layer_coverages],
            "fette": cube.total_slices_placed,
            "riempimento_pct": round(cube.fill_percentage * 100, 1),
            "altezza_media_mm": round(float(np.mean(cube.height_map)), 1),
            "altezza_max_mm": round(float(np.max(cube.height_map)), 1),
            "planarita_mm": round(cube.flatness_mm, 1),
            "tipo_richiesto": self.recipe.current_meat_type.name,
            "cubo_pieno": bool(cube.is_full),
            "mappa_altezze": np.round(cube.height_map, 1).tolist(),
            "celle": [cube.w, cube.l],
            "risoluzione_mm": CUBE.resolution_mm,
            "cubo_mm": [CUBE.width_mm, CUBE.length_mm, CUBE.height_mm],
        }
