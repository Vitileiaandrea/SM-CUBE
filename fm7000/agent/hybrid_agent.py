"""Hybrid agent - deterministic rules + RL optimization for cube filling."""

from dataclasses import dataclass, field

import numpy as np

from fm7000.config.constants import (
    CUBE,
    GRIPPER,
    ROBOT,
    SEARCH,
    CubeSpec,
    MeatType,
    PushDirection,
)
from fm7000.cube.slice_model import MeatSlice
from fm7000.cube.state import CubeState
from fm7000.robot.kuka_interface import (
    CycleMetrics,
    KukaRobotInterface,
    PickPlaceCommand,
    RobotPosition,
)
from fm7000.rules.gripper import GripperCommand, GripperPatternSelector
from fm7000.rules.placement import PlacementCandidate, PlacementEngine
from fm7000.rules.recipe import RecipeManager
from fm7000.vision.camera import CameraInterface, SliceDetection
from fm7000.vision.lidar import LiDARCubeMonitor
from fm7000.vision.profiler import ProfilerInterface


@dataclass
class PlacementDecision:
    meat_slice: MeatSlice
    candidate: PlacementCandidate
    gripper_command: GripperCommand
    pick_position: RobotPosition
    detection: SliceDetection
    estimated_cycle_time: float = 0.0


@dataclass
class LayerReport:
    index: int
    meat_type: str
    coverage: float
    slices: int
    height_mm: float
    thickness_mm: float


@dataclass
class CubeResult:
    total_slices: int
    total_time_sec: float
    fill_percentage: float
    layer_count: int
    recipe_followed: bool
    avg_cycle_time: float
    mean_height_mm: float = 0.0
    max_height_mm: float = 0.0
    flatness_mm: float = 0.0
    overlap_tolerated: int = 0
    recipe_violations: int = 0
    fat_column_index: float = 0.0
    planned_layers: int = 0
    layers: list[LayerReport] = field(default_factory=list)

    @property
    def cubes_per_hour(self) -> float:
        if self.total_time_sec <= 0:
            return 0.0
        return 3600.0 / self.total_time_sec

    @property
    def mean_layer_coverage(self) -> float:
        if not self.layers:
            return 0.0
        return float(np.mean([r.coverage for r in self.layers]))


class HybridAgent:
    """
    Main controller for the FM 7000 Filling Machine.

    Regole deterministiche (sempre imposte):
    - sequenza ricetta, pianificata sugli spessori reali misurati dal profilatore
    - riempimento perimeter-first con push-to-wall
    - pattern ventose con 10 mm di carne libera per la spinta
    - chiusura strato su copertura E su altezza verificata dal LiDAR

    Ottimizzazione (RL, opzionale):
    - posizione e rotazione fine, wedge matching, distribuzione del grasso
    """

    def __init__(
        self,
        cube_spec: CubeSpec | None = None,
        simulation: bool = True,
    ) -> None:
        self.spec = cube_spec or CUBE
        self.simulation = simulation

        self.cube_state = CubeState(self.spec)
        self.recipe = RecipeManager(cube_height_mm=self.spec.height_mm)
        self.placement_engine = PlacementEngine(self.spec)
        self.gripper_selector = GripperPatternSelector()

        self.camera = CameraInterface(simulation=simulation)
        self.profiler = ProfilerInterface(simulation=simulation)
        self.lidar = LiDARCubeMonitor(self.spec, simulation=simulation)
        self.robot = KukaRobotInterface(simulation=simulation)

        self._use_rl = False
        self._cycle_count = 0
        self._total_time = 0.0
        self._recipe_violations = 0
        self._layer_target_mm = 30.0

    def initialize(self) -> bool:
        success = True
        success = success and self.camera.connect()
        success = success and self.profiler.connect()
        success = success and self.lidar.connect()
        success = success and self.robot.connect()
        if success:
            self.robot.home()
        return success

    def shutdown(self) -> None:
        self.robot.home()
        self.camera.disconnect()
        self.profiler.disconnect()
        self.lidar.disconnect()
        self.robot.disconnect()

    # ------------------------------------------------------------ ciclo cubo

    def fill_cube(self, max_slices: int = 400) -> CubeResult:
        self.cube_state.reset()
        self.recipe.reset()
        self._cycle_count = 0
        self._total_time = 0.0
        self._recipe_violations = 0

        planned = self._plan_recipe_from_conveyor()
        layer_reports: list[LayerReport] = []
        layer_slices = 0
        stalls = 0

        while not self.cube_state.is_full and self._cycle_count < max_slices:
            required_type = self._required_type()
            decision = self._find_best_decision(required_type)

            if decision is None:
                decision = self._find_decision_any_lane(required_type)
            if decision is None:
                decision = self._find_decision_any_type(required_type)

            if decision is None:
                # nessuna fetta in arrivo entra nei vuoti rimasti: si chiude lo
                # strato e si passa al successivo
                if layer_slices == 0:
                    # nastro momentaneamente senza la carne richiesta: si
                    # aspetta il passaggio successivo, non si abbandona il cubo
                    stalls += 1
                    if stalls >= SEARCH.max_empty_conveyor_cycles:
                        break
                    continue
                stalls = 0
                layer_reports.append(
                    self._close_layer(len(layer_reports), required_type, layer_slices)
                )
                layer_slices = 0
                continue

            # la fetta e' gia in mano: se sullo strato corrente si accavallerebbe
            # troppo, si chiude lo strato e la si posa su quello nuovo
            if (
                layer_slices > 0
                and decision.candidate.overlap_ratio
                > self.spec.layer_close_overlap_ratio
            ):
                layer_reports.append(
                    self._close_layer(len(layer_reports), required_type, layer_slices)
                )
                layer_slices = 0
                decision = self._replan_in_hand(decision)
                if decision is None:
                    continue

            stalls = 0
            metrics = self._execute_placement(decision)
            if metrics is not None:
                self._cycle_count += 1
                self._total_time += metrics.total_time_sec
                layer_slices += 1

            if self._layer_complete():
                layer_reports.append(
                    self._close_layer(len(layer_reports), required_type, layer_slices)
                )
                layer_slices = 0

        if layer_slices > 0:
            layer_reports.append(
                self._close_layer(
                    len(layer_reports), self.recipe.current_meat_type, layer_slices
                )
            )

        return self._build_result(layer_reports, planned)

    def step(self) -> PlacementDecision | None:
        if self.cube_state.is_full:
            return None

        required_type = self._required_type()
        decision = self._find_best_decision(required_type)
        if decision is None:
            decision = self._find_decision_any_lane(required_type)
        if decision is None:
            return None

        metrics = self._execute_placement(decision)
        if metrics is not None:
            self._cycle_count += 1
            self._total_time += metrics.total_time_sec

        if self._layer_complete():
            self.cube_state.force_advance_layer()
            self.recipe.advance_layer()

        return decision

    # ------------------------------------------------- ricetta e riscontro 3D

    def _plan_recipe_from_conveyor(self) -> int:
        """
        Pianifica gli strati sullo spessore reale delle fette in arrivo.

        Campiona le tre corsie e misura lo spessore medio col profilatore: da
        qui esce quanti strati entrano nei 250 mm, primo e ultimo di alta
        qualita. Le fette sono solo appoggiate, quindi lo spessore misurato
        e' quello che occupa nel cubo.
        """
        thicknesses: list[float] = []
        for lane in range(3):
            for detection in self.camera.capture_and_detect(lane):
                meat_slice = self.profiler.scan_to_meat_slice(detection)
                if meat_slice is not None:
                    thicknesses.append(meat_slice.avg_thickness_mm)

        if not thicknesses:
            self._layer_target_mm = 30.0
            return len(self.recipe.sequence)

        avg = float(np.mean(thicknesses)) * self.spec.layer_compression_ratio
        self._layer_target_mm = avg
        return len(self.recipe.plan_for_thickness(avg))

    def measure_cube(self) -> float | None:
        """Altezza media del cubo letta dal LiDAR sopra la stazione."""
        scan = self.lidar.scan_cube(self.cube_state.height_map)
        return scan.mean_height_mm if scan is not None else None

    def _required_type(self) -> MeatType:
        # per decidere l'ultimo strato conta il punto piu alto letto dal LiDAR:
        # e' quello che chiude il cubo, e l'ultimo boccone deve essere di prima
        scan = self.lidar.scan_cube(self.cube_state.height_map)
        height = (
            scan.max_height_mm
            if scan is not None
            else float(np.max(self.cube_state.height_map))
        )
        if self.recipe.should_switch_to_final(height, self._layer_target_mm):
            self.recipe.force_final_layer()
        return self.recipe.current_meat_type

    def _layer_complete(self) -> bool:
        """
        Strato chiuso sulla copertura; il LiDAR sopra il cubo fa da riscontro
        di altezza e chiude comunque lo strato se e' cresciuto troppo, cosi
        l'altezza dei layer resta quella pianificata.
        """
        if self.cube_state.current_layer_coverage >= self.spec.layer_coverage_threshold:
            return True

        scan = self.lidar.scan_cube(self.cube_state.height_map)
        if scan is None:
            return False
        grown = scan.mean_height_mm - self.cube_state.get_layer_floor()
        return grown >= self._layer_target_mm * 1.5

    def _close_layer(
        self, index: int, meat_type: MeatType, slices: int
    ) -> LayerReport:
        floor_before = self.cube_state.get_layer_floor()
        coverage = self.cube_state.force_advance_layer()
        self.recipe.advance_layer()
        height = self.cube_state.get_layer_floor()
        return LayerReport(
            index=index,
            meat_type=meat_type.name,
            coverage=coverage,
            slices=slices,
            height_mm=height,
            thickness_mm=height - floor_before,
        )

    def _build_result(
        self, layer_reports: list[LayerReport], planned_layers: int
    ) -> CubeResult:
        fat_columns = self.cube_state.get_fat_column_map()
        fat_index = float(np.mean(fat_columns > 0.6))
        return CubeResult(
            total_slices=self.cube_state.total_slices_placed,
            total_time_sec=self._total_time,
            fill_percentage=self.cube_state.fill_percentage,
            layer_count=len(layer_reports),
            recipe_followed=self._recipe_violations == 0,
            avg_cycle_time=self._total_time / max(self._cycle_count, 1),
            mean_height_mm=float(np.mean(self.cube_state.height_map)),
            max_height_mm=float(np.max(self.cube_state.height_map)),
            flatness_mm=self.cube_state.flatness_mm,
            overlap_tolerated=self.cube_state.overlap_tolerated,
            recipe_violations=self._recipe_violations,
            fat_column_index=fat_index,
            planned_layers=planned_layers,
            layers=layer_reports,
        )

    # ----------------------------------------------------------- decisioni

    def _find_best_decision(
        self, required_type: MeatType
    ) -> PlacementDecision | None:
        lane = RecipeManager.lane_for_type(required_type)
        return self._decide_from_lane(lane, required_type)

    def _find_decision_any_lane(
        self, required_type: MeatType
    ) -> PlacementDecision | None:
        """Il tipo richiesto potrebbe trovarsi su un'altra corsia."""
        for lane in range(3):
            decision = self._decide_from_lane(lane, required_type)
            if decision is not None:
                return decision
        return None

    def _find_decision_any_type(
        self, required_type: MeatType
    ) -> PlacementDecision | None:
        """
        Ultima risorsa: accetta un tipo diverso per non fermare la macchina,
        ma non sul primo/ultimo strato, che devono restare alta qualita.
        """
        if required_type == MeatType.HIGH_QUALITY:
            return None
        for lane in range(3):
            decision = self._decide_from_lane(lane, None)
            if decision is not None:
                self._recipe_violations += 1
                return decision
        return None

    def _decide_from_lane(
        self, lane: int, required_type: MeatType | None
    ) -> PlacementDecision | None:
        """
        Decisione unica presa+deposito.

        Si parte dalla destinazione: per ogni fetta che passa sul nastro si
        cercano le posizioni nel cubo e l'angolo quadro di deposito; da quella
        scelta si deriva la presa (pattern ventose, 10 mm di carne libera,
        angolo di polso entro +/-180). Se la presa non e' compatibile la coppia
        viene scartata e si guarda la fetta successiva.
        """
        frames = max(1, int(SEARCH.conveyor_frames_per_pick))
        best_decision: PlacementDecision | None = None
        best_score = -float("inf")
        best_free: PlacementDecision | None = None
        best_free_score = -float("inf")

        for _ in range(frames):
            for detection in self.camera.capture_and_detect(lane):
                if required_type is not None and detection.meat_type != required_type:
                    continue

                meat_slice = self.profiler.scan_to_meat_slice(detection)
                if meat_slice is None:
                    continue

                candidates = self.placement_engine.find_candidates(
                    self.cube_state, meat_slice
                )
                for candidate in candidates:
                    decision = self._build_decision(detection, meat_slice, candidate)
                    if decision is None:
                        continue
                    score = candidate.score
                    if self._use_rl:
                        score += self._rl_score_adjustment(
                            candidate, decision.meat_slice
                        )
                    if score > best_score:
                        best_score = score
                        best_decision = decision
                    # fetta che entra in un vuoto libero dello strato corrente
                    if (
                        candidate.overlap_ratio <= self.spec.max_overlap_ratio
                        and score > best_free_score
                    ):
                        best_free_score = score
                        best_free = decision

            # si continua a guardare il nastro finche arriva una fetta che entra
            # nei vuoti rimasti, senza accavallarsi
            if best_free is not None:
                return best_free

        return best_free or best_decision

    def _replan_in_hand(
        self, decision: PlacementDecision
    ) -> PlacementDecision | None:
        """Ripianifica la fetta gia presa sullo strato appena aperto."""
        candidate = self.placement_engine.find_best_placement(
            self.cube_state, decision.meat_slice
        )
        if candidate is None:
            return None
        return self._build_decision(decision.detection, decision.meat_slice, candidate)

    def _build_decision(
        self,
        detection: SliceDetection,
        meat_slice: MeatSlice,
        candidate: PlacementCandidate,
    ) -> PlacementDecision | None:
        """Dalla destinazione scelta ricava la presa; None se non e' eseguibile."""
        # nel cubo conta il polso quadro: la corsa e' +-180 per i tubi aria
        if abs(candidate.wrist_deg) > ROBOT.wrist_limit_deg:
            return None

        prepared = candidate.prepared_slice or meat_slice.rotate(
            candidate.rotation_deg
        )
        gripper_cmd = self.gripper_selector.select_pattern(
            candidate.zone,
            prepared,
            candidate.wrist_deg,
            candidate.pick_angle_deg,
            candidate.pick_shift_x_mm,
            candidate.pick_shift_y_mm,
        )
        if not gripper_cmd.margin_ok and candidate.push_direction != PushDirection.NONE:
            # senza 10 mm di carne libera il perimetro non puo flettersi
            return None

        return PlacementDecision(
            meat_slice=prepared,
            candidate=candidate,
            gripper_command=gripper_cmd,
            # presa libera: fuori dal cubo la mano gira a qualunque angolo e la
            # griglia si sposta sul punto della fetta con piu' ventose valide
            pick_position=RobotPosition(
                x_mm=detection.centroid_x_mm + gripper_cmd.pick_offset_x_mm,
                y_mm=detection.centroid_y_mm + gripper_cmd.pick_offset_y_mm,
                z_mm=10.0,
                wrist_deg=self._pick_wrist(candidate.pick_angle_deg),
            ),
            detection=detection,
        )

    @staticmethod
    def _pick_wrist(pick_angle_deg: float) -> float:
        """Polso alla presa: libero, entro la corsa dei tubi aria."""
        limit = ROBOT.wrist_limit_deg
        angle = -float(pick_angle_deg)
        while angle > limit:
            angle -= 360.0
        while angle < -limit:
            angle += 360.0
        return round(angle, 1)

    @staticmethod
    def _clamp_hand(center_mm: float, cube_side_mm: float) -> float:
        """Tiene il centro mano dentro la corsa consentita dall'ingombro."""
        play = GRIPPER.hand_play_mm(cube_side_mm)
        mid = cube_side_mm / 2.0
        return float(np.clip(center_mm, mid - play, mid + play))

    def _execute_placement(
        self, decision: PlacementDecision
    ) -> CycleMetrics | None:
        meat_slice = decision.meat_slice
        sw, sl = meat_slice.shape_mask.shape
        place_x_mm = (decision.candidate.x + sw / 2.0) * self.spec.resolution_mm
        place_y_mm = (decision.candidate.y + sl / 2.0) * self.spec.resolution_mm
        # l'ingombro della mano (180 mm interasse + 30 mm di labbro) riempie il
        # cubo: il centro mano resta nella corsa ammessa, la fetta arriva a
        # parete grazie all'offset della presa
        place_x_mm = self._clamp_hand(place_x_mm, self.spec.width_mm)
        place_y_mm = self._clamp_hand(place_y_mm, self.spec.length_mm)
        place_z_mm = self.cube_state.get_layer_floor() + meat_slice.avg_thickness_mm

        command = PickPlaceCommand(
            pick_position=decision.pick_position,
            place_position=RobotPosition(
                x_mm=place_x_mm,
                y_mm=place_y_mm,
                z_mm=min(place_z_mm, self.spec.height_mm),
                wrist_deg=decision.candidate.wrist_deg,
            ),
            gripper_command=decision.gripper_command,
            push_x_mm=decision.candidate.push_x_mm,
            push_y_mm=decision.candidate.push_y_mm,
        )

        metrics = self.robot.execute_pick_place(command)
        placed = self.cube_state.place_slice(
            meat_slice,
            decision.candidate.x,
            decision.candidate.y,
            push_x_mm=decision.candidate.push_x_mm,
            push_y_mm=decision.candidate.push_y_mm,
        )
        if placed is None:
            return None
        return metrics

    def _rl_score_adjustment(
        self, candidate: PlacementCandidate, meat_slice: MeatSlice
    ) -> float:
        return 0.0

    def _meat_type_to_lane(self, meat_type: MeatType) -> int:
        return RecipeManager.lane_for_type(meat_type)

    def get_status(self) -> dict:
        return {
            "cube": self.cube_state.get_state_summary(),
            "robot": self.robot.get_status(),
            "recipe": {
                "current_layer": self.recipe.current_layer,
                "current_type": self.recipe.current_meat_type.name,
                "conveyor_lane": self.recipe.get_conveyor_lane(),
                "planned_layers": len(self.recipe.sequence),
                "layer_target_mm": self._layer_target_mm,
                "final_phase": self.recipe.in_final_phase,
            },
            "cycles": self._cycle_count,
            "total_time_sec": self._total_time,
            "avg_cycle_time": self._total_time / max(self._cycle_count, 1),
            "rl_enabled": self._use_rl,
        }
