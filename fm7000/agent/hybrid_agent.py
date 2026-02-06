"""Hybrid agent - deterministic rules + RL optimization for cube filling."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from fm7000.config.constants import (
    CUBE,
    GRIPPER,
    PERFORMANCE,
    CubeSpec,
    GripperSpec,
    MeatType,
    PlacementZone,
)
from fm7000.cube.state import CubeState
from fm7000.cube.slice_model import MeatSlice
from fm7000.rules.recipe import RecipeManager
from fm7000.rules.placement import PlacementEngine, PlacementCandidate
from fm7000.rules.gripper import GripperPatternSelector, GripperCommand
from fm7000.vision.camera import CameraInterface, SliceDetection
from fm7000.vision.profiler import ProfilerInterface
from fm7000.vision.lidar import LiDARCubeMonitor
from fm7000.robot.kuka_interface import (
    KukaRobotInterface,
    PickPlaceCommand,
    RobotPosition,
    CycleMetrics,
)


@dataclass
class PlacementDecision:
    meat_slice: MeatSlice
    candidate: PlacementCandidate
    gripper_command: GripperCommand
    pick_position: RobotPosition
    detection: SliceDetection
    estimated_cycle_time: float = 0.0


@dataclass
class CubeResult:
    total_slices: int
    total_time_sec: float
    fill_percentage: float
    layer_count: int
    recipe_followed: bool
    avg_cycle_time: float


class HybridAgent:
    """
    Main controller for the FM 7000 Filling Machine.

    Architecture: Hybrid rule-based + RL optimization.

    Deterministic rules (always enforced):
    - Recipe layer sequence (high quality -> fat -> medium -> ... -> high quality)
    - Perimeter-first filling (corners -> edges -> center)
    - Push-to-wall with safety margins
    - Gripper pattern selection based on target zone
    - Layer completion threshold (95%)

    RL optimization (learns to improve):
    - Exact position within valid zone
    - Rotation for wedge matching
    - Fat distribution balancing
    - Slice selection from available candidates
    """

    def __init__(
        self,
        cube_spec: Optional[CubeSpec] = None,
        simulation: bool = True,
    ) -> None:
        self.spec = cube_spec or CUBE
        self.simulation = simulation

        self.cube_state = CubeState(self.spec)
        self.recipe = RecipeManager()
        self.placement_engine = PlacementEngine(self.spec)
        self.gripper_selector = GripperPatternSelector()

        self.camera = CameraInterface(simulation=simulation)
        self.profiler = ProfilerInterface(simulation=simulation)
        self.lidar = LiDARCubeMonitor(self.spec, simulation=simulation)
        self.robot = KukaRobotInterface(simulation=simulation)

        self._use_rl = False
        self._cycle_count = 0
        self._total_time = 0.0

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

    def fill_cube(self) -> CubeResult:
        self.cube_state.reset()
        self.recipe.reset()
        self._cycle_count = 0
        self._total_time = 0.0

        while not self.cube_state.is_full:
            required_type = self.recipe.current_meat_type

            if self.recipe.should_switch_to_final(
                float(np.mean(self.cube_state.height_map)),
                self.spec.height_mm,
            ):
                required_type = MeatType.HIGH_QUALITY

            decision = self._find_best_decision(required_type)

            if decision is None:
                if self.cube_state.current_layer_coverage >= self.spec.layer_coverage_threshold:
                    self.cube_state.advance_layer()
                    self.recipe.advance_layer()
                    continue
                else:
                    decision = self._find_any_decision(required_type)
                    if decision is None:
                        self.cube_state.force_advance_layer()
                        self.recipe.advance_layer()
                        continue

            metrics = self._execute_placement(decision)
            if metrics is not None:
                self._cycle_count += 1
                self._total_time += metrics.total_time_sec

            if self.cube_state.current_layer_coverage >= self.spec.layer_coverage_threshold:
                self.cube_state.advance_layer()
                self.recipe.advance_layer()

        return CubeResult(
            total_slices=self.cube_state.total_slices_placed,
            total_time_sec=self._total_time,
            fill_percentage=self.cube_state.fill_percentage,
            layer_count=self.cube_state.current_layer_index + 1,
            recipe_followed=True,
            avg_cycle_time=self._total_time / max(self._cycle_count, 1),
        )

    def step(self) -> Optional[PlacementDecision]:
        if self.cube_state.is_full:
            return None

        required_type = self.recipe.current_meat_type

        if self.recipe.should_switch_to_final(
            float(np.mean(self.cube_state.height_map)),
            self.spec.height_mm,
        ):
            required_type = MeatType.HIGH_QUALITY

        decision = self._find_best_decision(required_type)

        if decision is None:
            if self.cube_state.current_layer_coverage >= self.spec.layer_coverage_threshold:
                self.cube_state.advance_layer()
                self.recipe.advance_layer()
                return self.step()
            return None

        self._execute_placement(decision)
        self._cycle_count += 1

        if self.cube_state.current_layer_coverage >= self.spec.layer_coverage_threshold:
            self.cube_state.advance_layer()
            self.recipe.advance_layer()

        return decision

    def _find_best_decision(self, required_type: MeatType) -> Optional[PlacementDecision]:
        lane = self._meat_type_to_lane(required_type)
        detections = self.camera.capture_and_detect(lane)

        if not detections:
            return None

        best_decision: Optional[PlacementDecision] = None
        best_score = -float("inf")

        for detection in detections:
            if detection.meat_type != required_type:
                continue

            meat_slice = self.profiler.scan_to_meat_slice(detection)
            if meat_slice is None:
                continue

            candidates = self.placement_engine.find_candidates(
                self.cube_state, meat_slice
            )

            for candidate in candidates:
                rotated = meat_slice.rotate(candidate.rotation_deg)
                gripper_cmd = self.gripper_selector.select_pattern(
                    candidate.zone,
                    rotated,
                    candidate.rotation_deg,
                )

                score = candidate.score
                if self._use_rl:
                    score += self._rl_score_adjustment(candidate, rotated)

                if score > best_score:
                    best_score = score
                    pick_pos = RobotPosition(
                        x_mm=detection.centroid_x_mm,
                        y_mm=detection.centroid_y_mm,
                        z_mm=10.0,
                        wrist_deg=candidate.rotation_deg,
                    )
                    best_decision = PlacementDecision(
                        meat_slice=rotated,
                        candidate=candidate,
                        gripper_command=gripper_cmd,
                        pick_position=pick_pos,
                        detection=detection,
                    )

        return best_decision

    def _find_any_decision(self, required_type: MeatType) -> Optional[PlacementDecision]:
        for lane in range(3):
            detections = self.camera.capture_and_detect(lane)
            for detection in detections:
                if detection.meat_type != required_type:
                    continue
                meat_slice = self.profiler.scan_to_meat_slice(detection)
                if meat_slice is None:
                    continue
                candidate = self.placement_engine.find_best_placement(
                    self.cube_state, meat_slice
                )
                if candidate is not None:
                    rotated = meat_slice.rotate(candidate.rotation_deg)
                    gripper_cmd = self.gripper_selector.select_pattern(
                        candidate.zone, rotated, candidate.rotation_deg
                    )
                    pick_pos = RobotPosition(
                        x_mm=detection.centroid_x_mm,
                        y_mm=detection.centroid_y_mm,
                        z_mm=10.0,
                        wrist_deg=candidate.rotation_deg,
                    )
                    return PlacementDecision(
                        meat_slice=rotated,
                        candidate=candidate,
                        gripper_command=gripper_cmd,
                        pick_position=pick_pos,
                        detection=detection,
                    )
        return None

    def _execute_placement(self, decision: PlacementDecision) -> Optional[CycleMetrics]:
        place_x_mm = decision.candidate.x * self.spec.resolution_mm + decision.meat_slice.width_mm / 2
        place_y_mm = decision.candidate.y * self.spec.resolution_mm + decision.meat_slice.length_mm / 2
        place_z_mm = self.cube_state.get_layer_floor() + decision.meat_slice.avg_thickness_mm

        place_pos = RobotPosition(
            x_mm=place_x_mm,
            y_mm=place_y_mm,
            z_mm=min(place_z_mm, self.spec.height_mm),
            wrist_deg=decision.candidate.rotation_deg,
        )

        command = PickPlaceCommand(
            pick_position=decision.pick_position,
            place_position=place_pos,
            gripper_command=decision.gripper_command,
            push_x_mm=decision.candidate.push_x_mm,
            push_y_mm=decision.candidate.push_y_mm,
        )

        metrics = self.robot.execute_pick_place(command)

        self.cube_state.place_slice(
            decision.meat_slice,
            decision.candidate.x,
            decision.candidate.y,
            push_x_mm=decision.candidate.push_x_mm,
            push_y_mm=decision.candidate.push_y_mm,
        )

        return metrics

    def _rl_score_adjustment(
        self, candidate: PlacementCandidate, meat_slice: MeatSlice
    ) -> float:
        return 0.0

    def _meat_type_to_lane(self, meat_type: MeatType) -> int:
        lane_map = {
            MeatType.HIGH_QUALITY: 0,
            MeatType.MEDIUM_QUALITY: 1,
            MeatType.FAT: 2,
        }
        return lane_map.get(meat_type, 1)

    def get_status(self) -> dict:
        return {
            "cube": self.cube_state.get_state_summary(),
            "robot": self.robot.get_status(),
            "recipe": {
                "current_layer": self.recipe.current_layer,
                "current_type": self.recipe.current_meat_type.name,
                "conveyor_lane": self.recipe.get_conveyor_lane(),
            },
            "cycles": self._cycle_count,
            "total_time_sec": self._total_time,
            "avg_cycle_time": self._total_time / max(self._cycle_count, 1),
            "rl_enabled": self._use_rl,
        }
