"""KUKA KR 3 DELTA D1200 HM robot interface."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from fm7000.config.constants import ROBOT, RobotSpec
from fm7000.robot.gripper import VacuumGripper
from fm7000.rules.gripper import GripperCommand


class RobotState(Enum):
    IDLE = "idle"
    MOVING_TO_PICK = "moving_to_pick"
    PICKING = "picking"
    MOVING_TO_PLACE = "moving_to_place"
    PLACING = "placing"
    PUSHING = "pushing"
    RETURNING = "returning"
    ERROR = "error"


@dataclass
class RobotPosition:
    x_mm: float = 0.0
    y_mm: float = 0.0
    z_mm: float = 0.0
    wrist_deg: float = 0.0


@dataclass
class PickPlaceCommand:
    pick_position: RobotPosition
    place_position: RobotPosition
    gripper_command: GripperCommand
    push_x_mm: float = 0.0
    push_y_mm: float = 0.0


@dataclass
class CycleMetrics:
    total_time_sec: float
    pick_time_sec: float
    move_time_sec: float
    place_time_sec: float
    push_time_sec: float
    return_time_sec: float


class KukaRobotInterface:
    """
    Interface to the KUKA KR 3 DELTA D1200 HM robot.

    Specs:
    - 3-DOF delta + 1 DOF wrist (axis 4, 360deg continuous)
    - Payload: 3kg (6kg max)
    - Reach: 1200mm diameter
    - Vertical workspace: 250mm
    - Adept cycle: 0.5 sec
    - Controller: KR C5 micro + KSS 8.7
    - PickControl 1.3 for conveyor tracking

    In production, communicates via KUKA.PLC mxAutomation or RSI.
    For simulation, calculates motion times and validates kinematics.
    """

    def __init__(
        self,
        spec: Optional[RobotSpec] = None,
        simulation: bool = True,
    ) -> None:
        self.spec = spec or ROBOT
        self.simulation = simulation
        self.gripper = VacuumGripper(simulation=simulation)
        self._state = RobotState.IDLE
        self._position = RobotPosition()
        self._connected = False
        self._total_cycles = 0
        self._total_cycle_time = 0.0

    @property
    def state(self) -> RobotState:
        return self._state

    @property
    def position(self) -> RobotPosition:
        return self._position

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def avg_cycle_time(self) -> float:
        if self._total_cycles == 0:
            return 0.0
        return self._total_cycle_time / self._total_cycles

    def connect(self) -> bool:
        if self.simulation:
            self._connected = True
            return True
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False
        self._state = RobotState.IDLE

    def execute_pick_place(self, command: PickPlaceCommand) -> Optional[CycleMetrics]:
        if not self._connected:
            return None

        if not self._validate_reach(command.pick_position):
            self._state = RobotState.ERROR
            return None
        if not self._validate_reach(command.place_position):
            self._state = RobotState.ERROR
            return None

        self._state = RobotState.MOVING_TO_PICK
        move_to_pick_time = max(
            self._calculate_move_time(self._position, command.pick_position),
            self.spec.move_to_pick_sec,
        )

        self._state = RobotState.PICKING
        self.gripper.pick(
            command.gripper_command.cup_pattern,
            command.gripper_command.vacuum_level,
        )
        pick_time = self.spec.pick_sec

        self._state = RobotState.MOVING_TO_PLACE
        traverse_time = max(
            self._calculate_move_time(command.pick_position, command.place_position),
            self.spec.traverse_sec,
        )
        wrist_time = abs(command.gripper_command.wrist_rotation_deg - self._position.wrist_deg) / 720.0
        move_time = max(traverse_time, wrist_time)

        self._state = RobotState.PLACING
        self.gripper.release()
        place_time = self.spec.cube_descent_sec + self.spec.release_sec

        push_time = 0.0
        if abs(command.push_x_mm) > 0 or abs(command.push_y_mm) > 0:
            self._state = RobotState.PUSHING
            push_time = self.spec.push_to_wall_sec

        self._state = RobotState.RETURNING
        home = RobotPosition(x_mm=0, y_mm=0, z_mm=100)
        return_time = max(
            self._calculate_move_time(command.place_position, home),
            self.spec.ascend_return_sec,
        )

        self._position = home
        self._state = RobotState.IDLE

        total = move_to_pick_time + pick_time + move_time + place_time + push_time + return_time
        self._total_cycles += 1
        self._total_cycle_time += total

        return CycleMetrics(
            total_time_sec=total,
            pick_time_sec=pick_time,
            move_time_sec=move_to_pick_time + move_time,
            place_time_sec=place_time,
            push_time_sec=push_time,
            return_time_sec=return_time,
        )

    def _validate_reach(self, pos: RobotPosition) -> bool:
        horizontal_dist = (pos.x_mm ** 2 + pos.y_mm ** 2) ** 0.5
        max_reach = self.spec.reach_diameter_mm / 2.0
        if horizontal_dist > max_reach:
            return False
        if pos.z_mm < 0 or pos.z_mm > self.spec.vertical_workspace_mm:
            return False
        return True

    def _calculate_move_time(self, start: RobotPosition, end: RobotPosition) -> float:
        dx = end.x_mm - start.x_mm
        dy = end.y_mm - start.y_mm
        dz = end.z_mm - start.z_mm
        dist = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5

        max_speed = 8000.0
        max_accel = 40000.0

        accel_dist = max_speed ** 2 / (2 * max_accel)

        if dist < 2 * accel_dist:
            time_s = 2 * (dist / max_accel) ** 0.5
        else:
            accel_time = max_speed / max_accel
            cruise_dist = dist - 2 * accel_dist
            cruise_time = cruise_dist / max_speed
            time_s = 2 * accel_time + cruise_time

        return max(time_s, 0.05)

    def home(self) -> bool:
        if not self._connected:
            return False
        self._position = RobotPosition(x_mm=0, y_mm=0, z_mm=100)
        self._state = RobotState.IDLE
        self.gripper.reset()
        return True

    def get_status(self) -> dict:
        return {
            "state": self._state.value,
            "position": {
                "x": self._position.x_mm,
                "y": self._position.y_mm,
                "z": self._position.z_mm,
                "wrist": self._position.wrist_deg,
            },
            "gripper": {
                "active_cups": self.gripper.active_cups,
                "vacuum_active": self.gripper.state.vacuum_active,
                "slice_detected": self.gripper.state.slice_detected,
            },
            "cycles": self._total_cycles,
            "avg_cycle_time": self.avg_cycle_time,
        }
