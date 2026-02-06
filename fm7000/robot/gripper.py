"""Vacuum gripper hardware abstraction for 4x4 cup grid."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from fm7000.config.constants import GRIPPER, GripperSpec


@dataclass
class GripperState:
    cup_pattern: np.ndarray
    vacuum_active: bool
    vacuum_level: float
    slice_detected: bool


class VacuumGripper:
    """
    Controls the 4x4 vacuum cup gripper.

    Physical specifications:
    - 16 cups in 4x4 grid
    - Cup diameter: 30mm (bellows/accordion style)
    - External interaxis: 180x180mm
    - Cup spacing: 40mm between centers
    - Each cup individually controllable (on/off)

    The gripper is mounted on the KUKA delta robot's axis 4 (360deg wrist).
    """

    def __init__(self, spec: Optional[GripperSpec] = None, simulation: bool = True) -> None:
        self.spec = spec or GRIPPER
        self.simulation = simulation
        self._cup_pattern = np.zeros((self.spec.rows, self.spec.cols), dtype=np.int8)
        self._vacuum_active = False
        self._vacuum_level = 0.0
        self._slice_on_gripper = False

    @property
    def state(self) -> GripperState:
        return GripperState(
            cup_pattern=self._cup_pattern.copy(),
            vacuum_active=self._vacuum_active,
            vacuum_level=self._vacuum_level,
            slice_detected=self._slice_on_gripper,
        )

    @property
    def active_cups(self) -> int:
        return int(np.sum(self._cup_pattern))

    def set_pattern(self, pattern: np.ndarray) -> bool:
        if pattern.shape != (self.spec.rows, self.spec.cols):
            return False
        self._cup_pattern = pattern.astype(np.int8)
        return True

    def activate_vacuum(self, level: float = 0.8) -> bool:
        if self.active_cups == 0:
            return False
        self._vacuum_active = True
        self._vacuum_level = min(max(level, 0.0), 1.0)
        return True

    def deactivate_vacuum(self) -> None:
        self._vacuum_active = False
        self._vacuum_level = 0.0
        self._slice_on_gripper = False

    def pick(self, pattern: np.ndarray, vacuum_level: float = 0.8) -> bool:
        if not self.set_pattern(pattern):
            return False
        if not self.activate_vacuum(vacuum_level):
            return False
        if self.simulation:
            self._slice_on_gripper = True
        return True

    def release(self) -> bool:
        if not self._slice_on_gripper:
            return False
        self.deactivate_vacuum()
        return True

    def reset(self) -> None:
        self._cup_pattern = np.zeros((self.spec.rows, self.spec.cols), dtype=np.int8)
        self._vacuum_active = False
        self._vacuum_level = 0.0
        self._slice_on_gripper = False
