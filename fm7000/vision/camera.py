"""Camera + Cognex ViDi interface for slice detection and classification."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from fm7000.config.constants import MeatType


class ViDiTool(Enum):
    GREEN_CLASSIFY = "green_classify"
    BLUE_LOCATE = "blue_locate"
    RED_ANALYZE = "red_analyze"


@dataclass
class SliceDetection:
    slice_id: int
    meat_type: MeatType
    confidence: float
    centroid_x_mm: float
    centroid_y_mm: float
    orientation_deg: float
    width_mm: float
    length_mm: float
    bounding_box: tuple[float, float, float, float]
    conveyor_lane: int
    fat_percentage: float


class CameraInterface:
    """
    Interface to the 4K/8K camera + Cognex ViDi deep learning system.

    In production, communicates with the camera over GigE Vision / GenICam
    and with Cognex ViDi runtime for classification, location, and analysis.

    For simulation, generates synthetic detections.
    """

    def __init__(self, resolution: str = "4K", simulation: bool = True) -> None:
        self.resolution = resolution
        self.simulation = simulation
        self._next_slice_id = 0
        self._connected = False

    def connect(self) -> bool:
        if self.simulation:
            self._connected = True
            return True
        self._connected = True
        return True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def capture_and_detect(self, conveyor_lane: int) -> list[SliceDetection]:
        if not self._connected:
            return []

        if self.simulation:
            return self._simulate_detections(conveyor_lane)

        return []

    def detect_all_lanes(self) -> list[SliceDetection]:
        all_detections: list[SliceDetection] = []
        for lane in range(3):
            detections = self.capture_and_detect(lane)
            all_detections.extend(detections)
        return all_detections

    def _simulate_detections(self, conveyor_lane: int) -> list[SliceDetection]:
        lane_to_type = {
            0: MeatType.HIGH_QUALITY,
            1: MeatType.MEDIUM_QUALITY,
            2: MeatType.FAT,
        }
        meat_type = lane_to_type.get(conveyor_lane, MeatType.MEDIUM_QUALITY)

        num_slices = np.random.randint(1, 4)
        detections = []

        for _ in range(num_slices):
            self._next_slice_id += 1
            width = np.random.uniform(50, 200)
            length = np.random.uniform(50, 200)
            cx = np.random.uniform(100, 500)
            cy = np.random.uniform(50, 150) + conveyor_lane * 200
            orientation = np.random.uniform(0, 360)

            if meat_type == MeatType.FAT:
                fat_pct = np.random.uniform(80, 95)
            elif meat_type == MeatType.HIGH_QUALITY:
                fat_pct = np.random.uniform(5, 15)
            else:
                fat_pct = np.random.uniform(15, 35)

            detection = SliceDetection(
                slice_id=self._next_slice_id,
                meat_type=meat_type,
                confidence=np.random.uniform(0.85, 0.99),
                centroid_x_mm=cx,
                centroid_y_mm=cy,
                orientation_deg=orientation,
                width_mm=width,
                length_mm=length,
                bounding_box=(cx - width / 2, cy - length / 2, width, length),
                conveyor_lane=conveyor_lane,
                fat_percentage=fat_pct,
            )
            detections.append(detection)

        return detections
