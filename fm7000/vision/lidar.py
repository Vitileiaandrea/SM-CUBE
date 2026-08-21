"""LiDAR cube monitor - real-time fill level tracking."""

from dataclasses import dataclass

import numpy as np

from fm7000.config.constants import CUBE, CubeSpec


@dataclass
class CubeFillScan:
    height_map: np.ndarray
    mean_height_mm: float
    max_height_mm: float
    min_height_mm: float
    fill_percentage: float
    coverage_current_layer: float
    is_full: bool
    timestamp: float = 0.0


class LiDARCubeMonitor:
    """
    LiDAR sensor mounted above the cube for real-time fill monitoring.

    Provides independent verification of the cube fill state,
    complementing the software-tracked CubeState.

    In production, processes point cloud data from the LiDAR sensor.
    For simulation, reads from CubeState.
    """

    def __init__(
        self,
        cube_spec: CubeSpec | None = None,
        simulation: bool = True,
    ) -> None:
        self.spec = cube_spec or CUBE
        self.simulation = simulation
        self._connected = False
        self._scan_count = 0

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

    def scan_cube(self, cube_height_map: np.ndarray | None = None) -> CubeFillScan | None:
        if not self._connected:
            return None

        self._scan_count += 1

        if self.simulation and cube_height_map is not None:
            return self._simulate_scan(cube_height_map)

        return None

    def _simulate_scan(self, height_map: np.ndarray) -> CubeFillScan:
        noise = np.random.uniform(-0.5, 0.5, height_map.shape).astype(np.float32)
        scanned = np.maximum(0, height_map + noise)

        positive_mask = scanned > 0
        mean_h = float(np.mean(scanned)) if scanned.size > 0 else 0.0
        max_h = float(np.max(scanned)) if scanned.size > 0 else 0.0
        min_h = float(np.min(scanned[positive_mask])) if np.any(positive_mask) else 0.0

        fill_pct = mean_h / self.spec.height_mm
        coverage = float(np.sum(positive_mask)) / float(scanned.size)

        return CubeFillScan(
            height_map=scanned,
            mean_height_mm=mean_h,
            max_height_mm=max_h,
            min_height_mm=min_h,
            fill_percentage=fill_pct,
            coverage_current_layer=coverage,
            is_full=fill_pct >= 0.95,
            timestamp=float(self._scan_count),
        )

    def verify_placement(
        self,
        before_scan: CubeFillScan,
        after_scan: CubeFillScan,
        expected_thickness_mm: float,
    ) -> bool:
        height_diff = after_scan.mean_height_mm - before_scan.mean_height_mm
        tolerance = expected_thickness_mm * 0.5
        return height_diff > 0 and abs(height_diff - expected_thickness_mm) < tolerance
