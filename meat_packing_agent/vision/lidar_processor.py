"""
LiDAR Vision System Interface

This module processes 3D point cloud data from the LiDAR sensor to extract
meat slice geometry (shape, dimensions, thickness) for the RL agent.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from scipy import ndimage
from scipy.spatial import ConvexHull


@dataclass
class SliceGeometry:
    """Extracted geometry of a meat slice from LiDAR data."""
    
    width: float  # mm
    length: float  # mm
    thickness: float  # mm
    centroid: Tuple[float, float, float]  # (x, y, z) in mm
    orientation: float  # degrees
    area: float  # mm^2
    volume: float  # mm^3
    shape_mask: np.ndarray  # 2D binary mask
    point_cloud: np.ndarray  # Original point cloud
    confidence: float  # 0-1 confidence score


class LiDARProcessor:
    """
    Processes LiDAR point cloud data to extract meat slice geometry.
    
    The processor handles:
    - Point cloud filtering and noise removal
    - Ground plane detection and removal
    - Slice segmentation
    - Geometry extraction (dimensions, shape, orientation)
    """
    
    def __init__(
        self,
        conveyor_height: float = 0.0,
        min_slice_area: float = 2500.0,  # mm^2 (50x50 minimum)
        max_slice_area: float = 40000.0,  # mm^2 (200x200 maximum)
        voxel_size: float = 2.0,  # mm
        noise_threshold: float = 5.0  # mm
    ):
        self.conveyor_height = conveyor_height
        self.min_slice_area = min_slice_area
        self.max_slice_area = max_slice_area
        self.voxel_size = voxel_size
        self.noise_threshold = noise_threshold
        
        self._calibration_offset = np.zeros(3)
        self._is_calibrated = False
    
    def calibrate(self, empty_conveyor_scan: np.ndarray) -> bool:
        """
        Calibrate the processor using an empty conveyor scan.
        
        Args:
            empty_conveyor_scan: Point cloud of empty conveyor (N, 3)
            
        Returns:
            True if calibration successful
        """
        if empty_conveyor_scan.shape[0] < 100:
            return False
        
        self.conveyor_height = float(np.median(empty_conveyor_scan[:, 2]))
        self._calibration_offset = np.mean(empty_conveyor_scan, axis=0)
        self._is_calibrated = True
        
        return True
    
    def process_point_cloud(
        self,
        point_cloud: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[SliceGeometry]:
        """
        Process a point cloud to extract meat slice geometries.
        
        Args:
            point_cloud: Raw point cloud data (N, 3) in mm
            timestamp: Optional timestamp for tracking
            
        Returns:
            List of detected SliceGeometry objects
        """
        if point_cloud.shape[0] < 10:
            return []
        
        filtered_cloud = self._filter_noise(point_cloud)
        
        object_cloud = self._remove_ground_plane(filtered_cloud)
        
        if object_cloud.shape[0] < 10:
            return []
        
        segments = self._segment_slices(object_cloud)
        
        geometries = []
        for segment in segments:
            geometry = self._extract_geometry(segment)
            if geometry is not None:
                geometries.append(geometry)
        
        return geometries
    
    def _filter_noise(self, point_cloud: np.ndarray) -> np.ndarray:
        """Remove noise and outliers from point cloud."""
        centroid = np.mean(point_cloud, axis=0)
        distances = np.linalg.norm(point_cloud - centroid, axis=1)
        
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        mask = distances < (mean_dist + 2 * std_dist)
        
        return point_cloud[mask]
    
    def _remove_ground_plane(self, point_cloud: np.ndarray) -> np.ndarray:
        """Remove points belonging to the conveyor surface."""
        height_threshold = self.conveyor_height + self.noise_threshold
        
        mask = point_cloud[:, 2] > height_threshold
        
        return point_cloud[mask]
    
    def _segment_slices(
        self,
        point_cloud: np.ndarray
    ) -> List[np.ndarray]:
        """Segment point cloud into individual meat slices."""
        if point_cloud.shape[0] < 10:
            return []
        
        x_min, y_min = point_cloud[:, 0].min(), point_cloud[:, 1].min()
        x_max, y_max = point_cloud[:, 0].max(), point_cloud[:, 1].max()
        
        grid_w = int((x_max - x_min) / self.voxel_size) + 1
        grid_h = int((y_max - y_min) / self.voxel_size) + 1
        
        if grid_w <= 0 or grid_h <= 0:
            return []
        
        occupancy = np.zeros((grid_w, grid_h), dtype=np.int32)
        point_indices = {}
        
        for idx, point in enumerate(point_cloud):
            gx = int((point[0] - x_min) / self.voxel_size)
            gy = int((point[1] - y_min) / self.voxel_size)
            gx = min(gx, grid_w - 1)
            gy = min(gy, grid_h - 1)
            
            occupancy[gx, gy] = 1
            
            key = (gx, gy)
            if key not in point_indices:
                point_indices[key] = []
            point_indices[key].append(idx)
        
        labeled, num_features = ndimage.label(occupancy)
        
        segments = []
        for label_id in range(1, num_features + 1):
            mask = labeled == label_id
            
            indices = []
            for gx in range(grid_w):
                for gy in range(grid_h):
                    if mask[gx, gy] and (gx, gy) in point_indices:
                        indices.extend(point_indices[(gx, gy)])
            
            if len(indices) > 10:
                segment = point_cloud[indices]
                segments.append(segment)
        
        return segments
    
    def _extract_geometry(
        self,
        segment: np.ndarray
    ) -> Optional[SliceGeometry]:
        """Extract geometry from a segmented point cloud."""
        if segment.shape[0] < 10:
            return None
        
        centroid = np.mean(segment, axis=0)
        
        z_values = segment[:, 2]
        min_z = np.min(z_values)
        max_z = np.max(z_values)
        thickness = max_z - min_z
        
        if thickness < 3 or thickness > 50:
            return None
        
        xy_points = segment[:, :2]
        
        try:
            hull = ConvexHull(xy_points)
            hull_points = xy_points[hull.vertices]
            
            area = hull.volume
            
            if area < self.min_slice_area or area > self.max_slice_area:
                return None
            
        except Exception:
            x_range = np.max(xy_points[:, 0]) - np.min(xy_points[:, 0])
            y_range = np.max(xy_points[:, 1]) - np.min(xy_points[:, 1])
            area = x_range * y_range
            hull_points = xy_points
        
        width, length, orientation = self._compute_oriented_bbox(xy_points)
        
        shape_mask = self._create_shape_mask(xy_points, width, length)
        
        volume = area * thickness
        
        confidence = self._compute_confidence(segment, area, thickness)
        
        return SliceGeometry(
            width=width,
            length=length,
            thickness=thickness,
            centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            orientation=orientation,
            area=area,
            volume=volume,
            shape_mask=shape_mask,
            point_cloud=segment,
            confidence=confidence
        )
    
    def _compute_oriented_bbox(
        self,
        points: np.ndarray
    ) -> Tuple[float, float, float]:
        """Compute oriented bounding box dimensions and angle."""
        if points.shape[0] < 3:
            x_range = np.max(points[:, 0]) - np.min(points[:, 0])
            y_range = np.max(points[:, 1]) - np.min(points[:, 1])
            return max(x_range, 1.0), max(y_range, 1.0), 0.0
        
        centered = points - np.mean(points, axis=0)
        
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sort_idx]
        
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
        angle_deg = np.degrees(angle)
        
        rotation_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle), np.cos(-angle)]
        ])
        
        rotated = centered @ rotation_matrix.T
        
        width = np.max(rotated[:, 0]) - np.min(rotated[:, 0])
        length = np.max(rotated[:, 1]) - np.min(rotated[:, 1])
        
        return max(width, 1.0), max(length, 1.0), angle_deg
    
    def _create_shape_mask(
        self,
        points: np.ndarray,
        width: float,
        length: float,
        resolution: float = 5.0
    ) -> np.ndarray:
        """Create a 2D binary mask of the slice shape."""
        w_cells = max(1, int(width / resolution))
        l_cells = max(1, int(length / resolution))
        
        mask = np.zeros((w_cells, l_cells), dtype=np.float32)
        
        x_min, y_min = np.min(points, axis=0)
        
        for point in points:
            gx = int((point[0] - x_min) / resolution)
            gy = int((point[1] - y_min) / resolution)
            gx = min(gx, w_cells - 1)
            gy = min(gy, l_cells - 1)
            mask[gx, gy] = 1.0
        
        mask = ndimage.binary_fill_holes(mask).astype(np.float32)
        
        return mask
    
    def _compute_confidence(
        self,
        segment: np.ndarray,
        area: float,
        thickness: float
    ) -> float:
        """Compute confidence score for the geometry extraction."""
        point_density = segment.shape[0] / (area / 100)
        density_score = min(point_density / 10, 1.0)
        
        if 5 <= thickness <= 40:
            thickness_score = 1.0
        else:
            thickness_score = 0.5
        
        if self.min_slice_area <= area <= self.max_slice_area:
            area_score = 1.0
        else:
            area_score = 0.5
        
        confidence = (density_score + thickness_score + area_score) / 3
        
        return confidence
    
    def simulate_scan(
        self,
        width: float,
        length: float,
        thickness: float,
        noise_level: float = 1.0
    ) -> np.ndarray:
        """
        Simulate a LiDAR scan of a meat slice for testing.
        
        Args:
            width: Slice width in mm
            length: Slice length in mm
            thickness: Slice thickness in mm
            noise_level: Standard deviation of noise in mm
            
        Returns:
            Simulated point cloud (N, 3)
        """
        points_per_mm2 = 0.5
        num_points = int(width * length * points_per_mm2)
        
        x = np.random.uniform(0, width, num_points)
        y = np.random.uniform(0, length, num_points)
        
        center_x, center_y = width / 2, length / 2
        dist_x = np.abs(x - center_x) / (width / 2)
        dist_y = np.abs(y - center_y) / (length / 2)
        edge_dist = np.maximum(dist_x, dist_y)
        
        keep_mask = edge_dist < 0.9 + np.random.uniform(-0.2, 0.1, num_points)
        x = x[keep_mask]
        y = y[keep_mask]
        
        z = np.random.uniform(
            self.conveyor_height + 10,
            self.conveyor_height + 10 + thickness,
            len(x)
        )
        
        x += np.random.normal(0, noise_level, len(x))
        y += np.random.normal(0, noise_level, len(y))
        z += np.random.normal(0, noise_level / 2, len(z))
        
        return np.column_stack([x, y, z])


class ConveyorTracker:
    """
    Tracks meat slices on a moving conveyor belt.
    
    Handles:
    - Slice detection and tracking across frames
    - Position prediction based on conveyor speed
    - Handoff timing for robot pickup
    """
    
    def __init__(
        self,
        conveyor_speed: float = 100.0,  # mm/s
        scan_frequency: float = 30.0,  # Hz
        pickup_zone_x: float = 500.0  # mm from scanner
    ):
        self.conveyor_speed = conveyor_speed
        self.scan_frequency = scan_frequency
        self.pickup_zone_x = pickup_zone_x
        
        self.tracked_slices: Dict[int, Dict[str, Any]] = {}
        self.next_slice_id = 0
        self.processor = LiDARProcessor()
    
    def update(
        self,
        point_cloud: np.ndarray,
        timestamp: float
    ) -> List[Dict[str, Any]]:
        """
        Update tracking with new scan data.
        
        Args:
            point_cloud: New LiDAR scan
            timestamp: Current timestamp in seconds
            
        Returns:
            List of slices ready for pickup
        """
        geometries = self.processor.process_point_cloud(point_cloud, timestamp)
        
        for geometry in geometries:
            matched = False
            for slice_id, tracked in self.tracked_slices.items():
                predicted_x = tracked["last_x"] + self.conveyor_speed * (
                    timestamp - tracked["last_timestamp"]
                )
                
                if abs(geometry.centroid[0] - predicted_x) < 50:
                    tracked["geometry"] = geometry
                    tracked["last_x"] = geometry.centroid[0]
                    tracked["last_timestamp"] = timestamp
                    matched = True
                    break
            
            if not matched:
                self.tracked_slices[self.next_slice_id] = {
                    "geometry": geometry,
                    "last_x": geometry.centroid[0],
                    "last_timestamp": timestamp,
                    "first_seen": timestamp
                }
                self.next_slice_id += 1
        
        ready_for_pickup = []
        to_remove = []
        
        for slice_id, tracked in self.tracked_slices.items():
            predicted_x = tracked["last_x"] + self.conveyor_speed * (
                timestamp - tracked["last_timestamp"]
            )
            
            if abs(predicted_x - self.pickup_zone_x) < 30:
                ready_for_pickup.append({
                    "slice_id": slice_id,
                    "geometry": tracked["geometry"],
                    "predicted_position": (
                        predicted_x,
                        tracked["geometry"].centroid[1],
                        tracked["geometry"].centroid[2]
                    ),
                    "time_in_zone": 0.3
                })
            
            if predicted_x > self.pickup_zone_x + 100:
                to_remove.append(slice_id)
        
        for slice_id in to_remove:
            del self.tracked_slices[slice_id]
        
        return ready_for_pickup
    
    def get_next_slice(self) -> Optional[Dict[str, Any]]:
        """Get the next slice approaching the pickup zone."""
        if not self.tracked_slices:
            return None
        
        closest = None
        closest_dist = float("inf")
        
        for slice_id, tracked in self.tracked_slices.items():
            dist = self.pickup_zone_x - tracked["last_x"]
            if 0 < dist < closest_dist:
                closest_dist = dist
                closest = {
                    "slice_id": slice_id,
                    "geometry": tracked["geometry"],
                    "distance_to_pickup": dist,
                    "time_to_pickup": dist / self.conveyor_speed
                }
        
        return closest
