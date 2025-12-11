"""
3D Bin-Packing Environment for Meat Slice Placement

This environment simulates a cube container (210x210x250mm) that needs to be filled
with meat slices of varying sizes and thicknesses. Uses a layer-based approach with
bottom-left-fill strategy to create compact, precise layers that fill every space.

Key features:
- Layer-based packing: complete each layer before starting the next
- Bottom-left-fill: systematic placement from corner to fill gaps
- Cavity penalty: heavily penalize unreachable air pockets
- Flatness reward: bonus for completing flat, uniform layers
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from enum import IntEnum


class PlacementStrategy(IntEnum):
    BOTTOM_LEFT_FILL = 0
    BEST_FIT = 1
    AGENT_CHOICE = 2


@dataclass
class MeatSlice:
    """
    Represents a meat slice with its 3D properties.
    
    Supports wedge-shaped slices with variable thickness across the surface.
    thickness_min and thickness_max define the gradient (e.g., 5mm on one side, 20mm on the other).
    The thickness_map stores per-voxel thickness values.
    """
    
    width: float  # mm (50-200)
    length: float  # mm (50-200)
    thickness_min: float = 5.0  # mm - minimum thickness (one edge)
    thickness_max: float = 40.0  # mm - maximum thickness (opposite edge)
    shape_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    thickness_map: np.ndarray = field(default_factory=lambda: np.array([]))
    slice_id: int = 0
    wedge_direction: int = 0  # 0=x-axis, 1=y-axis, 2=diagonal
    
    def __post_init__(self):
        if self.shape_mask.size == 0:
            self.shape_mask = self._generate_irregular_shape()
        if self.thickness_map.size == 0:
            self.thickness_map = self._generate_thickness_map()
    
    def _generate_irregular_shape(self, irregularity: float = 0.3) -> np.ndarray:
        """
        Generate a meat slice shape with configurable irregularity.
        
        Args:
            irregularity: 0.0 = perfect rectangle, 1.0 = very irregular
                         Default 0.3 for mostly rectangular with slight edge variation
        """
        resolution = 5  # mm per voxel
        w_voxels = max(1, int(self.width / resolution))
        l_voxels = max(1, int(self.length / resolution))
        
        mask = np.ones((w_voxels, l_voxels), dtype=np.float32)
        
        if irregularity > 0:
            center_x, center_y = w_voxels / 2, l_voxels / 2
            for i in range(w_voxels):
                for j in range(l_voxels):
                    dist_x = abs(i - center_x) / max(w_voxels / 2, 1)
                    dist_y = abs(j - center_y) / max(l_voxels / 2, 1)
                    edge_dist = max(dist_x, dist_y)
                    
                    threshold = 1.0 - irregularity * 0.3
                    if edge_dist > threshold:
                        noise = np.random.uniform(0, irregularity * 0.5)
                        if edge_dist + noise > 1.0:
                            mask[i, j] = 0
        
        return mask
    
    def _generate_thickness_map(self) -> np.ndarray:
        """
        Generate a thickness map for wedge-shaped slices.
        
        Creates a gradient from thickness_min to thickness_max across the slice,
        simulating real meat slices that are thicker on one side.
        """
        h, w = self.shape_mask.shape
        
        if self.wedge_direction == 0:
            gradient = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
            gradient = np.broadcast_to(gradient, (h, w)).copy()
        elif self.wedge_direction == 1:
            gradient = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
            gradient = np.broadcast_to(gradient, (h, w)).copy()
        else:
            x_grad = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, np.newaxis]
            y_grad = np.linspace(0.0, 1.0, w, dtype=np.float32)[np.newaxis, :]
            gradient = (x_grad + y_grad) / 2.0
        
        thickness_range = self.thickness_max - self.thickness_min
        thickness_map = self.thickness_min + gradient * thickness_range
        
        thickness_map = thickness_map * self.shape_mask
        
        return thickness_map.astype(np.float32)
    
    @property
    def thickness(self) -> float:
        """Average thickness for backward compatibility."""
        if self.thickness_map.size == 0:
            return (self.thickness_min + self.thickness_max) / 2
        active = self.thickness_map[self.shape_mask > 0]
        return float(np.mean(active)) if active.size > 0 else self.thickness_min
    
    def get_volume(self) -> float:
        """Calculate the volume of the meat slice in mm^3."""
        resolution = 5.0
        voxel_area = resolution ** 2
        return float(np.sum(self.thickness_map) * voxel_area)
    
    def rotate(self, angle: int) -> "MeatSlice":
        """Rotate the slice by 90-degree increments."""
        rotations = (angle // 90) % 4
        new_mask = np.rot90(self.shape_mask, rotations)
        new_thickness_map = np.rot90(self.thickness_map, rotations)
        new_width = self.length if rotations % 2 else self.width
        new_length = self.width if rotations % 2 else self.length
        
        active_thickness = new_thickness_map[new_mask > 0]
        new_min = float(active_thickness.min()) if active_thickness.size > 0 else self.thickness_min
        new_max = float(active_thickness.max()) if active_thickness.size > 0 else self.thickness_max
        
        return MeatSlice(
            width=new_width,
            length=new_length,
            thickness_min=new_min,
            thickness_max=new_max,
            shape_mask=new_mask,
            thickness_map=new_thickness_map,
            slice_id=self.slice_id,
            wedge_direction=(self.wedge_direction + rotations) % 4
        )


@dataclass
class PlacedSlice:
    """A meat slice that has been placed in the cube."""
    
    slice: MeatSlice
    x: float  # position in mm
    y: float  # position in mm
    z: float  # height in mm
    rotation: int  # 0, 90, 180, 270 degrees


class CubeState:
    """
    Represents the current state of the cube being filled.
    
    Uses STRICT layer-based packing: each layer MUST be completed to 95%
    coverage before starting the next layer. This is a HARD physical constraint
    because the vacuum gripper cannot reach lower positions once slices are
    placed higher up.
    
    Key constraint: Once you place a slice at height H, you can NEVER place
    another slice at height < H in the same area because the gripper would
    collide with the higher slice.
    """
    
    LAYER_COVERAGE_THRESHOLD = 0.95  # 95% minimum fill per layer
    LAYER_FLATNESS_THRESHOLD = 0.85
    CAVITY_PENALTY_MULTIPLIER = 10.0
    LAYER_THICKNESS_MM = 25.0  # Target layer thickness in mm
    
    def __init__(
        self,
        width: float = 210.0,
        length: float = 210.0,
        height: float = 250.0,
        resolution: float = 5.0
    ):
        self.width = width
        self.length = length
        self.height = height
        self.resolution = resolution
        
        self.w_voxels = int(width / resolution)
        self.l_voxels = int(length / resolution)
        self.h_voxels = int(height / resolution)
        
        self.height_map = np.zeros((self.w_voxels, self.l_voxels), dtype=np.float32)
        self.occupancy = np.zeros(
            (self.w_voxels, self.l_voxels, self.h_voxels), dtype=np.float32
        )
        self.placed_slices: List[PlacedSlice] = []
        self.total_volume_filled = 0.0
        self.max_volume = width * length * height
        
        self.layer_thickness_voxels = int(self.LAYER_THICKNESS_MM / resolution)
        self.current_layer_index = 0
        self.current_layer_floor_voxel = 0
        self.current_layer_ceiling_voxel = self.layer_thickness_voxels
        
        self.current_layer_height = 0.0
        self.current_layer_target = 0.0
        self.layers_completed = 0
        self.total_cavity_volume = 0.0
        self.layer_complete = False
    
    def reset(self):
        """Reset the cube to empty state."""
        self.height_map.fill(0)
        self.occupancy.fill(0)
        self.placed_slices.clear()
        self.total_volume_filled = 0.0
        self.current_layer_height = 0.0
        self.current_layer_target = 0.0
        self.layers_completed = 0
        self.total_cavity_volume = 0.0
        self.current_layer_index = 0
        self.current_layer_floor_voxel = 0
        self.current_layer_ceiling_voxel = self.layer_thickness_voxels
        self.layer_complete = False

    def get_layer_coverage(self) -> float:
        """
        Calculate coverage of the current layer (0-1).
        
        Coverage is defined as the percentage of cells that have been filled
        to at least the current layer floor height. For the first layer,
        any cell with height > 0 counts as covered.
        """
        total_cells = self.w_voxels * self.l_voxels
        
        if self.current_layer_index == 0:
            cells_covered = np.sum(self.height_map > 0)
        else:
            cells_covered = np.sum(self.height_map >= self.current_layer_floor_voxel)
        
        return cells_covered / total_cells

    def get_layer_flatness(self) -> float:
        """Calculate flatness within the current layer band."""
        if self.current_layer_target == 0:
            return 1.0
        target_voxel = int(self.current_layer_target)
        tolerance = 2
        in_band = (self.height_map >= target_voxel - tolerance) & (self.height_map <= target_voxel + tolerance)
        return np.sum(in_band) / (self.w_voxels * self.l_voxels)

    def find_bottom_left_position(self, slice: MeatSlice) -> Tuple[int, int, float]:
        """
        Find the best bottom-left-fill position for a slice.
        
        Strategy: Start from bottom-left corner, scan row by row,
        find the first position where the slice fits at the lowest height.
        """
        mask = slice.shape_mask
        h, w = mask.shape
        
        best_pos = (-1, -1)
        best_height = float('inf')
        best_gap_score = float('inf')
        
        for x in range(self.w_voxels - h + 1):
            for y in range(self.l_voxels - w + 1):
                can_place, base_height = self.can_place(slice, x, y)
                if not can_place:
                    continue
                
                gap_score = self._calculate_gap_score(slice, x, y, base_height)
                
                if base_height < best_height or (base_height == best_height and gap_score < best_gap_score):
                    best_height = base_height
                    best_gap_score = gap_score
                    best_pos = (x, y)
        
        return best_pos[0], best_pos[1], best_height if best_pos[0] >= 0 else 0.0

    def _calculate_gap_score(self, slice: MeatSlice, x_pos: int, y_pos: int, base_height: float) -> float:
        """Calculate gap score - lower is better (fewer gaps created)."""
        mask = slice.shape_mask
        h, w = mask.shape
        base_voxel = int(base_height / self.resolution)
        
        gap_volume = 0.0
        region_heights = self.height_map[x_pos:x_pos+h, y_pos:y_pos+w]
        
        for i in range(h):
            for j in range(w):
                if mask[i, j] > 0:
                    current_height = int(region_heights[i, j])
                    gap_volume += (base_voxel - current_height)
        
        edge_contact = 0
        if x_pos == 0:
            edge_contact += h
        if y_pos == 0:
            edge_contact += w
        if x_pos + h == self.w_voxels:
            edge_contact += h
        if y_pos + w == self.l_voxels:
            edge_contact += w
        
        neighbor_contact = 0
        if x_pos > 0:
            neighbor_contact += np.sum(self.height_map[x_pos-1, y_pos:y_pos+w] > 0)
        if y_pos > 0:
            neighbor_contact += np.sum(self.height_map[x_pos:x_pos+h, y_pos-1] > 0)
        if x_pos + h < self.w_voxels:
            neighbor_contact += np.sum(self.height_map[x_pos+h, y_pos:y_pos+w] > 0)
        if y_pos + w < self.l_voxels:
            neighbor_contact += np.sum(self.height_map[x_pos:x_pos+h, y_pos+w] > 0)
        
        return gap_volume - edge_contact * 2 - neighbor_contact

    def find_all_valid_positions(self, slice: MeatSlice) -> List[Tuple[int, int, int, float]]:
        """Find all valid positions for a slice with all rotations."""
        positions = []
        
        for rotation in range(4):
            rotated = slice.rotate(rotation * 90)
            mask = rotated.shape_mask
            h, w = mask.shape
            
            for x in range(self.w_voxels - h + 1):
                for y in range(self.l_voxels - w + 1):
                    can_place, base_height = self.can_place(rotated, x, y)
                    if can_place:
                        gap_score = self._calculate_gap_score(rotated, x, y, base_height)
                        positions.append((x, y, rotation, gap_score))
        
        positions.sort(key=lambda p: (p[3], p[0], p[1]))
        return positions

    def can_place(
        self,
        slice: MeatSlice,
        x_pos: int,
        y_pos: int,
        enforce_layer_constraint: bool = True
    ) -> Tuple[bool, float]:
        """
        Check if a slice can be placed at the given position.
        
        STRICT LAYER CONSTRAINT: 
        1. FLOOR: The slice cannot be placed if ANY part of it would touch
           a height below the current layer floor. This is ALWAYS enforced
           because the gripper cannot reach lower positions once slices are
           placed higher up - this is a HARD physical constraint.
        2. CEILING: The slice cannot exceed the current layer ceiling until
           the layer is 95% complete.
        
        Args:
            slice: The meat slice to place
            x_pos, y_pos: Position in voxel coordinates
            enforce_layer_constraint: If True, enforce strict layer-by-layer filling
            
        Returns:
            Tuple of (can_place, base_height_mm)
        """
        mask = slice.shape_mask
        thickness_map = slice.thickness_map
        h, w = mask.shape
        
        if x_pos < 0 or y_pos < 0:
            return False, 0.0
        if x_pos + h > self.w_voxels or y_pos + w > self.l_voxels:
            return False, 0.0
        
        region_heights = self.height_map[x_pos:x_pos+h, y_pos:y_pos+w]
        
        active_mask = mask > 0
        if not np.any(active_mask):
            return False, 0.0
        
        base_heights = region_heights[active_mask]
        local_thickness_voxels = thickness_map[active_mask] / self.resolution
        new_heights = base_heights + local_thickness_voxels
        
        if new_heights.max() > self.h_voxels:
            return False, 0.0
        
        if enforce_layer_constraint:
            min_base_height = base_heights.min()
            if min_base_height < self.current_layer_floor_voxel:
                return False, 0.0
            
            tolerance_voxels = 2
            if new_heights.max() > self.current_layer_ceiling_voxel + tolerance_voxels:
                return False, 0.0
        
        avg_base_height = base_heights.mean() * self.resolution
        return True, avg_base_height
    
    def place_slice_conforming(
        self,
        slice: MeatSlice,
        x_pos: int,
        y_pos: int,
        push_to_wall: bool = True
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Place a slice with CONFORMING behavior - slice adapts to surface below.
        
        The slice "settles" into the existing surface, filling gaps naturally.
        Each voxel of the slice sits on top of the existing height at that position,
        with its local thickness from the thickness_map.
        
        If push_to_wall is True, the robot will push the slice towards the nearest
        wall before releasing the vacuum gripper, eliminating small gaps (few mm).
        
        This eliminates air pockets under flexible meat slices.
        """
        mask = slice.shape_mask
        thickness_map = slice.thickness_map
        h, w = mask.shape
        
        if x_pos < 0 or y_pos < 0:
            return False, {"reward": -1.0, "gap_penalty": 0.0, "flatness": 0.0}
        if x_pos + h > self.w_voxels or y_pos + w > self.l_voxels:
            return False, {"reward": -1.0, "gap_penalty": 0.0, "flatness": 0.0}
        
        push_direction = 'none'
        compression_mm = 0.0
        if push_to_wall:
            new_x, new_y, push_direction, compression_mm = self.push_to_wall(x_pos, y_pos, h, w)
            if new_x != x_pos or new_y != y_pos:
                x_pos, y_pos = new_x, new_y
        
        region_heights = self.height_map[x_pos:x_pos+h, y_pos:y_pos+w].copy()
        
        local_thickness_voxels = thickness_map / self.resolution
        new_heights = region_heights + local_thickness_voxels
        
        max_new_height = np.max(new_heights * mask)
        if max_new_height > self.h_voxels:
            return False, {"reward": -1.0, "gap_penalty": 0.0, "flatness": 0.0}
        
        volume_added = 0.0
        for i in range(h):
            for j in range(w):
                if mask[i, j] > 0:
                    old_height = int(region_heights[i, j])
                    local_thickness = thickness_map[i, j]
                    thickness_voxels = int(local_thickness / self.resolution)
                    new_height = old_height + thickness_voxels
                    
                    self.height_map[x_pos+i, y_pos+j] = new_height
                    
                    for k in range(old_height, new_height):
                        if k < self.h_voxels:
                            self.occupancy[x_pos+i, y_pos+j, k] = 1.0
                    
                    volume_added += local_thickness * (self.resolution ** 2)
        
        avg_base_height = np.mean(region_heights[mask > 0]) * self.resolution
        placed = PlacedSlice(
            slice=slice,
            x=x_pos * self.resolution,
            y=y_pos * self.resolution,
            z=avg_base_height,
            rotation=0
        )
        self.placed_slices.append(placed)
        self.total_volume_filled += volume_added
        
        if self.current_layer_target == 0:
            self.current_layer_target = np.max(self.height_map)
        
        flatness = self._calculate_flatness()
        layer_coverage = self.get_layer_coverage()
        edge_bonus = self._calculate_edge_bonus(x_pos, y_pos, h, w)
        
        utilization = self.total_volume_filled / self.max_volume
        
        reward = (
            utilization * 20.0 +
            flatness * 10.0 +
            edge_bonus * 5.0 +
            layer_coverage * 15.0
        )
        
        if layer_coverage >= self.LAYER_COVERAGE_THRESHOLD:
            reward += 50.0
        
        return True, {
            "reward": reward,
            "gap_penalty": 0.0,
            "flatness": flatness,
            "utilization": utilization,
            "base_height": avg_base_height,
            "layer_coverage": layer_coverage,
            "edge_bonus": edge_bonus,
            "push_direction": push_direction,
            "compression_mm": compression_mm,
            "final_x": x_pos * self.resolution,
            "final_y": y_pos * self.resolution
        }

    def _calculate_edge_bonus(self, x_pos: int, y_pos: int, h: int, w: int) -> float:
        """Calculate bonus for placing against edges/corners (fills spaces better)."""
        bonus = 0.0
        if x_pos == 0:
            bonus += 1.0
        if y_pos == 0:
            bonus += 1.0
        if x_pos + h == self.w_voxels:
            bonus += 1.0
        if y_pos + w == self.l_voxels:
            bonus += 1.0
        if x_pos == 0 and y_pos == 0:
            bonus += 2.0
        if x_pos == 0 and y_pos + w == self.l_voxels:
            bonus += 2.0
        if x_pos + h == self.w_voxels and y_pos == 0:
            bonus += 2.0
        if x_pos + h == self.w_voxels and y_pos + w == self.l_voxels:
            bonus += 2.0
        return bonus

    def push_to_wall(self, x_pos: int, y_pos: int, h: int, w: int) -> Tuple[int, int, str, float]:
        """
        Calculate the optimal push direction to eliminate small gaps against walls.
        
        The robot can push the slice towards the nearest wall before releasing
        the vacuum gripper. This eliminates gaps of a few millimeters and helps
        the flexible slice conform to the wall shape.
        
        IMPORTANT: The slice is pushed 20-30mm BEYOND just touching the wall.
        This compression causes the flexible meat to deform and fill gaps better
        against walls and corners.
        
        Returns:
            Tuple of (new_x, new_y, push_direction, compression_mm)
            push_direction is one of: 'none', 'left', 'right', 'front', 'back', 
                                      'left_front', 'left_back', 'right_front', 'right_back'
            compression_mm is the amount of compression applied (20-30mm)
        """
        dist_to_left = x_pos
        dist_to_right = self.w_voxels - (x_pos + h)
        dist_to_front = y_pos
        dist_to_back = self.l_voxels - (y_pos + w)
        
        push_threshold_voxels = 6  # 30mm (6 voxels * 5mm resolution)
        compression_mm = 25.0  # Push 25mm beyond wall contact for better conforming
        
        new_x, new_y = x_pos, y_pos
        push_direction = 'none'
        actual_compression = 0.0
        
        push_left = dist_to_left > 0 and dist_to_left <= push_threshold_voxels
        push_right = dist_to_right > 0 and dist_to_right <= push_threshold_voxels
        push_front = dist_to_front > 0 and dist_to_front <= push_threshold_voxels
        push_back = dist_to_back > 0 and dist_to_back <= push_threshold_voxels
        
        if push_left and push_front:
            new_x = 0
            new_y = 0
            push_direction = 'left_front'
            actual_compression = compression_mm * 1.4  # Corner gets more compression (diagonal)
        elif push_left and push_back:
            new_x = 0
            new_y = self.l_voxels - w
            push_direction = 'left_back'
            actual_compression = compression_mm * 1.4
        elif push_right and push_front:
            new_x = self.w_voxels - h
            new_y = 0
            push_direction = 'right_front'
            actual_compression = compression_mm * 1.4
        elif push_right and push_back:
            new_x = self.w_voxels - h
            new_y = self.l_voxels - w
            push_direction = 'right_back'
            actual_compression = compression_mm * 1.4
        elif push_left:
            new_x = 0
            push_direction = 'left'
            actual_compression = compression_mm
        elif push_right:
            new_x = self.w_voxels - h
            push_direction = 'right'
            actual_compression = compression_mm
        elif push_front:
            new_y = 0
            push_direction = 'front'
            actual_compression = compression_mm
        elif push_back:
            new_y = self.l_voxels - w
            push_direction = 'back'
            actual_compression = compression_mm
        
        return new_x, new_y, push_direction, actual_compression

    def press_layer(self, compression_ratio: float = 0.9) -> Dict[str, float]:
        """
        Press/compact the current layer to create a flat, uniform surface.
        
        Called after a layer is complete (95% coverage) to:
        1. Flatten the top surface
        2. Compress slightly to remove any small gaps
        3. Advance to the next layer (update layer bounds)
        
        IMPORTANT: After pressing, the layer bounds are advanced so that
        new placements can only go on top of the pressed layer. This enforces
        the strict layer-by-layer constraint.
        
        Args:
            compression_ratio: How much to compress (0.9 = 10% compression)
        
        Returns:
            Metrics about the pressing operation
        """
        if not np.any(self.height_map > 0):
            return {"pressed": False, "new_layer_height": 0.0}
        
        current_max = np.max(self.height_map)
        current_min = np.min(self.height_map[self.height_map > 0])
        
        target_height = current_max * compression_ratio
        
        active_mask = self.height_map > 0
        self.height_map[active_mask] = np.minimum(
            self.height_map[active_mask],
            target_height
        )
        self.height_map[active_mask] = np.maximum(
            self.height_map[active_mask],
            target_height * 0.95
        )
        
        self.current_layer_height = target_height * self.resolution
        self.current_layer_target = target_height
        self.layers_completed += 1
        
        self.current_layer_index += 1
        self.current_layer_floor_voxel = int(target_height)
        self.current_layer_ceiling_voxel = self.current_layer_floor_voxel + self.layer_thickness_voxels
        self.layer_complete = False
        
        new_flatness = self._calculate_flatness()
        
        return {
            "pressed": True,
            "new_layer_height": self.current_layer_height,
            "flatness_after_press": new_flatness,
            "layers_completed": self.layers_completed,
            "current_layer_index": self.current_layer_index,
            "layer_floor_mm": self.current_layer_floor_voxel * self.resolution,
            "layer_ceiling_mm": self.current_layer_ceiling_voxel * self.resolution
        }

    def place_slice(
        self,
        slice: MeatSlice,
        x_pos: int,
        y_pos: int
    ) -> Tuple[bool, Dict[str, float]]:
        """Place a slice at the given position using conforming behavior."""
        return self.place_slice_conforming(slice, x_pos, y_pos)
    
    def _calculate_flatness(self) -> float:
        """Calculate how flat the current top surface is (0-1)."""
        active_mask = self.height_map > 0
        if not np.any(active_mask):
            return 1.0
        
        active_heights = self.height_map[active_mask]
        if len(active_heights) < 2:
            return 1.0
        
        std_dev = np.std(active_heights)
        max_std = self.h_voxels / 4
        flatness = 1.0 - min(std_dev / max_std, 1.0)
        
        return flatness
    
    def get_observation(self) -> np.ndarray:
        """Get the current state as an observation array."""
        normalized_height_map = self.height_map / self.h_voxels
        return normalized_height_map.flatten()
    
    def get_fill_percentage(self) -> float:
        """Get the percentage of cube volume filled."""
        return (self.total_volume_filled / self.max_volume) * 100


class MeatPackingEnv(gym.Env):
    """
    Gymnasium environment for meat slice packing.
    
    The agent must decide where to place incoming meat slices to maximize
    space utilization while maintaining flat layers.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        cube_width: float = 210.0,
        cube_length: float = 210.0,
        cube_height: float = 250.0,
        resolution: float = 5.0,
        max_slices: int = 50,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        
        self.cube_width = cube_width
        self.cube_length = cube_length
        self.cube_height = cube_height
        self.resolution = resolution
        self.max_slices = max_slices
        self.render_mode = render_mode
        
        self.cube = CubeState(cube_width, cube_length, cube_height, resolution)
        
        self.w_voxels = self.cube.w_voxels
        self.l_voxels = self.cube.l_voxels
        
        self.action_space = spaces.Dict({
            "x_position": spaces.Discrete(self.w_voxels),
            "y_position": spaces.Discrete(self.l_voxels),
            "rotation": spaces.Discrete(4),
            "gripper_pattern": spaces.MultiBinary(5)
        })
        
        self.flat_action_space = spaces.Discrete(
            self.w_voxels * self.l_voxels * 4
        )
        
        slice_obs_size = 3
        height_map_size = self.w_voxels * self.l_voxels
        
        self.observation_space = spaces.Dict({
            "height_map": spaces.Box(
                low=0, high=1, shape=(height_map_size,), dtype=np.float32
            ),
            "current_slice": spaces.Box(
                low=0, high=1, shape=(slice_obs_size,), dtype=np.float32
            ),
            "fill_percentage": spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
        })
        
        self.current_slice: Optional[MeatSlice] = None
        self.slices_placed = 0
        self.episode_reward = 0.0
        self.slice_queue: List[MeatSlice] = []
    
    def _generate_random_slice(self) -> MeatSlice:
        """
        Generate a random meat slice with realistic dimensions.
        
        Creates wedge-shaped slices with variable thickness (e.g., 5mm on one side,
        20mm on the other) to simulate real meat slices.
        """
        width = np.random.uniform(50, 200)
        length = np.random.uniform(50, 200)
        
        thickness_min = np.random.uniform(5, 15)
        thickness_max = np.random.uniform(thickness_min + 5, 40)
        
        wedge_direction = np.random.randint(0, 3)
        
        return MeatSlice(
            width=width,
            length=length,
            thickness_min=thickness_min,
            thickness_max=thickness_max,
            slice_id=self.slices_placed,
            wedge_direction=wedge_direction
        )
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        height_map = self.cube.get_observation()
        
        if self.current_slice is not None:
            slice_obs = np.array([
                self.current_slice.width / 200.0,
                self.current_slice.length / 200.0,
                self.current_slice.thickness / 40.0
            ], dtype=np.float32)
        else:
            slice_obs = np.zeros(3, dtype=np.float32)
        
        fill_pct = np.array(
            [self.cube.get_fill_percentage() / 100.0], dtype=np.float32
        )
        
        return {
            "height_map": height_map,
            "current_slice": slice_obs,
            "fill_percentage": fill_pct
        }
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.cube.reset()
        self.slices_placed = 0
        self.episode_reward = 0.0
        
        self.slice_queue = [
            self._generate_random_slice() for _ in range(self.max_slices)
        ]
        self.current_slice = self.slice_queue.pop(0) if self.slice_queue else None
        
        observation = self._get_observation()
        info = {
            "fill_percentage": 0.0,
            "slices_placed": 0,
            "flatness": 1.0
        }
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        x_pos = action // (self.l_voxels * 4)
        remainder = action % (self.l_voxels * 4)
        y_pos = remainder // 4
        rotation = remainder % 4
        
        if self.current_slice is None:
            return self._get_observation(), 0.0, True, False, {}
        
        rotated_slice = self.current_slice.rotate(rotation * 90)
        
        success, metrics = self.cube.place_slice(rotated_slice, x_pos, y_pos)
        
        if success:
            reward = metrics["reward"]
            self.slices_placed += 1
            
            if self.slice_queue:
                self.current_slice = self.slice_queue.pop(0)
            else:
                self.current_slice = None
        else:
            reward = -0.5
        
        self.episode_reward += reward
        
        terminated = (
            self.current_slice is None or
            self.cube.get_fill_percentage() >= 95.0
        )
        
        truncated = self.slices_placed >= self.max_slices
        
        observation = self._get_observation()
        info = {
            "fill_percentage": self.cube.get_fill_percentage(),
            "slices_placed": self.slices_placed,
            "flatness": self.cube._calculate_flatness(),
            "placement_success": success,
            "episode_reward": self.episode_reward
        }
        
        if success:
            info.update(metrics)
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None
    
    def _render_frame(self) -> np.ndarray:
        """Render the current state as an RGB array."""
        img_size = 400
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        scale = img_size / max(self.w_voxels, self.l_voxels)
        
        for i in range(self.w_voxels):
            for j in range(self.l_voxels):
                height = self.cube.height_map[i, j]
                intensity = int(255 * (1 - height / self.cube.h_voxels))
                x1, y1 = int(i * scale), int(j * scale)
                x2, y2 = int((i + 1) * scale), int((j + 1) * scale)
                img[x1:x2, y1:y2] = [intensity, intensity, 255]
        
        return img
    
    def get_placement_command(
        self,
        action: int
    ) -> Dict[str, Any]:
        """Convert action to robot placement command."""
        x_pos = action // (self.l_voxels * 4)
        remainder = action % (self.l_voxels * 4)
        y_pos = remainder // 4
        rotation = remainder % 4
        
        x_mm = x_pos * self.resolution
        y_mm = y_pos * self.resolution
        
        _, base_height = self.cube.can_place(
            self.current_slice.rotate(rotation * 90) if self.current_slice else MeatSlice(50, 50, 5),
            x_pos,
            y_pos
        )
        
        return {
            "x": x_mm,
            "y": y_mm,
            "z": base_height,
            "rotation": rotation * 90,
            "gripper_pattern": [1, 1, 1, 1, 1]
        }
