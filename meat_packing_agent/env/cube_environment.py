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
    
    Shape analysis features (computed in __post_init__):
    - corner_score: how much the slice has pronounced corners (0-1)
    - straight_edge_score: how straight the edges are (0-1)
    - roundness_score: how round/compact the slice is (0-1)
    - edge_straightness: per-side straightness scores
    - corner_masses: per-corner density scores
    """
    
    width: float  # mm (50-200)
    length: float  # mm (50-200)
    thickness_min: float = 5.0  # mm - minimum thickness (one edge)
    thickness_max: float = 40.0  # mm - maximum thickness (opposite edge)
    shape_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    thickness_map: np.ndarray = field(default_factory=lambda: np.array([]))
    slice_id: int = 0
    wedge_direction: int = 0  # 0=x-axis, 1=y-axis, 2=diagonal
    corner_score: float = 0.0
    straight_edge_score: float = 0.0
    roundness_score: float = 0.0
    edge_straightness: dict = field(default_factory=dict)
    corner_masses: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.shape_mask.size == 0:
            self.shape_mask = self._generate_irregular_shape()
        if self.thickness_map.size == 0:
            self.thickness_map = self._generate_thickness_map()
        self._analyze_shape()
    
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
    
    def _analyze_shape(self):
        """
        Analyze the shape of the slice to compute shape features.
        
        These features are used to match slices to appropriate positions:
        - corner_score: slices with pronounced corners -> cube corners
        - straight_edge_score: slices with straight edges -> cube walls
        - roundness_score: round/compact slices -> cube center
        """
        mask = self.shape_mask.astype(bool)
        ys, xs = np.where(mask)
        
        if xs.size == 0:
            self.corner_score = 0.0
            self.straight_edge_score = 0.0
            self.roundness_score = 0.0
            self.edge_straightness = {}
            self.corner_masses = {}
            return
        
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        height = y_max - y_min + 1
        width = x_max - x_min + 1
        area = mask.sum()
        
        # 1) Compute boundary
        boundary = self._compute_boundary(mask)
        perimeter = boundary.sum() if boundary.sum() > 0 else 1.0
        
        # 2) Roundness: 4πA / P^2 (≈1 for circle, smaller for jagged/elongated)
        roundness = 4.0 * np.pi * area / (perimeter ** 2)
        self.roundness_score = float(np.clip(roundness, 0.0, 1.0))
        
        # 3) Straight edges: how much of each side of the bounding box is boundary
        self.edge_straightness = self._compute_edge_straightness(boundary, x_min, x_max, y_min, y_max)
        self.straight_edge_score = float(max(self.edge_straightness.values()) if self.edge_straightness else 0.0)
        
        # 4) Corner mass: how much area is concentrated near each bounding-box corner
        self.corner_masses = self._compute_corner_masses(mask, x_min, x_max, y_min, y_max)
        max_corner_mass = max(self.corner_masses.values()) if self.corner_masses else 0.0
        self.corner_score = float(max_corner_mass)
    
    def _compute_boundary(self, mask: np.ndarray) -> np.ndarray:
        """Compute boundary pixels (pixels that are on the edge of the shape)."""
        up = np.roll(mask, 1, axis=0)
        down = np.roll(mask, -1, axis=0)
        left = np.roll(mask, 1, axis=1)
        right = np.roll(mask, -1, axis=1)
        interior = mask & up & down & left & right
        boundary = mask & ~interior
        # Clear rolled artifacts at edges
        boundary[0, :] &= mask[0, :]
        boundary[-1, :] &= mask[-1, :]
        boundary[:, 0] &= mask[:, 0]
        boundary[:, -1] &= mask[:, -1]
        return boundary
    
    def _compute_edge_straightness(self, boundary: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int) -> dict:
        """Compute how straight each edge of the slice is (0-1 per side)."""
        h = y_max - y_min + 1
        w = x_max - x_min + 1
        
        # Take a 1-pixel band along each side of the bounding box
        left_band = boundary[y_min:y_max+1, x_min:x_min+1] if x_min < boundary.shape[1] else np.array([])
        right_band = boundary[y_min:y_max+1, x_max:x_max+1] if x_max < boundary.shape[1] else np.array([])
        top_band = boundary[y_min:y_min+1, x_min:x_max+1] if y_min < boundary.shape[0] else np.array([])
        bottom_band = boundary[y_max:y_max+1, x_min:x_max+1] if y_max < boundary.shape[0] else np.array([])
        
        scores = {}
        scores['left'] = float(left_band.sum() / h) if h > 0 and left_band.size > 0 else 0.0
        scores['right'] = float(right_band.sum() / h) if h > 0 and right_band.size > 0 else 0.0
        scores['top'] = float(top_band.sum() / w) if w > 0 and top_band.size > 0 else 0.0
        scores['bottom'] = float(bottom_band.sum() / w) if w > 0 and bottom_band.size > 0 else 0.0
        
        return {k: float(np.clip(v, 0.0, 1.0)) for k, v in scores.items()}
    
    def _compute_corner_masses(self, mask: np.ndarray, x_min: int, x_max: int, y_min: int, y_max: int) -> dict:
        """Compute how much mass is concentrated in each corner of the bounding box."""
        h = y_max - y_min + 1
        w = x_max - x_min + 1
        
        # Sample a small patch in each corner of the bounding box
        patch_size_y = max(3, int(0.25 * h))
        patch_size_x = max(3, int(0.25 * w))
        
        tl = mask[y_min:y_min+patch_size_y, x_min:x_min+patch_size_x]
        tr = mask[y_min:y_min+patch_size_y, max(0, x_max-patch_size_x+1):x_max+1]
        bl = mask[max(0, y_max-patch_size_y+1):y_max+1, x_min:x_min+patch_size_x]
        br = mask[max(0, y_max-patch_size_y+1):y_max+1, max(0, x_max-patch_size_x+1):x_max+1]
        
        # Normalize by patch area to get a [0,1] density
        denom = float(patch_size_y * patch_size_x)
        return {
            'top_left': float(tl.sum() / denom) if denom > 0 else 0.0,
            'top_right': float(tr.sum() / denom) if denom > 0 else 0.0,
            'bottom_left': float(bl.sum() / denom) if denom > 0 else 0.0,
            'bottom_right': float(br.sum() / denom) if denom > 0 else 0.0,
        }
    
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
        
        # Create new slice (shape analysis will be recomputed in __post_init__)
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
    push_direction: str = 'none'  # Direction slice was pushed: none, left, right, front, back, or corner directions
    zone: str = 'center'  # Position zone: 'corner', 'edge', or 'center'
    layer_index: int = 0  # Which layer this slice was placed on


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
    
    LAYER_COVERAGE_THRESHOLD = 0.95  # 95% area coverage per layer
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
        
        # Per-layer tracking for corners->perimeter->center rule
        # Track which TRUE corners (4 exact positions) have been filled this layer
        self.corners_filled_this_layer: set = set()  # Set of corner names: 'front_left', 'front_right', 'back_left', 'back_right'
        self.layer_filling_stage = 'corners'  # 'corners', 'edges', 'center'
        
        # BRICK WALL PRINCIPLE: Store the previous layer's coverage map
        # This is used to stagger slices between layers (like bricks in a wall)
        # The layer above should cover the gaps/seams of the layer below
        self.last_layer_coverage: np.ndarray = None  # 2D bool array: True where meat exists
    
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
        self.corners_filled_this_layer = set()
        self.layer_filling_stage = 'corners'
        self.last_layer_coverage = None  # Reset brick wall tracking

    def get_layer_coverage(self) -> float:
        """
        Calculate coverage of the current layer (0-1) using OCCUPANCY grid.
        
        Coverage is defined as the percentage of cells that have at least one
        occupied voxel within the current layer band. This is more accurate
        than heightmap-based calculation because:
        1. It's not affected by press_layer() flattening
        2. It directly measures actual meat presence in the layer
        
        A cell is "covered" if ANY voxel in the layer band is occupied.
        """
        total_cells = self.w_voxels * self.l_voxels
        
        floor = self.current_layer_floor_voxel
        ceil = min(floor + self.layer_thickness_voxels, self.h_voxels)
        
        # Get the occupancy band for this layer
        band = self.occupancy[:, :, floor:ceil]
        
        # A cell is covered if at least one voxel in the band is occupied
        covered_cells = np.any(band > 0, axis=2).sum()
        
        return covered_cells / total_cells

    def count_14mm_holes(self) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Count 14x14mm holes (3x3 voxel windows) in the current layer.
        
        The SM 7000 machine cuts 14mm x 14mm squares for spiedini.
        ANY 3x3 voxel window (≈15x15mm at 5mm resolution) that is completely
        empty means that square will be wasted product.
        
        Returns:
            Tuple of (hole_count, list of hole positions as (x, y) tuples)
        """
        floor = self.current_layer_floor_voxel
        ceil = min(floor + self.layer_thickness_voxels, self.h_voxels)
        
        # Get the occupancy band for this layer
        band = self.occupancy[:, :, floor:ceil]
        
        # Create 2D coverage map: True if ANY voxel in the column is occupied
        coverage_2d = np.any(band > 0, axis=2)
        
        # Scan for 3x3 windows that are completely empty
        holes = []
        window_size = 3  # 3 voxels = 15mm ≈ 14mm SM7000 cut size
        
        for x in range(self.w_voxels - window_size + 1):
            for y in range(self.l_voxels - window_size + 1):
                window = coverage_2d[x:x+window_size, y:y+window_size]
                if not np.any(window):  # Entire 3x3 window is empty
                    holes.append((x, y))
        
        return len(holes), holes

    def get_hole_positions_for_filling(self) -> List[Tuple[int, int, str]]:
        """
        Get prioritized hole positions for filling.
        
        Returns list of (x, y, zone) tuples, prioritized:
        1. Corner holes (most critical - edges of cube)
        2. Edge holes (along walls)
        3. Center holes
        """
        _, hole_positions = self.count_14mm_holes()
        
        prioritized = []
        for x, y in hole_positions:
            zone = self._classify_position_zone(x, y)
            priority = 0 if zone == 'corner' else (1 if zone == 'edge' else 2)
            prioritized.append((priority, x, y, zone))
        
        # Sort by priority (corners first)
        prioritized.sort(key=lambda p: p[0])
        
        return [(x, y, zone) for _, x, y, zone in prioritized]
    
    def find_position_covering_hole(self, slice: MeatSlice, hole_x: int, hole_y: int, 
                                     hole_zone: str) -> Tuple[int, int, int, float]:
        """
        Find the best position for a slice that COVERS a specific hole.
        
        This is the KEY method for hole-filling mode:
        - The slice footprint MUST overlap with the hole position
        - Prefer positions that push the slice to the wall/corner
        - Use shape matching (corner_score for corners, edge_score for edges)
        
        Args:
            slice: The meat slice to place
            hole_x, hole_y: Position of the hole (3x3 window top-left corner)
            hole_zone: 'corner', 'edge', or 'center'
            
        Returns:
            Tuple of (x, y, rotation, score) or (-1, -1, 0, 0.0) if no valid position
        """
        best_pos = (-1, -1, 0, 0.0)
        best_score = -float('inf')
        
        # Try all 4 rotations
        for rotation in range(4):
            rotated = slice.rotate(rotation * 90)
            mask = rotated.shape_mask
            h, w = mask.shape
            
            # Calculate valid position range that would cover the hole
            # The slice at position (x, y) covers area [x, x+h) x [y, y+w)
            # The hole is at [hole_x, hole_x+3) x [hole_y, hole_y+3)
            # For overlap: x <= hole_x+2 AND x+h > hole_x AND y <= hole_y+2 AND y+w > hole_y
            
            x_min = max(0, hole_x + 3 - h)  # Slice must start before hole ends
            x_max = min(self.w_voxels - h, hole_x + 2)  # Slice must end after hole starts
            y_min = max(0, hole_y + 3 - w)
            y_max = min(self.l_voxels - w, hole_y + 2)
            
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    can_place, base_height = self.can_place(rotated, x, y)
                    if not can_place:
                        continue
                    
                    # Calculate score based on:
                    # 1. Shape matching (corner_score for corners, edge_score for edges)
                    # 2. Push to wall bonus
                    # 3. Coverage of the hole
                    
                    score = 0.0
                    
                    # Shape matching bonus
                    if hole_zone == 'corner':
                        score += rotated.corner_score * 10.0
                    elif hole_zone == 'edge':
                        score += rotated.straight_edge_score * 10.0
                    else:
                        score += rotated.roundness_score * 5.0
                    
                    # Push to wall bonus
                    _, _, push_dir, _ = self.push_to_wall(x, y, h, w)
                    if push_dir != 'none':
                        score += 5.0
                        if push_dir in ['left_front', 'left_back', 'right_front', 'right_back']:
                            score += 10.0  # Extra bonus for corner push
                    
                    # Prefer lower base height (fill from bottom)
                    score -= base_height * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_pos = (x, y, rotation, score)
        
        return best_pos

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

    def _classify_position_zone(self, x_pos: int, y_pos: int, h: int = 1, w: int = 1) -> str:
        """
        Classify a position into a zone: 'corner', 'edge', or 'center'.
        
        Used for perimeter-first filling strategy.
        
        Args:
            x_pos, y_pos: Position in voxel coordinates
            h, w: Optional slice dimensions (default 1 for single point classification)
        """
        edge_threshold = 4  # 4 voxels = 20mm from edge is considered "edge zone"
        
        # For single point classification (holes), check proximity to walls
        if h == 1 and w == 1:
            near_left = x_pos < edge_threshold
            near_right = x_pos >= self.w_voxels - edge_threshold
            near_front = y_pos < edge_threshold
            near_back = y_pos >= self.l_voxels - edge_threshold
            
            walls_near = sum([near_left, near_right, near_front, near_back])
            
            if walls_near >= 2:
                return 'corner'
            elif walls_near == 1:
                return 'edge'
            else:
                return 'center'
        
        # For slice placement, check if slice touches walls
        touches_left = (x_pos == 0)
        touches_right = (x_pos + h == self.w_voxels)
        touches_front = (y_pos == 0)
        touches_back = (y_pos + w == self.l_voxels)
        
        walls_touched = sum([touches_left, touches_right, touches_front, touches_back])
        
        if walls_touched >= 2:
            return 'corner'
        elif walls_touched == 1:
            return 'edge'
        else:
            return 'center'

    def find_perimeter_first_position(self, slice: MeatSlice, floor_only: bool = True) -> Tuple[int, int, float, bool]:
        """
        Find the best position using STRICT perimeter-first strategy per layer.
        
        SACRED RULE: For EACH layer, fill in this EXACT order:
        1. TRUE CORNERS FIRST - Fill all 4 exact corners before anything else
        2. EDGES (PERIMETER) - Positions touching 1 wall, with push_to_wall
        3. CENTER - Positions not touching any wall
        
        ANTI-TOWER RULE: Reject positions where most of the footprint already has
        slices placed (prevents vertical stacking before horizontal fill).
        
        Args:
            slice: The meat slice to place
            floor_only: If True, only accept floor-level positions (default True)
            
        Returns:
            Tuple of (x, y, base_height, is_floor_level)
            If no valid position found, returns (-1, -1, 0.0, False)
        """
        mask = slice.shape_mask
        h, w = mask.shape
        
        layer_coverage = self.get_layer_coverage()
        floor_voxel = self.current_layer_floor_voxel
        layer_thickness_voxels = self.layer_thickness_voxels
        
        # Define the 4 TRUE corner positions for this slice size
        corner_names = {
            (0, 0): 'front_left',
            (0, self.l_voxels - w): 'front_right',
            (self.w_voxels - h, 0): 'back_left',
            (self.w_voxels - h, self.l_voxels - w): 'back_right'
        }
        true_corners = list(corner_names.keys())
        
        # Determine which corners are still unfilled for this layer
        unfilled_corners = [c for c in true_corners 
                          if corner_names[c] not in self.corners_filled_this_layer]
        
        # Update filling stage based on corners filled
        if len(self.corners_filled_this_layer) >= 4:
            if self.layer_filling_stage == 'corners':
                self.layer_filling_stage = 'edges'
        
        # Check if we should move to center stage (edges sufficiently filled)
        if self.layer_filling_stage == 'edges' and layer_coverage >= 0.6:
            self.layer_filling_stage = 'center'
        
        # Pre-compute the layer band ceiling
        ceil = min(floor_voxel + layer_thickness_voxels, self.h_voxels)
        
        def get_brick_score(x: int, y: int) -> float:
            """BRICK WALL PRINCIPLE: Calculate how well this position straddles seams from the layer below.
            
            Like a brick wall - the layer above should cover the gaps/seams of the layer below.
            A good brick position has SOME parts over meat and SOME parts over gaps.
            
            Returns a score from 0 to 0.25:
            - 0.0 = entirely over meat OR entirely over gap (bad - not staggered)
            - 0.25 = 50% over meat, 50% over gap (ideal - perfect stagger)
            
            For the first layer (no previous layer), returns 0 (neutral).
            """
            if self.last_layer_coverage is None:
                return 0.0  # First layer - no staggering needed
            
            region_mask = mask > 0
            if not np.any(region_mask):
                return 0.0
            
            # Get the coverage from the previous layer under this slice footprint
            prev_coverage = self.last_layer_coverage[x:x+h, y:y+w]
            
            # Calculate what fraction of the slice footprint is over meat vs gaps
            below = prev_coverage[region_mask]  # True where meat exists below
            meat_fraction = np.mean(below)  # Fraction over meat
            gap_fraction = 1.0 - meat_fraction  # Fraction over gaps
            
            # Brick score: peaks at 0.5 mix (half over meat, half over gaps)
            # brick_score = meat_fraction * gap_fraction = meat_fraction * (1 - meat_fraction)
            # Max value is 0.25 when meat_fraction = 0.5
            brick_score = meat_fraction * gap_fraction
            
            return brick_score
        
        def is_valid_floor_position(x: int, y: int) -> bool:
            """Check if position is valid for floor-level placement.
            
            BRICK WALL PRINCIPLE: We no longer require covering holes in the CURRENT layer.
            Instead, we just check that the position has room for the slice.
            The brick_score will guide placement to cover gaps from the PREVIOUS layer.
            """
            # Check if there's at least some empty space in the current layer band
            band = self.occupancy[x:x+h, y:y+w, floor_voxel:ceil]
            region_mask = mask > 0
            
            if not np.any(region_mask):
                return False
            
            # A position is valid if at least some cells are empty (room for meat)
            empty_map = (np.all(band == 0, axis=2)) & region_mask
            empty_fraction = empty_map.sum() / region_mask.sum()
            
            # Allow positions with at least 10% empty space
            return empty_fraction > 0.1
        
        def calculate_overlap_penalty(x: int, y: int) -> float:
            """Calculate how much of the position already has slices (higher = worse).
            
            This is a soft penalty - overlap is allowed but we prefer positions
            that close more holes.
            """
            region_heights = self.height_map[x:x+h, y:y+w]
            region_mask = mask > 0
            
            if not np.any(region_mask):
                return 1.0
            
            masked_heights = region_heights[region_mask]
            # Count voxels that are already filled above floor
            filled_above_floor = masked_heights > floor_voxel
            return np.mean(filled_above_floor)
        
        # Collect valid positions by zone
        # We collect both floor-level and any-level positions
        floor_true_corner_positions = []
        floor_edge_positions = []
        floor_center_positions = []
        any_positions = []  # Fallback: any valid position
        
        for x in range(self.w_voxels - h + 1):
            for y in range(self.l_voxels - w + 1):
                can_place_result, base_height = self.can_place(slice, x, y)
                if not can_place_result:
                    continue
                
                gap_score = self._calculate_gap_score(slice, x, y, base_height)
                overlap_penalty = calculate_overlap_penalty(x, y)
                brick_score = get_brick_score(x, y)
                zone = self._classify_position_zone(x, y, h, w)
                
                # SHAPE BONUS: Match slice shape to position zone
                # - Slices with pronounced corners -> cube corners
                # - Slices with straight edges -> cube walls (edges)
                # - Round/compact slices -> cube center
                shape_bonus = 0.0
                if zone in ('corner',) and slice.corner_score > 0.5:
                    # Strong bonus for cornery slices in corner zones
                    shape_bonus -= slice.corner_score * 50.0
                elif zone == 'edge' and slice.straight_edge_score > 0.5:
                    # Bonus for slices with straight edges along walls
                    shape_bonus -= slice.straight_edge_score * 40.0
                elif zone == 'center' and slice.roundness_score > 0.5:
                    # Bonus for round slices in center
                    shape_bonus -= slice.roundness_score * 30.0
                
                # BRICK WALL SCORING: Prefer positions that straddle seams from the layer below
                # - brick_score: higher is better (0.25 = perfect stagger) -> negative weight
                # - base_height: lower is better -> positive weight
                # - gap_score: lower is better -> positive weight
                # - overlap_penalty: lower is better -> positive weight (soft penalty)
                # - shape_bonus: already negative for good matches
                combined_score = (
                    - brick_score * 400.0    # BRICK WALL: Strongly reward staggering over seams
                    + base_height            # Prefer lower placements
                    + gap_score * 10.0       # Prefer positions that fill gaps
                    + overlap_penalty * 30.0 # Soft penalty for overlap (but allowed)
                    + shape_bonus            # Shape-zone matching bonus
                )
                
                position_data = (x, y, base_height, combined_score)
                
                # Always add to any_positions as fallback
                any_positions.append(position_data)
                
                # Check if this is a valid floor-level position (anti-tower)
                if not is_valid_floor_position(x, y):
                    continue
                
                is_true_corner = (x, y) in true_corners
                
                if is_true_corner:
                    # Only consider unfilled corners
                    if (x, y) in unfilled_corners:
                        floor_true_corner_positions.append(position_data)
                elif zone == 'edge':
                    floor_edge_positions.append(position_data)
                elif zone == 'center':
                    floor_center_positions.append(position_data)
                # Note: 'corner' zone (touching 2 walls but not exact corner) 
                # is treated as edge for filling purposes
                elif zone == 'corner':
                    floor_edge_positions.append(position_data)
        
        def select_best(positions):
            if not positions:
                return None
            # Sort by combined score (lower is better)
            positions.sort(key=lambda p: p[3])
            return positions[0]
        
        # STAGE 1: Fill TRUE CORNERS first (if any unfilled)
        # STRICT ENFORCEMENT: If we're in corners stage and corners are unfilled,
        # ONLY accept corner positions. If this slice doesn't fit in a corner,
        # return "no valid position" so it gets retried later.
        # NOTE: We do NOT mark corners as filled here - that happens in place_slice()
        # because this function may be called multiple times for different rotations.
        if self.layer_filling_stage == 'corners' and unfilled_corners:
            best = select_best(floor_true_corner_positions)
            if best:
                # Return the corner position - corner will be marked as filled in place_slice()
                return best[0], best[1], best[2], True
            else:
                # STRICT: No corner position found for this slice, but corners still unfilled
                # Return "no valid position" - this slice will be retried later
                # This ensures we ONLY place in corners until all 4 are filled
                return -1, -1, 0.0, False
        
        # If all corners are filled, move to edges stage
        if self.layer_filling_stage == 'corners' and not unfilled_corners:
            self.layer_filling_stage = 'edges'
        
        # STAGE 2: Fill EDGES (perimeter) with push_to_wall
        if self.layer_filling_stage == 'edges':
            best = select_best(floor_edge_positions)
            if best:
                return best[0], best[1], best[2], True
        
        # STAGE 3: Fill CENTER
        best = select_best(floor_center_positions)
        if best:
            return best[0], best[1], best[2], True
        
        # FALLBACK: If no floor-level positions found, use ANY valid position
        # This allows filling gaps even if they're not strictly at floor level
        # The overlap_penalty in combined_score will still prefer lower positions
        best = select_best(any_positions)
        if best:
            return best[0], best[1], best[2], False
        
        return -1, -1, 0.0, False

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
        The slice must be placed on top of existing slices - it cannot go
        BELOW any existing slice in the same area. This is a HARD physical
        constraint because the gripper cannot reach lower positions once
        slices are placed higher up.
        
        The slice CAN be placed anywhere from the current surface up to
        the top of the cube (250mm). There is no artificial ceiling limit.
        
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
            # MEAT ADAPTS MODE: Be VERY aggressive with overlap
            # KEY INSIGHT: Meat is not rigid like Tetris blocks!
            # The press will compress everything and close holes
            # 
            # Only hard constraint: slice can't go BELOW existing slices
            # (gripper can't reach under existing meat)
            layer_floor = self.current_layer_floor_voxel
            
            min_base_height = base_heights.min()
            
            # The slice is valid if its lowest point is at or above the layer floor
            # This is the ONLY hard constraint - we can't place under existing meat
            if min_base_height < layer_floor:
                return False, 0.0
            
            # REMOVED: near_floor_fraction check
            # Meat adapts, so we allow slices to stack freely
            # The press will compact everything at the end of each layer
        
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
            
            # CORNERS (spigoli): Fill corner voxels - MOST CRITICAL
            if push_direction in ['left_front', 'left_back', 'right_front', 'right_back']:
                mask = self._fill_corner_voxels(mask.copy(), thickness_map, push_direction, h, w)
                thickness_map = self._fill_corner_thickness(thickness_map.copy(), mask, push_direction, h, w)
            # EDGES (bordi): Fill edge voxels - SECOND MOST CRITICAL
            elif push_direction in ['left', 'right', 'front', 'back']:
                mask = self._fill_edge_voxels(mask.copy(), thickness_map, push_direction, h, w)
                thickness_map = self._fill_edge_thickness(thickness_map.copy(), mask, push_direction, h, w)
        
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
        
        # Classify the zone for this placement (corner, edge, or center)
        zone = self._classify_position_zone(x_pos, y_pos, h, w)
        
        # Mark TRUE CORNERS as filled when a slice is placed there
        # This is done here (not in find_perimeter_first_position) because
        # find_perimeter_first_position may be called multiple times for different rotations
        # before a slice is actually placed.
        corner_names = {
            (0, 0): 'front_left',
            (0, self.l_voxels - w): 'front_right',
            (self.w_voxels - h, 0): 'back_left',
            (self.w_voxels - h, self.l_voxels - w): 'back_right'
        }
        corner_name = corner_names.get((x_pos, y_pos))
        if corner_name and corner_name not in self.corners_filled_this_layer:
            self.corners_filled_this_layer.add(corner_name)
            # Check if all 4 corners are now filled
            if len(self.corners_filled_this_layer) >= 4 and self.layer_filling_stage == 'corners':
                self.layer_filling_stage = 'edges'
        
        placed = PlacedSlice(
            slice=slice,
            x=x_pos * self.resolution,
            y=y_pos * self.resolution,
            z=avg_base_height,
            rotation=0,
            push_direction=push_direction,
            zone=zone,
            layer_index=self.current_layer_index
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

    def _fill_corner_voxels(self, mask: np.ndarray, thickness_map: np.ndarray, 
                            push_direction: str, h: int, w: int) -> np.ndarray:
        """
        Fill corner voxels when a slice is pushed into a corner.
        
        When the robot pushes a slice into a corner (spigolo), the flexible meat
        deforms to fill the corner completely. This method ensures the mask
        covers the corner voxels that would otherwise be empty due to the
        irregular shape.
        
        IMPORTANT: Corners are the MOST CRITICAL part of the cube.
        We fill a larger area (4 voxels = 20mm) to ensure no 14mm holes.
        """
        fill_radius = 4  # Fill 4 voxels (20mm) into the corner - ensures no 14mm holes
        
        if push_direction == 'left_front':
            for i in range(min(fill_radius, h)):
                for j in range(min(fill_radius, w)):
                    mask[i, j] = 1.0
        elif push_direction == 'left_back':
            for i in range(min(fill_radius, h)):
                for j in range(max(0, w - fill_radius), w):
                    mask[i, j] = 1.0
        elif push_direction == 'right_front':
            for i in range(max(0, h - fill_radius), h):
                for j in range(min(fill_radius, w)):
                    mask[i, j] = 1.0
        elif push_direction == 'right_back':
            for i in range(max(0, h - fill_radius), h):
                for j in range(max(0, w - fill_radius), w):
                    mask[i, j] = 1.0
        
        return mask
    
    def _fill_edge_voxels(self, mask: np.ndarray, thickness_map: np.ndarray,
                          push_direction: str, h: int, w: int) -> np.ndarray:
        """
        Fill edge voxels when a slice is pushed against a wall (bordo).
        
        When the robot pushes a slice against a wall, the flexible meat
        deforms to fill the edge completely. This ensures no 14mm holes
        along the walls of the cube.
        
        IMPORTANT: Edges (bordi) are the SECOND MOST CRITICAL part after corners.
        """
        fill_depth = 3  # Fill 3 voxels (15mm) along the edge
        
        if push_direction == 'left':
            # Fill left edge
            for i in range(min(fill_depth, h)):
                for j in range(w):
                    if mask[i, j] == 0 and j < w // 2:  # Only fill near the edge
                        mask[i, j] = 1.0
        elif push_direction == 'right':
            # Fill right edge
            for i in range(max(0, h - fill_depth), h):
                for j in range(w):
                    if mask[i, j] == 0 and j < w // 2:
                        mask[i, j] = 1.0
        elif push_direction == 'front':
            # Fill front edge
            for i in range(h):
                for j in range(min(fill_depth, w)):
                    if mask[i, j] == 0 and i < h // 2:
                        mask[i, j] = 1.0
        elif push_direction == 'back':
            # Fill back edge
            for i in range(h):
                for j in range(max(0, w - fill_depth), w):
                    if mask[i, j] == 0 and i < h // 2:
                        mask[i, j] = 1.0
        
        return mask

    def _fill_edge_thickness(self, thickness_map: np.ndarray, mask: np.ndarray,
                              push_direction: str, h: int, w: int) -> np.ndarray:
        """
        Fill thickness values for edge voxels that were added by _fill_edge_voxels.
        """
        active_thickness = thickness_map[mask > 0]
        if active_thickness.size == 0:
            avg_thickness = 20.0
        else:
            avg_thickness = float(np.mean(active_thickness))
        
        fill_depth = 3
        
        if push_direction == 'left':
            for i in range(min(fill_depth, h)):
                for j in range(w):
                    if thickness_map[i, j] == 0 and mask[i, j] > 0:
                        thickness_map[i, j] = avg_thickness
        elif push_direction == 'right':
            for i in range(max(0, h - fill_depth), h):
                for j in range(w):
                    if thickness_map[i, j] == 0 and mask[i, j] > 0:
                        thickness_map[i, j] = avg_thickness
        elif push_direction == 'front':
            for i in range(h):
                for j in range(min(fill_depth, w)):
                    if thickness_map[i, j] == 0 and mask[i, j] > 0:
                        thickness_map[i, j] = avg_thickness
        elif push_direction == 'back':
            for i in range(h):
                for j in range(max(0, w - fill_depth), w):
                    if thickness_map[i, j] == 0 and mask[i, j] > 0:
                        thickness_map[i, j] = avg_thickness
        
        return thickness_map

    def _fill_corner_thickness(self, thickness_map: np.ndarray, mask: np.ndarray,
                               push_direction: str, h: int, w: int) -> np.ndarray:
        """
        Fill thickness values for corner voxels that were added by _fill_corner_voxels.
        
        Uses the average thickness of nearby filled voxels to set the thickness
        of newly filled corner voxels.
        """
        active_thickness = thickness_map[mask > 0]
        if active_thickness.size == 0:
            avg_thickness = 20.0  # Default 20mm
        else:
            avg_thickness = float(np.mean(active_thickness))
        
        fill_radius = 4  # Match the corner fill radius
        
        if push_direction == 'left_front':
            for i in range(min(fill_radius, h)):
                for j in range(min(fill_radius, w)):
                    if thickness_map[i, j] == 0:
                        thickness_map[i, j] = avg_thickness
        elif push_direction == 'left_back':
            for i in range(min(fill_radius, h)):
                for j in range(max(0, w - fill_radius), w):
                    if thickness_map[i, j] == 0:
                        thickness_map[i, j] = avg_thickness
        elif push_direction == 'right_front':
            for i in range(max(0, h - fill_radius), h):
                for j in range(min(fill_radius, w)):
                    if thickness_map[i, j] == 0:
                        thickness_map[i, j] = avg_thickness
        elif push_direction == 'right_back':
            for i in range(max(0, h - fill_radius), h):
                for j in range(max(0, w - fill_radius), w):
                    if thickness_map[i, j] == 0:
                        thickness_map[i, j] = avg_thickness
        
        return thickness_map

    def push_to_wall(self, x_pos: int, y_pos: int, h: int, w: int) -> Tuple[int, int, str, float]:
        """
        Calculate the optimal push direction to eliminate small gaps against walls.
        
        The robot can push the slice towards the nearest wall before releasing
        the vacuum gripper. This eliminates gaps of a few millimeters and helps
        the flexible slice conform to the wall shape.
        
        For CORNERS (spigoli): The slice first touches both walls adapting to them,
        then is pushed exactly 10mm beyond the corner contact.
        
        For WALLS: The slice is pushed against the wall with configurable compression.
        
        Returns:
            Tuple of (new_x, new_y, push_direction, compression_mm)
            push_direction is one of: 'none', 'left', 'right', 'front', 'back', 
                                      'left_front', 'left_back', 'right_front', 'right_back'
            compression_mm is the amount of compression applied
        """
        dist_to_left = x_pos
        dist_to_right = self.w_voxels - (x_pos + h)
        dist_to_front = y_pos
        dist_to_back = self.l_voxels - (y_pos + w)
        
        push_threshold_voxels = 6  # 30mm (6 voxels * 5mm resolution)
        wall_compression_mm = 25.0  # Compression for single wall contact
        corner_compression_mm = 10.0  # Exactly 10mm beyond corner contact (user spec)
        
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
            actual_compression = corner_compression_mm
        elif push_left and push_back:
            new_x = 0
            new_y = self.l_voxels - w
            push_direction = 'left_back'
            actual_compression = corner_compression_mm
        elif push_right and push_front:
            new_x = self.w_voxels - h
            new_y = 0
            push_direction = 'right_front'
            actual_compression = corner_compression_mm
        elif push_right and push_back:
            new_x = self.w_voxels - h
            new_y = self.l_voxels - w
            push_direction = 'right_back'
            actual_compression = corner_compression_mm
        elif push_left:
            new_x = 0
            push_direction = 'left'
            actual_compression = wall_compression_mm
        elif push_right:
            new_x = self.w_voxels - h
            push_direction = 'right'
            actual_compression = wall_compression_mm
        elif push_front:
            new_y = 0
            push_direction = 'front'
            actual_compression = wall_compression_mm
        elif push_back:
            new_y = self.l_voxels - w
            push_direction = 'back'
            actual_compression = wall_compression_mm
        
        return new_x, new_y, push_direction, actual_compression

    def should_press_layer(self) -> bool:
        """
        Check if the current layer should be pressed.
        
        BRICK WALL PRINCIPLE: Holes in the current layer are covered by the NEXT layer.
        Don't block pressing based on holes - that creates "culi di carne" (meat bumps).
        
        Returns True only when:
        1. Layer coverage >= threshold (90%)
        2. We've built at least one layer's worth of thickness above the floor
        3. There's still room for more layers
        """
        coverage = self.get_layer_coverage()
        if coverage < self.LAYER_COVERAGE_THRESHOLD:
            return False
        
        current_max = np.max(self.height_map)
        band_thickness = current_max - self.current_layer_floor_voxel
        
        if band_thickness < self.layer_thickness_voxels - 1:
            return False
        
        if self.current_layer_floor_voxel + self.layer_thickness_voxels >= self.h_voxels:
            return False
        
        return True
    
    def press_layer(self) -> Dict[str, float]:
        """
        Press/compact the current layer to create a flat, uniform surface.
        
        Called after a layer is complete (95% coverage AND sufficient thickness) to:
        1. Save the current layer's coverage map for BRICK WALL staggering
        2. Flatten the top surface to the new floor level
        3. Advance to the next layer by a fixed band height
        
        BRICK WALL PRINCIPLE: Before pressing, we save the 2D coverage map of this layer.
        The next layer will use this to stagger slices (like bricks covering seams below).
        
        IMPORTANT: This should only be called when should_press_layer() returns True.
        The layer floor advances by layer_thickness_voxels (~25mm), not by compression.
        """
        if not np.any(self.height_map > 0):
            return {"pressed": False, "new_layer_height": 0.0}
        
        # BRICK WALL: Save the current layer's coverage map BEFORE pressing
        # This will be used by the next layer to stagger placements
        floor = self.current_layer_floor_voxel
        ceil = min(floor + self.layer_thickness_voxels, self.h_voxels)
        band = self.occupancy[:, :, floor:ceil]
        self.last_layer_coverage = np.any(band > 0, axis=2)  # 2D bool: True where meat exists
        
        new_floor_voxel = min(
            self.current_layer_floor_voxel + self.layer_thickness_voxels,
            self.h_voxels
        )
        
        self.height_map = np.maximum(self.height_map, float(new_floor_voxel))
        
        self.current_layer_floor_voxel = new_floor_voxel
        self.current_layer_ceiling_voxel = new_floor_voxel + self.layer_thickness_voxels
        self.current_layer_height = new_floor_voxel * self.resolution
        self.current_layer_target = float(new_floor_voxel)
        self.layers_completed += 1
        self.current_layer_index += 1
        self.layer_complete = False
        
        # Reset per-layer tracking for corners->perimeter->center rule
        self.corners_filled_this_layer = set()
        self.layer_filling_stage = 'corners'
        
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
