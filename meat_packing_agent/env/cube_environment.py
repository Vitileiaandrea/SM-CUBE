"""
3D Bin-Packing Environment for Meat Slice Placement

This environment simulates a cube container (210x210x250mm) that needs to be filled
with meat slices of varying sizes and thicknesses. The agent learns to place slices
optimally to minimize empty spaces and create flat layers.
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field


@dataclass
class MeatSlice:
    """Represents a meat slice with its 3D properties."""
    
    width: float  # mm (50-200)
    length: float  # mm (50-200)
    thickness: float  # mm (5-40)
    shape_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    slice_id: int = 0
    
    def __post_init__(self):
        if self.shape_mask.size == 0:
            self.shape_mask = self._generate_irregular_shape()
    
    def _generate_irregular_shape(self) -> np.ndarray:
        """Generate an irregular meat slice shape (not perfectly rectangular)."""
        resolution = 5  # mm per voxel
        w_voxels = max(1, int(self.width / resolution))
        l_voxels = max(1, int(self.length / resolution))
        
        mask = np.ones((w_voxels, l_voxels), dtype=np.float32)
        
        center_x, center_y = w_voxels / 2, l_voxels / 2
        for i in range(w_voxels):
            for j in range(l_voxels):
                dist_x = abs(i - center_x) / (w_voxels / 2)
                dist_y = abs(j - center_y) / (l_voxels / 2)
                edge_dist = max(dist_x, dist_y)
                
                if edge_dist > 0.7:
                    noise = np.random.uniform(0, 0.5)
                    if edge_dist + noise > 1.0:
                        mask[i, j] = 0
        
        return mask
    
    def get_volume(self) -> float:
        """Calculate the volume of the meat slice in mm^3."""
        resolution = 5
        filled_voxels = np.sum(self.shape_mask)
        area = filled_voxels * (resolution ** 2)
        return area * self.thickness
    
    def rotate(self, angle: int) -> "MeatSlice":
        """Rotate the slice by 90-degree increments."""
        rotations = (angle // 90) % 4
        new_mask = np.rot90(self.shape_mask, rotations)
        new_width = self.length if rotations % 2 else self.width
        new_length = self.width if rotations % 2 else self.length
        return MeatSlice(
            width=new_width,
            length=new_length,
            thickness=self.thickness,
            shape_mask=new_mask,
            slice_id=self.slice_id
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
    """Represents the current state of the cube being filled."""
    
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
    
    def reset(self):
        """Reset the cube to empty state."""
        self.height_map.fill(0)
        self.occupancy.fill(0)
        self.placed_slices.clear()
        self.total_volume_filled = 0.0
    
    def can_place(
        self,
        slice: MeatSlice,
        x_pos: int,
        y_pos: int
    ) -> Tuple[bool, float]:
        """Check if a slice can be placed at the given position."""
        mask = slice.shape_mask
        h, w = mask.shape
        
        if x_pos < 0 or y_pos < 0:
            return False, 0.0
        if x_pos + h > self.w_voxels or y_pos + w > self.l_voxels:
            return False, 0.0
        
        region_heights = self.height_map[x_pos:x_pos+h, y_pos:y_pos+w]
        masked_heights = region_heights * mask
        
        base_height = np.max(masked_heights)
        thickness_voxels = int(slice.thickness / self.resolution)
        
        if base_height + thickness_voxels > self.h_voxels:
            return False, 0.0
        
        return True, base_height * self.resolution
    
    def place_slice(
        self,
        slice: MeatSlice,
        x_pos: int,
        y_pos: int
    ) -> Tuple[bool, Dict[str, float]]:
        """Place a slice at the given position and return placement metrics."""
        can_place, base_height = self.can_place(slice, x_pos, y_pos)
        
        if not can_place:
            return False, {"reward": -1.0, "gap_penalty": 0.0, "flatness": 0.0}
        
        mask = slice.shape_mask
        h, w = mask.shape
        thickness_voxels = int(slice.thickness / self.resolution)
        base_voxel = int(base_height / self.resolution)
        
        gap_volume = 0.0
        region_heights = self.height_map[x_pos:x_pos+h, y_pos:y_pos+w].copy()
        
        for i in range(h):
            for j in range(w):
                if mask[i, j] > 0:
                    current_height = int(region_heights[i, j])
                    gap_volume += (base_voxel - current_height) * (self.resolution ** 3)
                    
                    new_height = base_voxel + thickness_voxels
                    self.height_map[x_pos+i, y_pos+j] = new_height
                    
                    for k in range(base_voxel, new_height):
                        if k < self.h_voxels:
                            self.occupancy[x_pos+i, y_pos+j, k] = 1.0
        
        placed = PlacedSlice(
            slice=slice,
            x=x_pos * self.resolution,
            y=y_pos * self.resolution,
            z=base_height,
            rotation=0
        )
        self.placed_slices.append(placed)
        self.total_volume_filled += slice.get_volume()
        
        flatness = self._calculate_flatness()
        gap_penalty = gap_volume / (self.resolution ** 3 * 100)
        
        utilization = self.total_volume_filled / self.max_volume
        reward = utilization * 10 - gap_penalty * 0.5 + flatness * 2
        
        return True, {
            "reward": reward,
            "gap_penalty": gap_penalty,
            "flatness": flatness,
            "utilization": utilization,
            "base_height": base_height
        }
    
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
        """Generate a random meat slice with realistic dimensions."""
        width = np.random.uniform(50, 200)
        length = np.random.uniform(50, 200)
        thickness = np.random.uniform(5, 40)
        
        return MeatSlice(
            width=width,
            length=length,
            thickness=thickness,
            slice_id=self.slices_placed
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
