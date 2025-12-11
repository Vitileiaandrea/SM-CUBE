"""
Fanuc Robot Interface

This module provides an interface to control a Fanuc anthropomorphic/SCARA robot
with a 5-finger vacuum gripper for meat slice manipulation.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import json
import time


class RobotState(Enum):
    """Robot operational states."""
    IDLE = "idle"
    MOVING = "moving"
    PICKING = "picking"
    PLACING = "placing"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class GripperState(Enum):
    """Gripper states."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


@dataclass
class RobotPosition:
    """6-DOF robot position."""
    x: float  # mm
    y: float  # mm
    z: float  # mm
    rx: float = 0.0  # degrees (rotation around X)
    ry: float = 0.0  # degrees (rotation around Y)
    rz: float = 0.0  # degrees (rotation around Z)
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.rx, self.ry, self.rz])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "RobotPosition":
        return cls(
            x=float(arr[0]),
            y=float(arr[1]),
            z=float(arr[2]),
            rx=float(arr[3]) if len(arr) > 3 else 0.0,
            ry=float(arr[4]) if len(arr) > 4 else 0.0,
            rz=float(arr[5]) if len(arr) > 5 else 0.0
        )
    
    def distance_to(self, other: "RobotPosition") -> float:
        """Calculate Euclidean distance to another position."""
        return float(np.linalg.norm(
            np.array([self.x, self.y, self.z]) - 
            np.array([other.x, other.y, other.z])
        ))


class PlacementZone(Enum):
    """Target placement zones in the cube."""
    CENTER = "center"
    CORNER_TL = "corner_top_left"
    CORNER_TR = "corner_top_right"
    CORNER_BL = "corner_bottom_left"
    CORNER_BR = "corner_bottom_right"
    EDGE_TOP = "edge_top"
    EDGE_BOTTOM = "edge_bottom"
    EDGE_LEFT = "edge_left"
    EDGE_RIGHT = "edge_right"


@dataclass
class VacuumCupGrid:
    """
    Represents the vacuum cup grid on the robot gripper.
    
    Based on the actual gripper design with ~20 vacuum cups arranged in a 5x4 grid.
    The cups are fixed on the robot wrist, so which cups are activated determines
    where the slice will be positioned when released.
    
    Grid layout (5 rows x 4 columns):
        [0,0] [0,1] [0,2] [0,3]   <- Top row (for bottom edge placement)
        [1,0] [1,1] [1,2] [1,3]
        [2,0] [2,1] [2,2] [2,3]   <- Center rows
        [3,0] [3,1] [3,2] [3,3]
        [4,0] [4,1] [4,2] [4,3]   <- Bottom row (for top edge placement)
    
    When the robot picks with corner cups, the slice hangs from that corner,
    so when placed, it goes to the opposite corner of the cube.
    """
    rows: int = 5
    cols: int = 4
    cup_spacing: float = 40.0  # mm between cups
    cup_diameter: float = 25.0  # mm
    
    def get_pattern_for_zone(self, zone: PlacementZone) -> np.ndarray:
        """
        Get the vacuum cup activation pattern for a target placement zone.
        
        The key insight: cups activated determine where slice hangs from gripper,
        which determines where it lands in the cube.
        
        Returns:
            5x4 numpy array with 1s for active cups, 0s for inactive
        """
        pattern = np.zeros((self.rows, self.cols), dtype=np.int32)
        
        if zone == PlacementZone.CENTER:
            pattern[1:4, 1:3] = 1
        elif zone == PlacementZone.CORNER_TL:
            pattern[0:2, 0:2] = 1
        elif zone == PlacementZone.CORNER_TR:
            pattern[0:2, 2:4] = 1
        elif zone == PlacementZone.CORNER_BL:
            pattern[3:5, 0:2] = 1
        elif zone == PlacementZone.CORNER_BR:
            pattern[3:5, 2:4] = 1
        elif zone == PlacementZone.EDGE_TOP:
            pattern[0:2, 1:3] = 1
        elif zone == PlacementZone.EDGE_BOTTOM:
            pattern[3:5, 1:3] = 1
        elif zone == PlacementZone.EDGE_LEFT:
            pattern[1:4, 0:2] = 1
        elif zone == PlacementZone.EDGE_RIGHT:
            pattern[1:4, 2:4] = 1
        
        return pattern
    
    def get_zone_for_position(
        self,
        target_x: float,
        target_y: float,
        cube_width: float = 210.0,
        cube_length: float = 210.0
    ) -> PlacementZone:
        """
        Determine which placement zone based on target position in cube.
        
        Args:
            target_x, target_y: Target position in cube (0-210mm)
            cube_width, cube_length: Cube dimensions
            
        Returns:
            PlacementZone enum value
        """
        edge_threshold = 50.0  # mm from edge to be considered edge/corner
        
        near_left = target_x < edge_threshold
        near_right = target_x > (cube_width - edge_threshold)
        near_top = target_y < edge_threshold
        near_bottom = target_y > (cube_length - edge_threshold)
        
        if near_left and near_top:
            return PlacementZone.CORNER_TL
        elif near_right and near_top:
            return PlacementZone.CORNER_TR
        elif near_left and near_bottom:
            return PlacementZone.CORNER_BL
        elif near_right and near_bottom:
            return PlacementZone.CORNER_BR
        elif near_top:
            return PlacementZone.EDGE_TOP
        elif near_bottom:
            return PlacementZone.EDGE_BOTTOM
        elif near_left:
            return PlacementZone.EDGE_LEFT
        elif near_right:
            return PlacementZone.EDGE_RIGHT
        else:
            return PlacementZone.CENTER
    
    def pattern_to_list(self, pattern: np.ndarray) -> List[int]:
        """Convert 2D pattern to flat list for transmission."""
        return pattern.flatten().tolist()
    
    def get_active_cup_count(self, pattern: np.ndarray) -> int:
        """Count number of active cups in pattern."""
        return int(np.sum(pattern))
    
    def get_grip_center_offset(self, pattern: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the offset from gripper center to grip center.
        
        This is important because the slice will hang from the centroid
        of the active cups, not the gripper center.
        
        Returns:
            (x_offset, y_offset) in mm from gripper center
        """
        active_positions = np.argwhere(pattern > 0)
        if len(active_positions) == 0:
            return (0.0, 0.0)
        
        centroid = np.mean(active_positions, axis=0)
        
        grid_center = np.array([(self.rows - 1) / 2, (self.cols - 1) / 2])
        
        offset = (centroid - grid_center) * self.cup_spacing
        
        return (float(offset[1]), float(offset[0]))


@dataclass
class GripperCommand:
    """
    Command for the vacuum gripper with selective cup activation.
    
    The gripper has a 5x4 grid of vacuum cups. Which cups are activated
    determines where the slice will be positioned when released:
    - Corner cups -> slice placed at cube corner
    - Center cups -> slice placed at cube center
    - Edge cups -> slice placed at cube edge
    
    The robot can also rotate its wrist to orient the slice.
    """
    cup_pattern: np.ndarray = field(default_factory=lambda: np.ones((5, 4), dtype=np.int32))
    vacuum_level: float = 0.8  # 0-1
    placement_zone: PlacementZone = PlacementZone.CENTER
    wrist_rotation: float = 0.0  # degrees
    
    def __post_init__(self):
        if isinstance(self.cup_pattern, list):
            self.cup_pattern = np.array(self.cup_pattern).reshape(5, 4)
    
    @classmethod
    def for_placement(
        cls,
        target_x: float,
        target_y: float,
        rotation: float = 0.0,
        vacuum_level: float = 0.85
    ) -> "GripperCommand":
        """
        Create a gripper command for placing a slice at a specific position.
        
        Args:
            target_x, target_y: Target position in cube (0-210mm)
            rotation: Wrist rotation in degrees
            vacuum_level: Vacuum strength (0-1)
            
        Returns:
            GripperCommand configured for the target position
        """
        grid = VacuumCupGrid()
        zone = grid.get_zone_for_position(target_x, target_y)
        pattern = grid.get_pattern_for_zone(zone)
        
        return cls(
            cup_pattern=pattern,
            vacuum_level=vacuum_level,
            placement_zone=zone,
            wrist_rotation=rotation
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cup_pattern": self.cup_pattern.tolist(),
            "cup_pattern_flat": self.cup_pattern.flatten().tolist(),
            "vacuum_level": self.vacuum_level,
            "placement_zone": self.placement_zone.value,
            "wrist_rotation": self.wrist_rotation,
            "active_cups": int(np.sum(self.cup_pattern))
        }
    
    def get_grip_offset(self) -> Tuple[float, float]:
        """Get the offset from gripper center to grip centroid."""
        grid = VacuumCupGrid()
        return grid.get_grip_center_offset(self.cup_pattern)


@dataclass
class MotionCommand:
    """Robot motion command."""
    target_position: RobotPosition
    speed: float = 100.0  # mm/s
    acceleration: float = 500.0  # mm/s^2
    motion_type: str = "linear"  # "linear" or "joint"
    blend_radius: float = 0.0  # mm for smooth transitions
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": {
                "x": self.target_position.x,
                "y": self.target_position.y,
                "z": self.target_position.z,
                "rx": self.target_position.rx,
                "ry": self.target_position.ry,
                "rz": self.target_position.rz
            },
            "speed": self.speed,
            "acceleration": self.acceleration,
            "motion_type": self.motion_type,
            "blend_radius": self.blend_radius
        }


@dataclass
class PickPlaceSequence:
    """Complete pick and place operation sequence."""
    pick_approach: RobotPosition
    pick_position: RobotPosition
    pick_retreat: RobotPosition
    place_approach: RobotPosition
    place_position: RobotPosition
    place_retreat: RobotPosition
    gripper_command: GripperCommand
    slice_rotation: float = 0.0


class FanucRobotInterface:
    """
    Interface for controlling a Fanuc robot with vacuum gripper.
    
    This class handles:
    - Motion planning and execution
    - Gripper control
    - Safety checks
    - Communication with Schneider PLC
    """
    
    def __init__(
        self,
        robot_ip: str = "192.168.1.100",
        plc_ip: str = "192.168.1.200",
        home_position: Optional[RobotPosition] = None,
        conveyor_pickup_y: float = 300.0,
        cube_center_x: float = 500.0,
        cube_center_y: float = 0.0,
        safe_z: float = 300.0
    ):
        self.robot_ip = robot_ip
        self.plc_ip = plc_ip
        self.safe_z = safe_z
        self.conveyor_pickup_y = conveyor_pickup_y
        self.cube_center_x = cube_center_x
        self.cube_center_y = cube_center_y
        
        self.home_position = home_position or RobotPosition(
            x=0.0, y=0.0, z=safe_z, rx=0.0, ry=180.0, rz=0.0
        )
        
        self.current_position = self.home_position
        self.state = RobotState.IDLE
        self.gripper_state = GripperState.OPEN
        
        self.workspace_limits = {
            "x_min": -400.0, "x_max": 800.0,
            "y_min": -400.0, "y_max": 600.0,
            "z_min": 0.0, "z_max": 500.0
        }
        
        self.max_speed = 500.0  # mm/s
        self.max_acceleration = 2000.0  # mm/s^2
        
        self._command_queue: List[Dict[str, Any]] = []
        self._is_connected = False
    
    def connect(self) -> bool:
        """Establish connection to robot and PLC."""
        self._is_connected = True
        self.state = RobotState.IDLE
        return True
    
    def disconnect(self):
        """Disconnect from robot and PLC."""
        self._is_connected = False
        self.state = RobotState.IDLE
    
    def is_position_safe(self, position: RobotPosition) -> Tuple[bool, str]:
        """Check if a position is within safe workspace limits."""
        if position.x < self.workspace_limits["x_min"]:
            return False, f"X position {position.x} below minimum"
        if position.x > self.workspace_limits["x_max"]:
            return False, f"X position {position.x} above maximum"
        if position.y < self.workspace_limits["y_min"]:
            return False, f"Y position {position.y} below minimum"
        if position.y > self.workspace_limits["y_max"]:
            return False, f"Y position {position.y} above maximum"
        if position.z < self.workspace_limits["z_min"]:
            return False, f"Z position {position.z} below minimum"
        if position.z > self.workspace_limits["z_max"]:
            return False, f"Z position {position.z} above maximum"
        return True, "Position is safe"
    
    def generate_pick_place_sequence(
        self,
        pick_x: float,
        pick_y: float,
        pick_z: float,
        place_x: float,
        place_y: float,
        place_z: float,
        rotation: float = 0.0,
        gripper_pattern: List[int] = None
    ) -> PickPlaceSequence:
        """
        Generate a complete pick and place motion sequence.
        
        The gripper uses a 5x4 grid of vacuum cups. Which cups are activated
        determines where the slice will be positioned in the cube:
        - Corner cups -> slice placed at cube corner
        - Center cups -> slice placed at cube center
        - Edge cups -> slice placed at cube edge
        
        Args:
            pick_x, pick_y, pick_z: Pick position in mm
            place_x, place_y, place_z: Place position in mm (0-210mm in cube)
            rotation: Rotation angle for the slice in degrees (wrist rotation)
            gripper_pattern: Legacy parameter, ignored - use place_x/place_y instead
            
        Returns:
            PickPlaceSequence with all waypoints and gripper command
        """
        approach_height = 50.0  # mm above target
        retreat_height = 80.0  # mm above after pick/place
        
        gripper_cmd = GripperCommand.for_placement(
            target_x=place_x,
            target_y=place_y,
            rotation=rotation,
            vacuum_level=0.85
        )
        
        grip_offset_x, grip_offset_y = gripper_cmd.get_grip_offset()
        
        pick_approach = RobotPosition(
            x=pick_x, y=pick_y, z=pick_z + approach_height,
            rx=0.0, ry=180.0, rz=rotation
        )
        
        pick_position = RobotPosition(
            x=pick_x, y=pick_y, z=pick_z + 5.0,
            rx=0.0, ry=180.0, rz=rotation
        )
        
        pick_retreat = RobotPosition(
            x=pick_x, y=pick_y, z=pick_z + retreat_height,
            rx=0.0, ry=180.0, rz=rotation
        )
        
        place_world_x = self.cube_center_x + place_x - 105.0 + grip_offset_x
        place_world_y = self.cube_center_y + place_y - 105.0 + grip_offset_y
        
        place_approach = RobotPosition(
            x=place_world_x,
            y=place_world_y,
            z=place_z + approach_height + 50.0,
            rx=0.0, ry=180.0, rz=rotation
        )
        
        place_position = RobotPosition(
            x=place_world_x,
            y=place_world_y,
            z=place_z + 10.0,
            rx=0.0, ry=180.0, rz=rotation
        )
        
        place_retreat = RobotPosition(
            x=place_world_x,
            y=place_world_y,
            z=place_z + retreat_height,
            rx=0.0, ry=180.0, rz=rotation
        )
        
        return PickPlaceSequence(
            pick_approach=pick_approach,
            pick_position=pick_position,
            pick_retreat=pick_retreat,
            place_approach=place_approach,
            place_position=place_position,
            place_retreat=place_retreat,
            gripper_command=gripper_cmd,
            slice_rotation=rotation
        )
    
    def execute_pick_place(
        self,
        sequence: PickPlaceSequence
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a pick and place sequence.
        
        Returns:
            Tuple of (success, execution_info)
        """
        execution_log = []
        
        for position in [sequence.pick_approach, sequence.pick_position,
                        sequence.pick_retreat, sequence.place_approach,
                        sequence.place_position, sequence.place_retreat]:
            is_safe, msg = self.is_position_safe(position)
            if not is_safe:
                return False, {"error": msg, "log": execution_log}
        
        commands = []
        
        commands.append({
            "type": "move",
            "motion": MotionCommand(
                target_position=sequence.pick_approach,
                speed=300.0,
                motion_type="joint"
            ).to_dict(),
            "description": "Move to pick approach"
        })
        
        commands.append({
            "type": "move",
            "motion": MotionCommand(
                target_position=sequence.pick_position,
                speed=100.0,
                motion_type="linear"
            ).to_dict(),
            "description": "Descend to pick position"
        })
        
        commands.append({
            "type": "gripper",
            "command": sequence.gripper_command.to_dict(),
            "action": "activate",
            "description": "Activate vacuum gripper"
        })
        
        commands.append({
            "type": "wait",
            "duration": 0.2,
            "description": "Wait for vacuum"
        })
        
        commands.append({
            "type": "move",
            "motion": MotionCommand(
                target_position=sequence.pick_retreat,
                speed=150.0,
                motion_type="linear"
            ).to_dict(),
            "description": "Retreat from pick"
        })
        
        commands.append({
            "type": "move",
            "motion": MotionCommand(
                target_position=sequence.place_approach,
                speed=400.0,
                motion_type="joint"
            ).to_dict(),
            "description": "Move to place approach"
        })
        
        commands.append({
            "type": "move",
            "motion": MotionCommand(
                target_position=sequence.place_position,
                speed=80.0,
                motion_type="linear"
            ).to_dict(),
            "description": "Descend to place position"
        })
        
        commands.append({
            "type": "gripper",
            "command": {"fingers": [0, 0, 0, 0, 0], "vacuum": 0.0},
            "action": "release",
            "description": "Release vacuum"
        })
        
        commands.append({
            "type": "wait",
            "duration": 0.1,
            "description": "Wait for release"
        })
        
        commands.append({
            "type": "move",
            "motion": MotionCommand(
                target_position=sequence.place_retreat,
                speed=200.0,
                motion_type="linear"
            ).to_dict(),
            "description": "Retreat from place"
        })
        
        self._command_queue = commands
        
        return True, {
            "commands": commands,
            "estimated_time": self._estimate_cycle_time(sequence),
            "log": execution_log
        }
    
    def _estimate_cycle_time(self, sequence: PickPlaceSequence) -> float:
        """Estimate the cycle time for a pick-place sequence."""
        positions = [
            self.current_position,
            sequence.pick_approach,
            sequence.pick_position,
            sequence.pick_retreat,
            sequence.place_approach,
            sequence.place_position,
            sequence.place_retreat
        ]
        
        total_distance = 0.0
        for i in range(len(positions) - 1):
            total_distance += positions[i].distance_to(positions[i + 1])
        
        avg_speed = 200.0  # mm/s average
        motion_time = total_distance / avg_speed
        
        gripper_time = 0.5  # seconds for gripper operations
        
        return motion_time + gripper_time
    
    def get_command_queue(self) -> List[Dict[str, Any]]:
        """Get the current command queue."""
        return self._command_queue.copy()
    
    def clear_command_queue(self):
        """Clear the command queue."""
        self._command_queue.clear()
    
    def go_home(self) -> Dict[str, Any]:
        """Generate command to return to home position."""
        return {
            "type": "move",
            "motion": MotionCommand(
                target_position=self.home_position,
                speed=200.0,
                motion_type="joint"
            ).to_dict(),
            "description": "Return to home position"
        }
    
    def emergency_stop(self):
        """Trigger emergency stop."""
        self.state = RobotState.EMERGENCY_STOP
        self._command_queue.clear()
    
    def reset_from_emergency(self) -> bool:
        """Reset from emergency stop state."""
        if self.state == RobotState.EMERGENCY_STOP:
            self.state = RobotState.IDLE
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get current robot status."""
        return {
            "state": self.state.value,
            "position": {
                "x": self.current_position.x,
                "y": self.current_position.y,
                "z": self.current_position.z,
                "rx": self.current_position.rx,
                "ry": self.current_position.ry,
                "rz": self.current_position.rz
            },
            "gripper_state": self.gripper_state.value,
            "is_connected": self._is_connected,
            "queue_length": len(self._command_queue)
        }


class PLCInterface:
    """
    Interface for Schneider PLC communication.
    
    Handles:
    - Conveyor control
    - Safety interlocks
    - Sensor readings
    - System coordination
    """
    
    def __init__(self, plc_ip: str = "192.168.1.200"):
        self.plc_ip = plc_ip
        self._is_connected = False
        
        self.conveyor_speed = 100.0  # mm/s
        self.conveyor_running = False
        
        self.safety_ok = True
        self.emergency_stop_active = False
        
        self.cube_present = False
        self.cube_full = False
    
    def connect(self) -> bool:
        """Connect to PLC."""
        self._is_connected = True
        return True
    
    def disconnect(self):
        """Disconnect from PLC."""
        self._is_connected = False
    
    def start_conveyor(self, speed: float = 100.0) -> bool:
        """Start the conveyor belt."""
        if not self.safety_ok:
            return False
        self.conveyor_speed = speed
        self.conveyor_running = True
        return True
    
    def stop_conveyor(self) -> bool:
        """Stop the conveyor belt."""
        self.conveyor_running = False
        return True
    
    def set_conveyor_speed(self, speed: float) -> bool:
        """Set conveyor speed in mm/s."""
        if speed < 0 or speed > 500:
            return False
        self.conveyor_speed = speed
        return True
    
    def check_safety(self) -> Dict[str, bool]:
        """Check all safety interlocks."""
        return {
            "safety_ok": self.safety_ok,
            "emergency_stop": self.emergency_stop_active,
            "guards_closed": True,
            "pressure_ok": True
        }
    
    def trigger_emergency_stop(self):
        """Trigger emergency stop."""
        self.emergency_stop_active = True
        self.safety_ok = False
        self.conveyor_running = False
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop."""
        self.emergency_stop_active = False
        self.safety_ok = True
        return True
    
    def signal_cube_ready(self) -> bool:
        """Signal that a new cube is ready for filling."""
        self.cube_present = True
        self.cube_full = False
        return True
    
    def signal_cube_complete(self) -> bool:
        """Signal that the current cube is complete."""
        self.cube_full = True
        return True
    
    def request_new_cube(self) -> bool:
        """Request a new empty cube."""
        self.cube_present = False
        self.cube_full = False
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get PLC status."""
        return {
            "connected": self._is_connected,
            "conveyor_running": self.conveyor_running,
            "conveyor_speed": self.conveyor_speed,
            "safety_ok": self.safety_ok,
            "emergency_stop": self.emergency_stop_active,
            "cube_present": self.cube_present,
            "cube_full": self.cube_full
        }


class RobotCommandGenerator:
    """
    High-level command generator that converts agent decisions
    to robot commands.
    """
    
    def __init__(
        self,
        robot: Optional[FanucRobotInterface] = None,
        plc: Optional[PLCInterface] = None
    ):
        self.robot = robot or FanucRobotInterface()
        self.plc = plc or PLCInterface()
    
    def generate_placement_commands(
        self,
        slice_position: Tuple[float, float, float],
        target_position: Dict[str, float],
        gripper_pattern: List[int]
    ) -> Dict[str, Any]:
        """
        Generate robot commands for a placement operation.
        
        Args:
            slice_position: Current (x, y, z) of slice on conveyor
            target_position: Target position in cube {x, y, z, rotation}
            gripper_pattern: Which gripper fingers to use
            
        Returns:
            Dictionary with commands and metadata
        """
        sequence = self.robot.generate_pick_place_sequence(
            pick_x=slice_position[0],
            pick_y=slice_position[1],
            pick_z=slice_position[2],
            place_x=target_position["x"],
            place_y=target_position["y"],
            place_z=target_position["z"],
            rotation=target_position.get("rotation", 0.0),
            gripper_pattern=gripper_pattern
        )
        
        success, execution_info = self.robot.execute_pick_place(sequence)
        
        return {
            "success": success,
            "sequence": {
                "pick": {
                    "approach": sequence.pick_approach.__dict__,
                    "position": sequence.pick_position.__dict__,
                    "retreat": sequence.pick_retreat.__dict__
                },
                "place": {
                    "approach": sequence.place_approach.__dict__,
                    "position": sequence.place_position.__dict__,
                    "retreat": sequence.place_retreat.__dict__
                },
                "gripper": sequence.gripper_command.to_dict()
            },
            "commands": execution_info.get("commands", []),
            "estimated_time": execution_info.get("estimated_time", 0.0)
        }
    
    def generate_press_command(
        self,
        press_z: float,
        force: float = 50.0
    ) -> Dict[str, Any]:
        """
        Generate command to press/compact the current layer.
        
        Args:
            press_z: Z height to press to
            force: Pressing force in N
            
        Returns:
            Press command dictionary
        """
        return {
            "type": "press",
            "position": {
                "x": self.robot.cube_center_x,
                "y": self.robot.cube_center_y,
                "z": press_z
            },
            "force": force,
            "duration": 0.5
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get combined system status."""
        return {
            "robot": self.robot.get_status(),
            "plc": self.plc.get_status(),
            "ready": (
                self.robot.state == RobotState.IDLE and
                self.plc.safety_ok and
                not self.plc.emergency_stop_active
            )
        }
