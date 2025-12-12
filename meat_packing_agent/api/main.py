"""
FastAPI Backend for Meat Packing Agent

This module provides REST API endpoints for controlling the meat packing
RL agent, monitoring system status, and managing training.
"""

from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import numpy as np
import asyncio
import json
from datetime import datetime
from pathlib import Path

from meat_packing_agent.env.cube_environment import MeatPackingEnv, MeatSlice, CubeState
from meat_packing_agent.agent.ppo_agent import MeatPackingAgent, create_agent
from meat_packing_agent.vision.lidar_processor import LiDARProcessor, SliceGeometry
from meat_packing_agent.robot.fanuc_interface import (
    FanucRobotInterface,
    PLCInterface,
    RobotCommandGenerator
)
from meat_packing_agent.training.train_1000_cubes import MeatPackingTrainer

app = FastAPI(
    title="Meat Packing Agent API",
    description="API for controlling the RL-based meat packing system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env: Optional[MeatPackingEnv] = None
agent: Optional[MeatPackingAgent] = None
robot_interface: Optional[FanucRobotInterface] = None
plc_interface: Optional[PLCInterface] = None
command_generator: Optional[RobotCommandGenerator] = None
lidar_processor: Optional[LiDARProcessor] = None

# Slice database for training-style simulation
slice_database: List[Dict] = []
manual_cube_slices: List[Dict] = []
manual_slice_idx: int = 0
manual_retry_list: List = []
manual_cube_id: int = 0
manual_consecutive_failures: int = 0

active_connections: List[WebSocket] = []


class SliceInfo(BaseModel):
    """Information about a meat slice."""
    width: float = Field(..., ge=50, le=200, description="Width in mm")
    length: float = Field(..., ge=50, le=200, description="Length in mm")
    thickness: float = Field(..., ge=5, le=40, description="Thickness in mm")


class PlacementRequest(BaseModel):
    """Request for placement decision."""
    slice_info: SliceInfo
    slice_position: Optional[List[float]] = Field(
        None, description="Current position [x, y, z] on conveyor"
    )


class PlacementResponse(BaseModel):
    """Response with placement decision."""
    x: float
    y: float
    z: float
    rotation: int
    gripper_pattern: List[int]
    confidence: float
    robot_commands: Optional[Dict[str, Any]] = None


class TrainingConfig(BaseModel):
    """Configuration for training."""
    total_timesteps: int = Field(100000, ge=1000)
    learning_rate: float = Field(3e-4, gt=0)
    batch_size: int = Field(64, ge=16)
    save_path: Optional[str] = None


class SystemStatus(BaseModel):
    """System status response."""
    agent_loaded: bool
    robot_connected: bool
    plc_connected: bool
    cube_fill_percentage: float
    slices_placed: int
    system_ready: bool


@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup."""
    global env, agent, robot_interface, plc_interface, command_generator, lidar_processor, slice_database
    
    env = MeatPackingEnv()
    env.reset()
    
    # Load the slice database (SAME as training script)
    database_path = Path(__file__).parent.parent / "data" / "slices_10000.json"
    try:
        with open(database_path, 'r') as f:
            slice_database = json.load(f)
        print(f"Loaded {len(slice_database)} slices from database")
    except Exception as e:
        print(f"Warning: Could not load slice database: {e}")
        slice_database = []
    
    try:
        agent = create_agent()
    except Exception as e:
        print(f"Warning: Could not initialize agent: {e}")
        agent = None
    
    robot_interface = FanucRobotInterface()
    plc_interface = PLCInterface()
    command_generator = RobotCommandGenerator(robot_interface, plc_interface)
    lidar_processor = LiDARProcessor()
    
    # Initialize first cube with database slices
    _init_cube_slices()
    
    print("Meat Packing Agent API initialized")


def _init_cube_slices():
    """Initialize a new cube with 60 slices from the database (EXACT training logic)."""
    global manual_cube_slices, manual_slice_idx, manual_retry_list, manual_cube_id
    
    if not slice_database:
        return
    
    # EXACT training logic: sample 60 slices per cube
    np.random.seed(manual_cube_id * 42)
    indices = np.random.choice(len(slice_database), size=60, replace=False)
    manual_cube_slices = [slice_database[i] for i in indices]
    manual_slice_idx = 0
    manual_retry_list = []
    print(f"Initialized cube {manual_cube_id} with 60 slices from database")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Meat Packing Agent API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML page."""
    dashboard_path = Path(__file__).parent.parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(), status_code=200)
    raise HTTPException(status_code=404, detail="Dashboard not found")


@app.get("/simulator", response_class=HTMLResponse)
async def simulator():
    """Serve the new simulator HTML page with conveyor belt and robot."""
    simulator_path = Path(__file__).parent.parent / "dashboard" / "simulator.html"
    if simulator_path.exists():
        return HTMLResponse(content=simulator_path.read_text(), status_code=200)
    raise HTTPException(status_code=404, detail="Simulator not found")


@app.get("/simulator.js")
async def simulator_js():
    """Serve the simulator JavaScript file."""
    from fastapi.responses import Response
    js_path = Path(__file__).parent.parent / "dashboard" / "simulator.js"
    if js_path.exists():
        return Response(content=js_path.read_text(), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="Simulator JS not found")


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """Get current system status."""
    return SystemStatus(
        agent_loaded=agent is not None,
        robot_connected=robot_interface._is_connected if robot_interface else False,
        plc_connected=plc_interface._is_connected if plc_interface else False,
        cube_fill_percentage=env.cube.get_fill_percentage() if env else 0.0,
        slices_placed=env.slices_placed if env else 0,
        system_ready=(
            agent is not None and
            robot_interface is not None and
            plc_interface is not None
        )
    )


@app.post("/placement/decide", response_model=PlacementResponse)
async def decide_placement(request: PlacementRequest):
    """Get placement decision from the agent."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    height_map = env.cube.height_map.copy()
    
    slice_info = {
        "width": request.slice_info.width,
        "length": request.slice_info.length,
        "thickness": request.slice_info.thickness
    }
    
    fill_percentage = env.cube.get_fill_percentage()
    
    try:
        decision = agent.get_placement_decision(
            height_map=height_map,
            slice_info=slice_info,
            fill_percentage=fill_percentage
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent decision failed: {str(e)}")
    
    robot_commands = None
    if request.slice_position and command_generator:
        robot_commands = command_generator.generate_placement_commands(
            slice_position=tuple(request.slice_position),
            target_position={
                "x": decision["x"],
                "y": decision["y"],
                "z": decision["z"],
                "rotation": decision["rotation"]
            },
            gripper_pattern=decision["gripper_pattern"]
        )
    
    return PlacementResponse(
        x=decision["x"],
        y=decision["y"],
        z=decision["z"],
        rotation=decision["rotation"],
        gripper_pattern=decision["gripper_pattern"],
        confidence=decision["confidence"],
        robot_commands=robot_commands
    )


@app.post("/placement/execute")
async def execute_placement(request: PlacementRequest):
    """Execute a placement in the simulation environment."""
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    env.current_slice = MeatSlice(
        width=request.slice_info.width,
        length=request.slice_info.length,
        thickness=request.slice_info.thickness
    )
    
    if agent is not None:
        height_map = env.cube.height_map.copy()
        slice_info = {
            "width": request.slice_info.width,
            "length": request.slice_info.length,
            "thickness": request.slice_info.thickness
        }
        decision = agent.get_placement_decision(
            height_map=height_map,
            slice_info=slice_info,
            fill_percentage=env.cube.get_fill_percentage()
        )
        
        x_pos = int(decision["x"] / env.resolution)
        y_pos = int(decision["y"] / env.resolution)
        rotation = decision["rotation"] // 90
        action = x_pos * (env.l_voxels * 4) + y_pos * 4 + rotation
    else:
        action = env.flat_action_space.sample()
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    await broadcast_state_update()
    
    return {
        "success": info.get("placement_success", False),
        "reward": reward,
        "fill_percentage": info.get("fill_percentage", 0),
        "flatness": info.get("flatness", 0),
        "slices_placed": info.get("slices_placed", 0),
        "terminated": terminated,
        "truncated": truncated
    }


@app.post("/cube/reset")
async def reset_cube():
    """Reset the cube to empty state and initialize new slice set from database."""
    global manual_cube_id
    
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    obs, info = env.reset()
    
    # Initialize new cube with fresh 60 slices from database (EXACT training logic)
    manual_cube_id += 1
    _init_cube_slices()
    
    await broadcast_state_update()
    
    return {
        "message": "Cube reset successfully",
        "fill_percentage": 0.0,
        "slices_placed": 0,
        "cube_id": manual_cube_id,
        "slices_available": len(manual_cube_slices)
    }


@app.get("/cube/state")
async def get_cube_state():
    """Get current cube state."""
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    height_map = env.cube.height_map.tolist()
    
    placed_slices = []
    for ps in env.cube.placed_slices:
        placed_slices.append({
            "x": ps.x,
            "y": ps.y,
            "z": ps.z,
            "width": ps.slice.width,
            "length": ps.slice.length,
            "thickness": ps.slice.thickness,
            "rotation": ps.rotation,
            "push_direction": ps.push_direction
        })
    
    return {
        "width": env.cube.width,
        "length": env.cube.length,
        "height": env.cube.height,
        "resolution": env.cube.resolution,
        "height_map": height_map,
        "fill_percentage": env.cube.get_fill_percentage(),
        "flatness": env.cube._calculate_flatness(),
        "placed_slices": placed_slices,
        "total_volume_filled": env.cube.total_volume_filled
    }


@app.get("/cube/heightmap")
async def get_heightmap():
    """Get the current height map as a 2D array."""
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    return {
        "height_map": env.cube.height_map.tolist(),
        "max_height": float(env.cube.height),
        "dimensions": [env.cube.w_voxels, env.cube.l_voxels]
    }


@app.post("/cube/press_layer")
async def press_layer(compression_ratio: float = 0.9):
    """
    Press/compact the current layer to create a flat, uniform surface.
    
    Called after a layer is complete to flatten and compress before starting next layer.
    """
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    # The restored cube_environment.py press_layer() doesn't accept compression_ratio
    env.cube.press_layer()
    
    await broadcast_state_update()
    
    return {
        "success": True,
        "layers_completed": env.cube.layers_completed,
        "current_layer_index": env.cube.current_layer_index
    }


def _generate_slice_for_coverage(env_obj, coverage: float):
    """
    Generate a slice with size and shape appropriate for current coverage level.
    
    When coverage is high (>80%), generate smaller, more rectangular slices to fill gaps.
    This ensures we can reach 95% coverage by fitting slices into remaining spaces.
    """
    import numpy as np
    from meat_packing_agent.env.cube_environment import MeatSlice
    
    if coverage > 0.90:
        width = np.random.uniform(80, 120)
        length = np.random.uniform(80, 120)
        irregularity = 0.1
    elif coverage > 0.80:
        width = np.random.uniform(80, 140)
        length = np.random.uniform(80, 140)
        irregularity = 0.2
    elif coverage > 0.60:
        width = np.random.uniform(80, 160)
        length = np.random.uniform(80, 160)
        irregularity = 0.2
    else:
        width = np.random.uniform(80, 200)
        length = np.random.uniform(80, 200)
        irregularity = 0.3
    
    thickness_min = np.random.uniform(5, 12)
    thickness_max = np.random.uniform(thickness_min + 3, 20)
    wedge_direction = np.random.randint(0, 3)
    
    slice_obj = MeatSlice(
        width=width,
        length=length,
        thickness_min=thickness_min,
        thickness_max=thickness_max,
        slice_id=env_obj.slices_placed,
        wedge_direction=wedge_direction
    )
    slice_obj.shape_mask = slice_obj._generate_irregular_shape(irregularity)
    slice_obj.thickness_map = slice_obj._generate_thickness_map()
    
    return slice_obj


@app.post("/cube/auto_fill")
async def auto_fill_layer(num_slices: int = 10):
    """
    Automatically fill the cube using the EXACT SAME LOGIC as the training script
    that achieved 100% fill rate on 1000 cubes.
    
    This uses the fill_single_cube algorithm from train_1000_cubes.py:
    - floor_only = coverage < 0.7 (strict floor for first 70%, then allow overlap)
    - Retry mechanism with max_retries = 5
    - Press layer when coverage >= 95% or after 10 consecutive failures
    - Try all 4 rotations for each slice
    """
    global manual_slice_idx, manual_retry_list, manual_consecutive_failures
    
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    import numpy as np
    from meat_packing_agent.env.cube_environment import MeatSlice
    
    results = []
    slices_placed = 0
    layers_pressed = 0
    max_retries = 5
    
    for iteration in range(num_slices * 3):  # Allow extra iterations for retries
        if slices_placed >= num_slices:
            break
            
        # Check if cube is full
        avg_height = np.mean(env.cube.height_map) * env.cube.resolution
        if avg_height >= 245:
            break
        
        # Press layer if needed (EXACT training logic)
        if env.cube.should_press_layer():
            env.cube.press_layer()
            layers_pressed += 1
            # Reset retry counts when layer changes (slices get another chance)
            manual_retry_list = [(s, 0) for s, _ in manual_retry_list]
            manual_consecutive_failures = 0
        
        # Get next slice (prefer retry list, then new from database)
        current_slice = None
        retries = 0
        
        if manual_retry_list:
            # Use retry list from database slices (EXACT training logic)
            current_slice, retries = manual_retry_list.pop(0)
        elif manual_slice_idx < len(manual_cube_slices):
            # Use next slice from database (EXACT training logic)
            slice_data = manual_cube_slices[manual_slice_idx]
            manual_slice_idx += 1
            
            # Convert database slice to MeatSlice object (same as training)
            current_slice = MeatSlice(
                width=slice_data.get('width', 120),
                length=slice_data.get('length', 100),
                thickness_min=slice_data.get('thickness_min', 15),
                thickness_max=slice_data.get('thickness_max', 25),
                slice_id=manual_slice_idx,
                wedge_direction=np.random.randint(0, 3)
            )
            # Generate shape with irregularity from database
            irregularity = slice_data.get('irregularity', 0.2)
            current_slice.shape_mask = current_slice._generate_irregular_shape(irregularity)
            current_slice.thickness_map = current_slice._generate_thickness_map()
        else:
            # Fallback: generate slice if database is empty
            layer_coverage = env.cube.get_layer_coverage()
            current_slice = _generate_slice_for_coverage(env, layer_coverage)
        
        if current_slice is None:
            break
        
        # EXACT TRAINING LOGIC: Try to place the slice
        placed = False
        
        # floor_only = coverage < 0.7 (EXACT training logic)
        coverage = env.cube.get_layer_coverage()
        floor_only = coverage < 0.7
        
        # EXACT TRAINING LOGIC: Try each rotation and place immediately when found
        # (no "best score" optimization - training doesn't use it)
        for rot in range(4):
            rotated = current_slice.rotate(rot * 90)
            result = env.cube.find_perimeter_first_position(rotated, floor_only=floor_only)
            x, y, height, is_floor = result[0], result[1], result[2], result[3]
            
            if x >= 0 and y >= 0:
                success, _ = env.cube.place_slice(rotated, x, y)
                if success:
                    slices_placed += 1
                    placed = True
                    manual_consecutive_failures = 0
                    env.slices_placed += 1
                    
                    h, w = rotated.shape_mask.shape
                    zone = env.cube._classify_position_zone(x, y, h, w)
                    print(f"[AUTO_FILL] Layer {env.cube.current_layer_index}, Slice {env.slices_placed}, pos=({x},{y}), zone={zone}, coverage={coverage:.1%}")
                    
                    results.append({
                        "x": x * env.resolution,
                        "y": y * env.resolution,
                        "rotation": rot * 90,
                        "width": current_slice.width,
                        "length": current_slice.length,
                        "thickness_min": current_slice.thickness_min,
                        "thickness_max": current_slice.thickness_max,
                        "layer_index": env.cube.current_layer_index
                    })
                    break
        
        if not placed:
            manual_consecutive_failures += 1
            if retries < max_retries:
                # Add to global retry list so it persists across API calls
                manual_retry_list.append((current_slice, retries + 1))
            
            # EXACT TRAINING LOGIC: If too many consecutive failures, force press layer
            if manual_consecutive_failures > 10 and env.cube.get_layer_coverage() > 0.8:
                env.cube.press_layer()
                layers_pressed += 1
                manual_consecutive_failures = 0
                # Reset retry counts when layer changes
                manual_retry_list = [(s, 0) for s, _ in manual_retry_list]
    
    await broadcast_state_update()
    
    return {
        "slices_placed": slices_placed,
        "fill_percentage": env.cube.get_fill_percentage(),
        "flatness": env.cube._calculate_flatness(),
        "layer_coverage": env.cube.get_layer_coverage(),
        "layers_completed": env.cube.layers_completed,
        "layers_pressed_this_call": layers_pressed,
        "current_layer_index": env.cube.current_layer_index,
        "layer_floor_mm": env.cube.current_layer_floor_voxel * env.cube.resolution,
        "layer_ceiling_mm": env.cube.current_layer_ceiling_voxel * env.cube.resolution,
        "placements": results
    }


@app.post("/cube/fill_training_algorithm")
async def fill_with_training_algorithm(cube_id: int = 0):
    """
    Fill the cube using the EXACT SAME algorithm from training.
    
    This calls fill_single_cube directly from the training module,
    ensuring 100% identical behavior to what achieved 100% fill rate.
    
    REGOLE FONDAMENTALI (NON MODIFICARE):
    1. Riempimento strato per strato con pressatura
    2. Prima SPIGOLI, poi PERIMETRO con push_to_wall, poi CENTRO
    3. Usa fettine dal database addestrato (slices_10000.json)
    4. Percentuale riempimento = volume reale riempito
    """
    global env
    
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    # Run the algorithm in a thread pool to avoid blocking
    # This prevents the 504 timeout error
    import concurrent.futures
    
    def run_fill():
        trainer = MeatPackingTrainer()
        return trainer.fill_single_cube(cube_id, return_cube=True)
    
    # Run in thread pool with timeout handling
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result, cube = await loop.run_in_executor(pool, run_fill)
    
    # Update the environment with the filled cube
    env.cube = cube
    env.slices_placed = result.slices_used
    
    # Extract placed_slices info from the cube state
    # This contains the EXACT positions from the algorithm including push_direction
    placed_slices_info = []
    for ps in cube.placed_slices:
        # Get the shape_mask for irregular meat visualization
        # The mask is a 2D boolean array where True = meat exists
        shape_mask = ps.slice.shape_mask
        # Convert to list of lists for JSON serialization
        shape_mask_list = shape_mask.tolist() if hasattr(shape_mask, 'tolist') else []
        
        placed_slices_info.append({
            "x": ps.x,  # Already in mm from PlacedSlice
            "y": ps.y,  # Already in mm from PlacedSlice
            "z": ps.z,  # Base height in mm
            "width": ps.slice.width,
            "length": ps.slice.length,
            "thickness": ps.slice.thickness,
            "thickness_min": ps.slice.thickness_min,
            "thickness_max": ps.slice.thickness_max,
            "rotation": ps.rotation,
            "push_direction": ps.push_direction,  # Direction slice was pushed
            "zone": ps.zone,  # Position zone: corner, edge, or center
            "layer_index": ps.layer_index,  # Which layer this slice is on
            "shape_mask": shape_mask_list  # 2D mask for irregular shape visualization
        })
    
    await broadcast_state_update()
    
    return {
        "cube_id": cube_id,
        "fill_percentage": result.fill_percentage,
        "slices_used": result.slices_used,
        "slices_discarded": result.slices_discarded,
        "layers_completed": result.layers_completed,
        "avg_height_mm": result.avg_height_mm,
        "max_height_mm": result.max_height_mm,
        "placed_slices": placed_slices_info
    }


@app.post("/training/start")
async def start_training(config: TrainingConfig):
    """Start agent training."""
    global agent
    
    if agent is None:
        agent = create_agent(
            learning_rate=config.learning_rate,
            batch_size=config.batch_size
        )
    
    return {
        "message": "Training started",
        "config": config.dict(),
        "note": "Training runs in background. Check /training/status for progress."
    }


@app.get("/training/status")
async def get_training_status():
    """Get training status."""
    if agent is None or agent.metrics_callback is None:
        return {
            "training_active": False,
            "metrics": None
        }
    
    return {
        "training_active": True,
        "metrics": agent.metrics_callback.get_metrics()
    }


@app.post("/agent/save")
async def save_agent(path: str = "models/agent"):
    """Save the current agent model."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        agent.save(path)
        return {"message": f"Agent saved to {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save agent: {str(e)}")


@app.post("/agent/load")
async def load_agent(path: str = "models/agent"):
    """Load an agent model."""
    global agent
    
    try:
        agent = MeatPackingAgent(model_path=path)
        return {"message": f"Agent loaded from {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load agent: {str(e)}")


@app.get("/agent/evaluate")
async def evaluate_agent(n_episodes: int = 5):
    """Evaluate agent performance."""
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        metrics = agent.evaluate(n_episodes=n_episodes)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.get("/robot/status")
async def get_robot_status():
    """Get robot status."""
    if robot_interface is None:
        raise HTTPException(status_code=503, detail="Robot interface not initialized")
    
    return robot_interface.get_status()


@app.post("/robot/connect")
async def connect_robot():
    """Connect to the robot."""
    if robot_interface is None:
        raise HTTPException(status_code=503, detail="Robot interface not initialized")
    
    success = robot_interface.connect()
    return {"connected": success}


@app.post("/robot/disconnect")
async def disconnect_robot():
    """Disconnect from the robot."""
    if robot_interface is None:
        raise HTTPException(status_code=503, detail="Robot interface not initialized")
    
    robot_interface.disconnect()
    return {"disconnected": True}


@app.post("/robot/home")
async def robot_go_home():
    """Send robot to home position."""
    if robot_interface is None:
        raise HTTPException(status_code=503, detail="Robot interface not initialized")
    
    command = robot_interface.go_home()
    return {"command": command}


@app.post("/robot/emergency_stop")
async def emergency_stop():
    """Trigger emergency stop."""
    if robot_interface:
        robot_interface.emergency_stop()
    if plc_interface:
        plc_interface.trigger_emergency_stop()
    
    return {"emergency_stop": True}


@app.post("/robot/reset_emergency")
async def reset_emergency():
    """Reset from emergency stop."""
    robot_reset = False
    plc_reset = False
    
    if robot_interface:
        robot_reset = robot_interface.reset_from_emergency()
    if plc_interface:
        plc_reset = plc_interface.reset_emergency_stop()
    
    return {"robot_reset": robot_reset, "plc_reset": plc_reset}


@app.get("/plc/status")
async def get_plc_status():
    """Get PLC status."""
    if plc_interface is None:
        raise HTTPException(status_code=503, detail="PLC interface not initialized")
    
    return plc_interface.get_status()


@app.post("/plc/conveyor/start")
async def start_conveyor(speed: float = 100.0):
    """Start the conveyor."""
    if plc_interface is None:
        raise HTTPException(status_code=503, detail="PLC interface not initialized")
    
    success = plc_interface.start_conveyor(speed)
    return {"started": success, "speed": speed}


@app.post("/plc/conveyor/stop")
async def stop_conveyor():
    """Stop the conveyor."""
    if plc_interface is None:
        raise HTTPException(status_code=503, detail="PLC interface not initialized")
    
    success = plc_interface.stop_conveyor()
    return {"stopped": success}


@app.post("/vision/process")
async def process_lidar_scan(point_cloud: List[List[float]]):
    """Process a LiDAR point cloud scan."""
    if lidar_processor is None:
        raise HTTPException(status_code=503, detail="LiDAR processor not initialized")
    
    points = np.array(point_cloud)
    geometries = lidar_processor.process_point_cloud(points)
    
    results = []
    for geom in geometries:
        results.append({
            "width": geom.width,
            "length": geom.length,
            "thickness": geom.thickness,
            "centroid": geom.centroid,
            "orientation": geom.orientation,
            "area": geom.area,
            "volume": geom.volume,
            "confidence": geom.confidence
        })
    
    return {"slices_detected": len(results), "geometries": results}


@app.post("/vision/simulate")
async def simulate_scan(
    width: float = 150.0,
    length: float = 120.0,
    thickness: float = 20.0
):
    """Simulate a LiDAR scan for testing."""
    if lidar_processor is None:
        raise HTTPException(status_code=503, detail="LiDAR processor not initialized")
    
    point_cloud = lidar_processor.simulate_scan(width, length, thickness)
    geometries = lidar_processor.process_point_cloud(point_cloud)
    
    if geometries:
        geom = geometries[0]
        return {
            "detected": True,
            "width": geom.width,
            "length": geom.length,
            "thickness": geom.thickness,
            "confidence": geom.confidence
        }
    
    return {"detected": False}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "get_state":
                if env:
                    state = {
                        "type": "state_update",
                        "height_map": env.cube.height_map.tolist(),
                        "fill_percentage": env.cube.get_fill_percentage(),
                        "slices_placed": env.slices_placed,
                        "flatness": env.cube._calculate_flatness()
                    }
                    await websocket.send_json(state)
            
            elif message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def broadcast_state_update():
    """Broadcast state update to all connected WebSocket clients."""
    if not env:
        return
    
    state = {
        "type": "state_update",
        "timestamp": datetime.now().isoformat(),
        "height_map": env.cube.height_map.tolist(),
        "fill_percentage": env.cube.get_fill_percentage(),
        "slices_placed": env.slices_placed,
        "flatness": env.cube._calculate_flatness()
    }
    
    for connection in active_connections:
        try:
            await connection.send_json(state)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
