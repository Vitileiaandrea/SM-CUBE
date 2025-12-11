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
    global env, agent, robot_interface, plc_interface, command_generator, lidar_processor
    
    env = MeatPackingEnv()
    env.reset()
    
    try:
        agent = create_agent()
    except Exception as e:
        print(f"Warning: Could not initialize agent: {e}")
        agent = None
    
    robot_interface = FanucRobotInterface()
    plc_interface = PLCInterface()
    command_generator = RobotCommandGenerator(robot_interface, plc_interface)
    lidar_processor = LiDARProcessor()
    
    print("Meat Packing Agent API initialized")


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
    """Reset the cube to empty state."""
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    obs, info = env.reset()
    
    await broadcast_state_update()
    
    return {
        "message": "Cube reset successfully",
        "fill_percentage": 0.0,
        "slices_placed": 0
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
    
    result = env.cube.press_layer(compression_ratio)
    
    await broadcast_state_update()
    
    return {
        "success": result.get("pressed", False),
        "new_layer_height": result.get("new_layer_height", 0),
        "flatness_after_press": result.get("flatness_after_press", 0),
        "layers_completed": result.get("layers_completed", 0)
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
    Automatically fill the cube using STRICT layer-by-layer strategy with
    CIRCULAR CONVEYOR slice selection.
    
    STRICT LAYER CONSTRAINT: Each layer must reach 95% coverage before
    starting the next layer. This is a physical requirement because the
    gripper cannot reach lower positions once slices are placed higher up.
    
    CIRCULAR CONVEYOR: The conveyor belt is circular, so slices that don't
    fit can be skipped and will come back around. The algorithm generates
    multiple candidate slices and picks the best one for the current gaps.
    
    The algorithm:
    1. Generate multiple candidate slices (simulating conveyor with multiple slices)
    2. For each candidate, find the best position within current layer bounds
    3. Pick the slice that best fills gaps (lowest gap score)
    4. If no slice fits and coverage >= 95%, press and advance layer
    5. If coverage < 95%, generate smaller slices to fill remaining gaps
    """
    if env is None:
        raise HTTPException(status_code=503, detail="Environment not initialized")
    
    import numpy as np
    
    results = []
    slices_placed = 0
    layers_pressed = 0
    slices_skipped = 0
    
    conveyor_size = 5
    
    for _ in range(num_slices):
        layer_coverage = env.cube.get_layer_coverage()
        
        if layer_coverage >= env.cube.LAYER_COVERAGE_THRESHOLD:
            press_result = env.cube.press_layer()
            layers_pressed += 1
            layer_coverage = env.cube.get_layer_coverage()
        
        candidates = []
        for _ in range(conveyor_size):
            slice_obj = _generate_slice_for_coverage(env, layer_coverage)
            candidates.append(slice_obj)
        
        best_candidate = None
        best_pos = None
        best_score = float('inf')
        best_rotation = 0
        
        for slice_obj in candidates:
            for rotation in range(4):
                rotated = slice_obj.rotate(rotation * 90)
                result = env.cube.find_perimeter_first_position(rotated)
                x, y, height = result[0], result[1], result[2]
                is_floor_level = result[3] if len(result) > 3 else True
                
                if x >= 0 and y >= 0 and is_floor_level:
                    can_place_result, _ = env.cube.can_place(rotated, x, y, enforce_layer_constraint=True)
                    if can_place_result:
                        score = env.cube._calculate_gap_score(rotated, x, y, height)
                        if score < best_score:
                            best_score = score
                            best_pos = (x, y)
                            best_rotation = rotation
                            best_candidate = slice_obj
        
        if best_candidate is None:
            slices_skipped += 1
            
            if layer_coverage >= 0.90:
                smaller_slices = []
                for _ in range(10):
                    width = np.random.uniform(80, 100)
                    length = np.random.uniform(80, 100)
                    from meat_packing_agent.env.cube_environment import MeatSlice
                    smaller = MeatSlice(
                        width=width,
                        length=length,
                        thickness_min=15,
                        thickness_max=25,
                        slice_id=env.slices_placed
                    )
                    smaller.shape_mask = smaller._generate_irregular_shape(0.1)
                    smaller.thickness_map = smaller._generate_thickness_map()
                    smaller_slices.append(smaller)
                
                for slice_obj in smaller_slices:
                    for rotation in range(4):
                        rotated = slice_obj.rotate(rotation * 90)
                        result = env.cube.find_perimeter_first_position(rotated)
                        x, y, height = result[0], result[1], result[2]
                        is_floor_level = result[3] if len(result) > 3 else True
                        
                        if x >= 0 and y >= 0 and is_floor_level:
                            can_place_result, _ = env.cube.can_place(rotated, x, y, enforce_layer_constraint=True)
                            if can_place_result:
                                score = env.cube._calculate_gap_score(rotated, x, y, height)
                                if score < best_score:
                                    best_score = score
                                    best_pos = (x, y)
                                    best_rotation = rotation
                                    best_candidate = slice_obj
            
            if best_candidate is None:
                if slices_skipped > 30:
                    break
                continue
        
        env.current_slice = best_candidate
        rotated_slice = best_candidate.rotate(best_rotation * 90)
        
        h, w = rotated_slice.shape_mask.shape
        zone = env.cube._classify_position_zone(best_pos[0], best_pos[1], h, w)
        print(f"[AUTO_FILL] Layer {env.cube.current_layer_index}, Slice {env.slices_placed}, pos=({best_pos[0]},{best_pos[1]}), zone={zone}")
        
        success, metrics = env.cube.place_slice(rotated_slice, best_pos[0], best_pos[1])
        
        if success:
            slices_placed += 1
            slices_skipped = 0
            env.slices_placed += 1
            results.append({
                "x": best_pos[0] * env.resolution,
                "y": best_pos[1] * env.resolution,
                "rotation": best_rotation * 90,
                "width": best_candidate.width,
                "length": best_candidate.length,
                "thickness_min": best_candidate.thickness_min,
                "thickness_max": best_candidate.thickness_max,
                "layer_index": env.cube.current_layer_index
            })
        
        layer_coverage = env.cube.get_layer_coverage()
        if layer_coverage >= env.cube.LAYER_COVERAGE_THRESHOLD:
            env.cube.press_layer()
            layers_pressed += 1
    
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
