# Meat Packing RL Agent

An intelligent reinforcement learning agent for optimal meat slice packing in SM7000 machine cubes. The system uses PPO (Proximal Policy Optimization) to learn how to efficiently fill 210x210x250mm cubes with meat slices of varying sizes and thicknesses.

## System Overview

This project implements an automated meat packing system with the following components:

### Core Components

1. **3D Bin-Packing Environment** (`meat_packing_agent/env/`)
   - Simulates the cube container (210x210x250mm internal capacity)
   - Handles meat slices of varying dimensions (50-200mm) and thicknesses (5-40mm)
   - Tracks height maps, volume utilization, and layer flatness

2. **RL Agent** (`meat_packing_agent/agent/`)
   - PPO-based agent using stable-baselines3
   - Learns optimal placement strategies
   - Outputs position (x, y), rotation, and gripper finger pattern

3. **Vision System Interface** (`meat_packing_agent/vision/`)
   - Processes LiDAR 3D point cloud data
   - Extracts slice geometry (shape, dimensions, thickness)
   - Tracks slices on the conveyor belt

4. **Robot Interface** (`meat_packing_agent/robot/`)
   - Controls Fanuc anthropomorphic/SCARA robot
   - Manages 5-finger vacuum gripper
   - Interfaces with Schneider PLC for conveyor control

5. **API Backend** (`meat_packing_agent/api/`)
   - FastAPI REST endpoints for system control
   - WebSocket support for real-time updates
   - Training management and model persistence

6. **Dashboard** (`meat_packing_agent/dashboard/`)
   - Real-time 3D visualization of cube filling
   - Height map display
   - System status monitoring
   - Manual controls

## Installation

```bash
# Clone the repository
git clone https://github.com/Vitileiaandrea/vnwash-nodejs.git
cd vnwash-nodejs

# Install dependencies with Poetry
poetry install

# Or with pip
pip install -e .
```

## Quick Start

### Training the Agent

```bash
# Train with default settings (100k timesteps)
poetry run python -m meat_packing_agent.train

# Train with custom settings
poetry run python -m meat_packing_agent.train \
    --timesteps 500000 \
    --learning-rate 1e-4 \
    --batch-size 128 \
    --save-path models/my_agent

# Continue training from a saved model
poetry run python -m meat_packing_agent.train \
    --load-path models/my_agent \
    --timesteps 100000

# Evaluate only
poetry run python -m meat_packing_agent.train \
    --load-path models/my_agent \
    --eval-only \
    --eval-episodes 20
```

### Running the API Server

```bash
# Start the FastAPI server
poetry run uvicorn meat_packing_agent.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Using the Dashboard

1. Start the API server (see above)
2. Open `meat_packing_agent/dashboard/index.html` in a browser
3. Click "Connect" to connect to the API
4. Use the controls to test the system

## API Endpoints

### System Status
- `GET /status` - Get system status
- `GET /cube/state` - Get current cube state
- `GET /cube/heightmap` - Get height map data

### Placement
- `POST /placement/decide` - Get placement decision from agent
- `POST /placement/execute` - Execute placement in simulation
- `POST /cube/reset` - Reset cube to empty state

### Training
- `POST /training/start` - Start agent training
- `GET /training/status` - Get training progress
- `POST /agent/save` - Save current model
- `POST /agent/load` - Load saved model
- `GET /agent/evaluate` - Evaluate agent performance

### Robot Control
- `GET /robot/status` - Get robot status
- `POST /robot/connect` - Connect to robot
- `POST /robot/home` - Send robot to home position
- `POST /robot/emergency_stop` - Trigger emergency stop

### PLC Control
- `GET /plc/status` - Get PLC status
- `POST /plc/conveyor/start` - Start conveyor
- `POST /plc/conveyor/stop` - Stop conveyor

### Vision
- `POST /vision/process` - Process LiDAR point cloud
- `POST /vision/simulate` - Simulate a scan for testing

## Configuration

### Cube Dimensions
- Width: 210mm
- Length: 210mm
- Height: 250mm
- Resolution: 5mm per voxel

### Meat Slice Constraints
- Width/Length: 50-200mm
- Thickness: 5-40mm
- Irregular shapes supported

### Robot Configuration
- Type: Fanuc anthropomorphic/SCARA
- End effector: 5-finger vacuum gripper
- PLC: Schneider

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   LiDAR Sensor  │────▶│ Vision Processor│
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  Conveyor Belt  │     │    RL Agent     │
│   (Schneider)   │     │     (PPO)       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│      PLC        │◀───▶│ Robot Interface │
│   Interface     │     │    (Fanuc)      │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │   SM7000 Cube   │
                        └─────────────────┘
```

## Development

### Running Tests

```bash
poetry run pytest
```

### Code Formatting

```bash
poetry run black meat_packing_agent/
poetry run ruff check meat_packing_agent/
```

## License

MIT License

## Author

Vitileiaandrea
