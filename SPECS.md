# FM 7000 - Filling Machine - Technical Specifications

## Overview

The FM 7000 (Filling Machine) is an automated system for filling 210x210x250mm cubes with meat slices of varying types and thicknesses. The filled cubes are then fed into the SM 7000 (Skewering Machine), which cuts them vertically into arrosticini (Italian meat skewers).

The FM 7000 must produce approximately **31 cubes/hour** (~1 cube every 2 minutes) to keep up with the SM 7000's output of **7,000 arrosticini/hour** (225 arrosticini per cube).

## Product: Arrosticini

Each arrosticino is a vertical slice cut from the filled cube. The composition of each arrosticino follows a specific recipe (from first bite to last bite):

1. **Carne alta qualita** (high quality meat) - first bite must be good
2. **Grasso** (fat layer)
3. **Carne media qualita** (medium quality meat)
4. **Grasso** (fat layer)
5. **Carne media qualita** (medium quality meat)
6. **Grasso** (fat layer)
7. ... (repeating medium + fat pattern)
8. **Carne alta qualita** (high quality meat) - last bite makes you want another one

This "bookend" strategy ensures the consumer's first and last bites are premium quality.

## Cube Specifications

- **Internal dimensions**: 210mm x 210mm x 250mm (width x length x height)
- **Arrosticini per cube**: 225
- **Target fill rate**: ~31 cubes/hour (to feed SM 7000)

## Hardware Components

### Robot: KUKA KR 3 DELTA D1200 HM
- **Type**: Delta robot (parallel kinematics)
- **Payload**: 3 kg rated (6 kg max)
- **Reach**: 1200mm diameter workspace
- **Vertical workspace**: 250mm
- **Cycle time**: 0.5 sec (Adept cycle benchmark)
- **Realistic pick & place cycle**: ~2-2.5 sec per slice
- **Wrist rotation (Axis 4)**: 360 degrees continuous
- **Protection**: IP67 body, IP69K axis 4
- **Certifications**: TUV food safety, US FDA, GER LFGB
- **Controller**: KR C5 micro + KSS 8.7
- **Software**: PickControl 1.3 (conveyor tracking + vision)
- **Material**: Full stainless steel, hygienic design
- **Scalability**: Up to 10 robots per line via PickControl

### End Effector: 4x4 Vacuum Gripper
- **Configuration**: 16 vacuum cups in 4x4 grid
- **Cup diameter**: 30mm (bellows/accordion style for soft materials)
- **External interaxis**: 180mm x 180mm
- **Cup spacing**: ~40mm between cup centers
- **Cube internal space**: 210mm x 210mm
- **Gap gripper-to-wall**: 15mm per side (210-180)/2

### Vision System: Camera + Cognex ViDi
- **Camera**: 4K or 8K resolution (model TBD)
- **Software**: Cognex ViDi deep learning suite
- **Purpose on conveyor**:
  - Classify meat slices (high quality, medium quality, fat)
  - Detect slice position and orientation on belt
  - Identify fat percentage distribution within each slice
- **ViDi tools used**:
  - GREEN-CLASSIFY: classify meat type/quality
  - BLUE-LOCATE: locate and orient slices on belt
  - RED-ANALYZE: detect fat marbling/distribution patterns

### Profiler: Laser Profiler
- **Brand/Model**: TBD
- **Purpose**:
  - Measure slice thickness (non-uniform/wedge shaped: 5-40mm)
  - Generate 3D thickness map of each slice
  - Detect fat percentage distribution on the slice surface
  - Used for wedge-matching decisions (thin side with thick side)

### LiDAR: Cube Fill Monitor
- **Purpose**: Mounted ABOVE the cube to monitor fill state in real-time
- **NOT on the conveyor** - specifically for cube fill level verification
- **Provides feedback to RL agent** about current cube state

### Conveyor Belt
- **3 separate lanes/sections**:
  1. Carne alta qualita (high quality meat)
  2. Carne media qualita (medium quality meat)
  3. Grasso (fat)
- Pre-sorted material arrives on separate lanes

### Compute
- **PC locale** (local PC) running the RL agent
- No PLC required for initial version
- Agent runs locally and communicates with KUKA controller

## Placement Rules (Deterministic Base)

### 1. Recipe Layer Sequence
Layers are stacked bottom-to-top following the arrosticino recipe:
- Layer 1: High quality meat
- Layer 2: Fat
- Layer 3: Medium quality meat
- Layer 4: Fat
- Layer 5: Medium quality meat
- Layer 6: Fat
- ... (continue pattern)
- Layer N-1: Fat
- Layer N: High quality meat (top)

### 2. Perimeter-First Filling Strategy
Within each layer, slices are placed following priority order:
1. **CORNERS (spigoli)**: positions touching 2 cube walls
2. **EDGES (perimeter)**: positions touching 1 cube wall
3. **CENTER**: positions not touching any wall

This mimics expert operator technique: fill from outside in.

### 3. Push-to-Wall
After positioning a slice, the robot pushes it against the nearest wall(s):
- **Corners**: slice pushed 10mm beyond corner contact point against both walls
- **Single wall**: slice pushed 25mm against the wall
- **Push threshold**: activate when slice is within 30mm of a wall
- **Safety margin**: slice must protrude at least 10mm beyond the outer lip of the vacuum cup on the push side, to prevent cups from hitting cube walls

### 4. Wedge Thickness Matching
Meat slices are NOT uniform thickness (wedge/cuneo shaped):
- Each slice has a thick side and a thin side
- When placing adjacent slices, the thin side of one must match the thick side of the neighbor
- This creates a flat, uniform layer surface
- The profiler provides the thickness map for each slice

### 5. Fat Distribution (Intrinsic)
- Independent of the fat LAYERS in the recipe
- Each meat slice (even "lean" ones) has zones with varying fat percentage (marbling)
- The agent must position/rotate slices so that **fatty zones do NOT overlap vertically** with fatty zones of the layer below
- Prevents "hidden fat columns" in the finished arrosticino

### 6. Layer Completion
- Each layer must be filled to **95% minimum coverage** before starting the next
- Slices are soft/flexible - slight overlaps are OK, they conform and flatten
- After layer completion, a **press operation** compacts the layer (10% compression)
- The gripper physically cannot reach lower positions once higher slices are placed

## Gripper Strategy

### Pattern Selection
The 4x4 vacuum cup pattern determines where the slice lands in the cube:
- **Corner cups** (e.g., top-left 2x2) -> slice positioned at corresponding cube corner
- **Edge cups** (e.g., top 2x2) -> slice at cube edge
- **Center cups** (center 2x2) -> slice at cube center

### Decision Flow
1. Agent decides **target position** in cube (corner/edge/center + exact location)
2. Based on target -> selects **vacuum cup pattern** (which cups to activate)
3. Based on slice orientation needs -> selects **wrist rotation** (0-360 degrees)
4. Robot picks slice from conveyor with selected pattern
5. Moves to cube and places
6. Push-to-wall if near wall/corner

### Safety Constraint
- Slice must extend **at least 10mm beyond the vacuum cup lip** on the side facing the cube wall
- Cup diameter: 30mm, so the slice edge must be at least ~25mm from cup center on push side
- This prevents vacuum cups from colliding with cube walls during push-to-wall

## Architecture: Hybrid Rule-Based + RL

### Deterministic Rule Engine (Base Layer)
Handles all fixed rules:
- Recipe sequence (which meat type for current layer)
- Perimeter-first priority (corner > edge > center)
- Push-to-wall mechanics
- Gripper pattern selection from target zone
- Safety constraints (cup clearance, layer bounds)

### RL Agent (Optimization Layer)
Optimizes on top of rules:
- Exact position within the valid zone
- Optimal rotation for wedge-matching
- Fat distribution balancing across layers
- Slice selection from available slices on conveyor
- Throughput optimization to meet SM 7000 demand

## Performance Targets

| Metric | Target |
|--------|--------|
| Cubes per hour | 31 |
| Arrosticini per hour | 7,000 |
| Arrosticini per cube | 225 |
| Estimated slices per cube | 60-80 |
| Pick & place cycle | ~2-2.5 sec |
| Multi-robot support | Up to 10 KUKA deltas |
