# Traffic Simulation with Krauss Car-Following Model

A traffic simulation system that generates realistic vehicle trajectories on custom-marked road networks. The simulation uses the Krauss car-following model and MOBIL lane-changing model to create realistic vehicle behavior.

## What It Does

This tool allows you to:
1. **Mark custom road networks** on aerial images using an interactive UI
2. **Simulate traffic** with realistic car-following and lane-changing behavior
3. **Generate CSV output** with vehicle bounding boxes for visualization
4. **Visualize results** as videos showing vehicles moving along marked lanes

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Mark Lanes on Your Image

Use the interactive tool to mark centerlines, spawn points, and lane connections:

```bash
python mark_centerlines_ui.py --image eda/data/media/SiteA.jpg --output eda/data/lanes.json
```

**Controls:**
- **Draw Mode** (default): Left-click to add points to a centerline
  - `n`: Start new lane
  - `s`: Save current lane
  - `u`: Undo last point
  - `1-9, 0`: Select lane by number
- **Spawn Mode** (`t`): Toggle spawn points on lanes
  - `t`: Toggle spawn point for selected lane (vehicles spawn at first point)
- **Connection Mode** (`c`): Create lane-changing connections
  - Left-click on source lane → target lane to create transition path
  - `ENTER`: Finish connection
  - `ESC`: Cancel connection
- **Navigation**:
  - Right mouse + drag: Pan
  - `+/-`: Zoom in/out
  - `r`: Reset zoom/pan
- **Save**: `w` to save all data, `q` to quit

### 3. Run Simulation

Run the simulation with your marked lanes:

```bash
python run_mesa_simulation.py \
    --custom-lanes eda/data/lanes.json \
    --output mesa_simulation.csv \
    --frames 300
```

**Key Parameters:**
- `--custom-lanes`: Path to your marked lanes JSON file
- `--output`: Output CSV file path
- `--frames`: Number of frames to simulate (default: 900)
- `--fps`: Frames per second (default: 8.0)
- `--max-vehicles`: Maximum vehicles in simulation (default: 150)
- `--spawn-rate`: Vehicles per second (default: 0.5)

### 4. Visualize Results

Create a video visualization of the simulation:

```bash
python eda/vis/traffic_visualizer.py \
    --image eda/data/media/SiteA.jpg \
    --csv mesa_simulation.csv \
    --output traffic_video.mp4
```

## How It Works

### Vehicle Behavior

- **Krauss Car-Following Model**: Vehicles maintain safe following distances based on speed, reaction time, and distance to leader
- **MOBIL Lane-Changing**: Vehicles change lanes when it's beneficial and safe
- **Lane Following**: Vehicles strictly follow marked centerlines
- **Automatic Transitions**: Vehicles automatically transition to connected lanes at lane ends

### Lane Marking

- **Centerlines**: Single lines marking the path vehicles follow
- **Spawn Points**: Marked on lanes where vehicles should spawn (at first point)
- **Lane Connections**: Transition paths between lanes for lane-changing or merging

### Coordinate System

The simulation uses a world coordinate system that's transformed from image coordinates:
- Image coordinates are converted to world coordinates using offset and scale
- Default: `offset_x=2064.0`, `offset_y=526.0`, `scale=0.3507` (pixels per meter)
- Vehicles spawn and move in world coordinates, then converted back to image coordinates for CSV output

## Project Structure

```
TrafficSimulation/
├── mark_centerlines_ui.py    # Interactive lane marking tool
├── run_mesa_simulation.py     # Main simulation runner
├── src/
│   ├── agents/
│   │   └── vehicle.py        # Vehicle agent with car-following and lane-changing
│   ├── models/
│   │   ├── traffic_model.py  # Main MESA simulation model
│   │   └── road_network.py   # Road network and lane representation
│   └── utils/
│       ├── krauss_model.py   # Krauss car-following model
│       ├── mobil_model.py    # MOBIL lane-changing model
│       └── simulation_to_csv.py  # CSV conversion utility
└── eda/
    ├── vis/
    │   └── traffic_visualizer.py  # Video visualization tool
    └── data/
        ├── lanes.json        # Marked lane data
        └── media/
            └── SiteA.jpg     # Background image
```

## Features

- ✅ Interactive lane marking with zoom/pan
- ✅ Realistic car-following behavior (Krauss model)
- ✅ Intelligent lane-changing (MOBIL model)
- ✅ Custom road networks from marked images
- ✅ Vehicle spawning at designated points
- ✅ Automatic lane transitions at connections
- ✅ CSV output for visualization
- ✅ Video generation from simulation results

## Technical Details

### Simulation Parameters

- **Time Step**: 0.1 seconds
- **Vehicle Speed**: 5x faster than normal (for realistic frame generation)
- **FPS**: 8 frames per second (default)
- **Max Vehicles**: 150 (configurable)
- **Spawn Rate**: 0.5 vehicles/second (configurable)

### Models Used

- **Krauss Model**: Calculates safe following speed based on distance to leader, reaction time, and random deceleration
- **MOBIL Model**: Evaluates lane-changing incentives considering acceleration gains and safety

## Example Workflow

```bash
# 1. Mark lanes on your image
python mark_centerlines_ui.py --image eda/data/media/SiteA.jpg --output eda/data/lanes.json

# 2. Run simulation
python run_mesa_simulation.py --custom-lanes eda/data/lanes.json --output mesa_simulation.csv --frames 300

# 3. Visualize
python eda/vis/traffic_visualizer.py --image eda/data/media/SiteA.jpg --csv mesa_simulation.csv --output video.mp4
```

## License

This project is licensed under the MIT License.
