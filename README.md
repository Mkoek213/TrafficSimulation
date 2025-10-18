# Traffic Simulation with Krauss Car-Following Model

A traffic simulation system built with MESA and PyGame that implements the Krauss car-following model for realistic vehicle behavior. The simulation supports lane-changing, intersections, and can load real-world road networks from OpenStreetMap data.

## Features

- **Krauss Car-Following Model**: Realistic vehicle behavior with safe following distances and random deceleration
- **Lane-Changing**: Vehicles can change lanes when safe conditions are met
- **Road Networks**: Support for multi-lane roads and intersections
- **Real-time Visualization**: PyGame-based visualization with zoom, pan, and statistics
- **OpenStreetMap Integration**: Load real-world road networks from OSM data
- **MESA Framework**: Agent-based modeling with proper scheduling and state management
- **Configurable Parameters**: Adjustable simulation parameters for different scenarios

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd TrafficSimulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Simulation
Run the simulation with default parameters:
```bash
python main.py
```

### Custom Parameters
```bash
python main.py --width 1500 --height 1000 --max-vehicles 100 --spawn-rate 0.2
```

### Headless Mode
Run without visualization for performance testing:
```bash
python main.py --headless --steps 1000
```

## Usage

### Command Line Options

- `--width`: Simulation width (default: 1000)
- `--height`: Simulation height (default: 1000)
- `--time-step`: Time step in seconds (default: 0.1)
- `--spawn-rate`: Vehicle spawn rate per second (default: 0.1)
- `--max-vehicles`: Maximum number of vehicles (default: 50)
- `--viz-width`: Visualization window width (default: 1200)
- `--viz-height`: Visualization window height (default: 800)
- `--fps`: Target FPS (default: 60)
- `--headless`: Run without visualization
- `--steps`: Number of simulation steps (headless mode only)

### Controls

When running with visualization:
- **SPACE**: Pause/Resume simulation
- **R**: Reset simulation
- **Mouse Wheel**: Zoom in/out
- **Mouse Drag**: Pan around the simulation
- **ESC**: Exit

## Architecture

### Core Components

1. **KraussModel** (`src/utils/krauss_model.py`): Implements the Krauss car-following model
2. **Vehicle** (`src/agents/vehicle.py`): Vehicle agent with car-following and lane-changing behavior
3. **RoadNetwork** (`src/models/road_network.py`): Road network representation with lanes and intersections
4. **TrafficSimulationModel** (`src/models/traffic_model.py`): Main MESA model managing the simulation
5. **TrafficVisualization** (`src/visualization/pygame_viz.py`): PyGame-based visualization
6. **OSMNetworkLoader** (`src/utils/osm_loader.py`): OpenStreetMap data loading

### Krauss Model Implementation

The Krauss car-following model calculates vehicle speed based on:

1. **Safe Speed**: Based on distance to leading vehicle and reaction time
2. **Desired Speed**: Free flow speed or safe following speed
3. **Random Deceleration**: Stochastic driver behavior
4. **Physical Constraints**: Maximum acceleration and deceleration limits

### Vehicle Behavior

- **Car-Following**: Vehicles maintain safe distances using the Krauss model
- **Lane-Changing**: Vehicles can change lanes when safe conditions are met
- **Speed Adaptation**: Vehicles adjust speed based on traffic conditions
- **Collision Avoidance**: Built-in safety mechanisms prevent collisions

## Road Network

### Creating Custom Networks

```python
from src.models.road_network import RoadNetwork, Point

# Create a new road network
network = RoadNetwork()

# Create a highway
start_point = Point(0, 0)
end_point = Point(1000, 0)
highway = network.create_simple_highway(start_point, end_point, num_lanes=3)
```

### OpenStreetMap Integration

```python
from src.utils.osm_loader import OSMNetworkLoader

# Load network from place name
loader = OSMNetworkLoader()
network = loader.load_network_from_place("Manhattan, New York, USA")

# Load network from bounding box
network = loader.load_network_from_bbox(40.7, 40.6, -74.0, -74.1)
```

## Simulation Parameters

### Vehicle Parameters
- **Max Speed**: 25-35 m/s (90-126 km/h)
- **Max Acceleration**: 1.5-2.5 m/s²
- **Max Deceleration**: -3.5 to -4.5 m/s²
- **Reaction Time**: 1.0 seconds
- **Random Deceleration**: 10% probability, 0.5 m/s² max

### Simulation Parameters
- **Time Step**: 0.1 seconds
- **Spawn Rate**: 0.1 vehicles/second
- **Max Vehicles**: 50 (configurable)

## Performance

The simulation is optimized for real-time performance:
- Efficient vehicle collision detection
- Optimized rendering with PyGame
- Configurable frame rates and vehicle limits
- Headless mode for batch processing

## Future Enhancements

- **Traffic Lights**: Advanced intersection control
- **Route Planning**: A* pathfinding for vehicles
- **Traffic Jams**: Congestion modeling
- **Different Vehicle Types**: Trucks, motorcycles, etc.
- **Weather Effects**: Rain, snow impact on driving
- **Accident Simulation**: Collision detection and response
- **Data Export**: CSV/JSON output for analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MESA framework for agent-based modeling
- PyGame for visualization
- OpenStreetMap for road data
- Krauss et al. for the car-following model