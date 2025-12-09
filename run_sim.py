import argparse
import sys

# ============ CONFIG ============
LANES_JSON = "eda/data/lanes.json"  # Your lanes file (or None for test network)
# LANES_JSON = "data/custom_lanes.json"
IMAGE_FILE = "eda/data/media/SiteA.jpg"
OUTPUT_CSV = "eda/data/drift/simulation.csv"
OUTPUT_VIDEO = "simulation.mp4"

NUM_FRAMES = 300
TIME_STEP = 0.1
MAX_VEHICLES = 12
FPS = 8.0
DEFAULT_SPAWN_RATE = 0.5

# Image params for coordinate transformation
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160
OFFSET_X = 2064.0
OFFSET_Y = 526.0
SCALE = 0.3507
# ================================

from src.models.traffic_model import TrafficSimulationModel
from src.utils.simulation_to_csv import SimulationToCSVConverter
from eda.vis.traffic_visualizer import TrafficVisualizer

def parse_arguments():
    """Parse command line arguments for per-lane spawn rates."""
    parser = argparse.ArgumentParser(description='Run traffic simulation.')
    
    # Parse known args to allow for flexible -laneX arguments
    args, unknown = parser.parse_known_args()
    
    spawn_rates = {}
    
    # Process unknown arguments for -laneX pattern
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        # Handle both -laneX and --laneX
        if arg.startswith('-lane') or arg.startswith('--lane'):
            try:
                # Remove leading dashes and 'lane' prefix
                lane_str = arg.lstrip('-').replace('lane', '')
                lane_id = int(lane_str)
                
                if i + 1 < len(unknown):
                    rate = float(unknown[i+1])
                    spawn_rates[lane_id] = rate
                    i += 2
                else:
                    print(f"Warning: No value provided for {arg}")
                    i += 1
            except ValueError:
                print(f"Warning: Invalid lane argument format {arg}")
                i += 1
        else:
            i += 1
            
    return spawn_rates

if __name__ == "__main__":
    # Parse per-lane spawn rates
    per_lane_rates = parse_arguments()
    
    # Determine spawn rate configuration
    if per_lane_rates:
        print(f"🚗 Using per-lane spawn rates: {per_lane_rates}")
        vehicle_spawn_rate = per_lane_rates
    else:
        print(f"🚗 Using global spawn rate: {DEFAULT_SPAWN_RATE}")
        vehicle_spawn_rate = DEFAULT_SPAWN_RATE

    print("🚗 Creating simulation...")
    model = TrafficSimulationModel(
        custom_lanes_path=LANES_JSON,
        time_step=TIME_STEP,
        max_vehicles=MAX_VEHICLES,
        vehicle_spawn_rate=vehicle_spawn_rate
    )

    print("📊 Running simulation and saving CSV...")
    converter = SimulationToCSVConverter(
        model=model,
        image_width=IMAGE_WIDTH,
    image_height=IMAGE_HEIGHT,
    offset_x=OFFSET_X,
    offset_y=OFFSET_Y,
    scale=SCALE
    )
    converter.run_simulation_and_save(OUTPUT_CSV, NUM_FRAMES, FPS)

    print("🎬 Creating video...")
    viz = TrafficVisualizer(IMAGE_FILE, OUTPUT_CSV)
    viz.create_video(OUTPUT_VIDEO, fps=FPS)

    print(f"✅ Done! Video saved to: {OUTPUT_VIDEO}")

