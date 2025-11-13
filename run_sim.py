"""Simple script to run traffic simulation and create video"""

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

print("🚗 Creating simulation...")
model = TrafficSimulationModel(
    custom_lanes_path=LANES_JSON,
    time_step=TIME_STEP,
    max_vehicles=MAX_VEHICLES
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

