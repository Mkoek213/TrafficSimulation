"""
Traffic Jam Video Generator

Creates a video showing:
1. Traffic lights RED → congestion builds up
2. Traffic lights switch to NORMAL → congestion clears

Perfect for presentations demonstrating traffic flow dynamics.
"""

import sys
import pandas as pd
from pathlib import Path

# ============ CONFIG ============
LANES_JSON = "eda/data/lanes.json"
IMAGE_FILE = "eda/data/media/SiteA.jpg"
OUTPUT_CSV = "jam_simulation.csv"
OUTPUT_VIDEO = "traffic_jam.mp4"

# Jam creation phase settings
JAM_DURATION_FRAMES = 350      # How many frames to keep lights RED (congestion builds)
JAM_SPAWN_RATE = 5           # Cars per second during jam creation

# Normal clearing phase settings  
CLEARING_DURATION_FRAMES = 1000  # How many frames for normal operation (clearing)
CLEARING_SPAWN_RATE = 0.5       # Lower spawn rate during clearing

# Simulation settings
TIME_STEP = 0.1
MAX_VEHICLES = 150
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


def run_jam_video_simulation():
    """
    Run simulation that creates a traffic jam, then clears it.
    Returns combined DataFrame of all frames.
    """
    print("=" * 60)
    print("🚗 TRAFFIC JAM VIDEO GENERATOR")
    print("=" * 60)
    
    # Initialize model with high spawn rate for jam creation
    print("\n📦 Creating simulation model...")
    model = TrafficSimulationModel(
        custom_lanes_path=LANES_JSON,
        time_step=TIME_STEP,
        max_vehicles=MAX_VEHICLES,
        vehicle_spawn_rate=JAM_SPAWN_RATE,
        model_boost=1.0
    )
    
    # Create CSV converter
    converter = SimulationToCSVConverter(
        model=model,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT
    )
    
    # Calculate steps per frame for proper FPS
    time_per_frame = 1.0 / FPS
    steps_per_frame = max(1, round(time_per_frame / TIME_STEP))
    
    print(f"  FPS: {FPS}, Steps per frame: {steps_per_frame}")
    
    all_frames_data = []
    frame_idx = 0
    
    # =========================================================================
    # PHASE 1: JAM CREATION - All lights RED, high spawn rate
    # =========================================================================
    print(f"\n🔴 [PHASE 1] Creating traffic jam... (frames 0-{JAM_DURATION_FRAMES-1})")
    print(f"   - All traffic lights: RED")
    print(f"   - Spawn rate: {JAM_SPAWN_RATE} cars/second")
    
    for phase_frame in range(JAM_DURATION_FRAMES):
        # Force all traffic lights to RED before each step
        for tl in model.traffic_lights_node._traffic_lights_instances:
            tl._state = 'red'
        
        # Also freeze intersection-based lights
        for intersection in model.road_network.intersections:
            for lane_id in intersection.traffic_lights:
                intersection.traffic_lights[lane_id] = 'red'
        
        # Run simulation steps for this frame
        for _ in range(steps_per_frame):
            # Keep forcing lights red during the step
            for tl in model.traffic_lights_node._traffic_lights_instances:
                tl._state = 'red'
            model.step()
        
        # Collect frame data
        frame_data = converter.get_frame_data(frame_idx)
        all_frames_data.append(frame_data)
        
        if phase_frame % 30 == 0:
            print(f"   Frame {frame_idx}: {len(model.vehicles)} vehicles queued")
        
        frame_idx += 1
    
    jam_size = len(model.vehicles)
    print(f"\n✅ Phase 1 complete: {jam_size} vehicles in jam")
    
    # =========================================================================
    # PHASE 2: JAM CLEARING - Normal lights, reduced spawn rate
    # =========================================================================
    print(f"\n🟢 [PHASE 2] Clearing traffic jam... (frames {frame_idx}-{frame_idx + CLEARING_DURATION_FRAMES - 1})")
    print(f"   - Traffic lights: NORMAL CYCLE")
    print(f"   - Spawn rate: {CLEARING_SPAWN_RATE} cars/second")
    
    # Reset traffic lights to normal cycling
    model.traffic_lights_node._steps_counter = 0
    model.traffic_lights_node._stage_counter = 0
    model.traffic_lights_node._in_interstage_buffer = False
    model.traffic_lights_node._change_stage()
    
    # Reset intersection traffic light timers
    for intersection in model.road_network.intersections:
        intersection.traffic_light_timer = 0.0
        intersection.current_phase = 0
        if intersection.traffic_light_phases:
            _, first_phase = intersection.traffic_light_phases[0]
            intersection.traffic_lights = first_phase.copy()
    
    # Reduce spawn rate for clearing phase
    model.vehicle_spawn_rate = CLEARING_SPAWN_RATE
    if model.spawn_timers:
        for lane_id in model.spawn_timers:
            model.spawn_timers[lane_id] = 0.0
    if model.global_spawn_timer is not None:
        model.global_spawn_timer = 0.0
    
    for phase_frame in range(CLEARING_DURATION_FRAMES):
        # Run simulation steps for this frame (normal operation)
        for _ in range(steps_per_frame):
            model.step()
        
        # Collect frame data
        frame_data = converter.get_frame_data(frame_idx)
        all_frames_data.append(frame_data)
        
        if phase_frame % 30 == 0:
            print(f"   Frame {frame_idx}: {len(model.vehicles)} vehicles remaining")
        
        frame_idx += 1
    
    print(f"\n✅ Phase 2 complete: {len(model.vehicles)} vehicles remaining")
    
    # Combine all frame data
    combined_df = pd.concat(all_frames_data, ignore_index=True)
    
    print(f"\n📊 Simulation Summary:")
    print(f"   Total frames: {frame_idx}")
    print(f"   Total detections: {len(combined_df)}")
    print(f"   Unique vehicles: {combined_df['track_id'].nunique()}")
    
    return combined_df


if __name__ == "__main__":
    # Run simulation and collect data
    print("\n" + "=" * 60)
    df = run_jam_video_simulation()
    
    # Save to CSV
    print(f"\n💾 Saving simulation data to {OUTPUT_CSV}...")
    Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"   ✅ Saved!")
    
    # Create video
    print(f"\n🎬 Creating video: {OUTPUT_VIDEO}")
    viz = TrafficVisualizer(IMAGE_FILE, OUTPUT_CSV)
    viz.create_video(OUTPUT_VIDEO, fps=int(FPS), show_trails=True)

