#!/usr/bin/env python3
"""
Example: Run MESA Simulation and Visualize

This example demonstrates how to:
1. Run a MESA traffic simulation
2. Generate bounding box CSV
3. Visualize using the existing visualization tools
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.traffic_model import TrafficSimulationModel
from src.utils.simulation_to_csv import SimulationToCSVConverter


def main():
    print("="*60)
    print("MESA Traffic Simulation Example")
    print("="*60)
    
    # Create simulation model
    print("\n1. Creating simulation model...")
    model = TrafficSimulationModel(
        width=3000,
        height=3000,
        time_step=0.1,
        vehicle_spawn_rate=0.1,  # 1 vehicle every 10 seconds
        max_vehicles=30
    )
    
    print(f"   ✓ Road network: {len(model.road_network.all_lanes)} lanes")
    print(f"   ✓ Initial vehicles: {len(model.vehicles)}")
    
    # Create converter
    print("\n2. Setting up CSV converter...")
    # Calibrated parameters from ROI analysis:
    # - Image size: 3840x2160 (SiteA.jpg)
    # - ROI center: (2064, 526) - where world origin (0,0) maps to
    # - Scale: 0.3507 pixels per meter (optimal to fit all roads within image bounds)
    #   This ensures vehicles at ±1500m stay within image bounds
    converter = SimulationToCSVConverter(
        model=model,
        image_width=3840,  # SiteA.jpg width
        image_height=2160,  # SiteA.jpg height
        offset_x=2064.0,  # ROI center X
        offset_y=526.0,   # ROI center Y
        scale=0.3507      # pixels per meter (for positioning, keeps vehicles in bounds)
    )
    
    print(f"   ✓ World bounds: {converter.world_bounds}")
    
    # Run simulation
    print("\n3. Running simulation...")
    output_path = "eda/data/mesa_simulation.csv"
    df = converter.run_simulation_and_save(
        output_path=output_path,
        num_frames=300,  # 10 seconds at 30 fps
        frames_per_second=30.0
    )
    
    print("\n" + "="*60)
    print("✅ Simulation complete!")
    print(f"\nOutput CSV: {output_path}")
    print(f"Total detections: {len(df)}")
    print(f"Unique vehicles: {df['track_id'].nunique()}")
    print(f"Frame range: {df['frame'].min()} to {df['frame'].max()}")
    print("\nTo visualize, run:")
    print(f"  python eda/vis/traffic_visualizer.py \\")
    print(f"      --image eda/data/media/SiteA.jpg \\")
    print(f"      --csv {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

