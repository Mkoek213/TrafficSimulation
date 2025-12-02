#!/usr/bin/env python3
"""
Run MESA Traffic Simulation and Generate Bounding Box CSV

This script runs a MESA traffic simulation using the Krauss model and outputs
bounding boxes in CSV format compatible with the visualization system.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.traffic_model import TrafficSimulationModel
from src.utils.simulation_to_csv import SimulationToCSVConverter


def main():
    parser = argparse.ArgumentParser(
        description="Run MESA traffic simulation and generate bounding box CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation with default settings
  python run_mesa_simulation.py --output mesa_simulation.csv --frames 300

  # Run simulation with custom image dimensions
  python run_mesa_simulation.py --output mesa_simulation.csv --frames 300 \\
      --image-width 1920 --image-height 1080

  # Run simulation with coordinate transformation
  python run_mesa_simulation.py --output mesa_simulation.csv --frames 300 \\
      --offset-x 0 --offset-y 0 --scale 0.5
        """
    )
    
    parser.add_argument('--output', required=True,
                       help='Output CSV file path')
    parser.add_argument('--frames', type=int, default=900,
                       help='Number of frames to simulate (default: 900 for 112.5 seconds at 8fps)')
    parser.add_argument('--fps', type=float, default=8.0,
                       help='Frames per second / detections per second (default: 8.0 for 8 detections/second)')
    parser.add_argument('--time-step', type=float, default=0.1,
                       help='Simulation time step in seconds (default: 0.1)')
    parser.add_argument('--max-vehicles', type=int, default=150,
                       help='Maximum number of vehicles (default: 150)')
    parser.add_argument('--spawn-rate', type=float, default=0.5,
                       help='Vehicle spawn rate (vehicles/second, default: 0.5 for more frequent spawning)')
    parser.add_argument('--image-width', type=int, default=3840,
                       help='Image width in pixels (default: 3840 for SiteA.jpg)')
    parser.add_argument('--image-height', type=int, default=2160,
                       help='Image height in pixels (default: 2160 for SiteA.jpg)')
    parser.add_argument('--offset-x', type=float, default=2064.0,
                       help='X offset (ROI center X) for coordinate transformation (default: 2064.0)')
    parser.add_argument('--offset-y', type=float, default=526.0,
                       help='Y offset (ROI center Y) for coordinate transformation (default: 526.0)')
    parser.add_argument('--scale', type=float, default=0.3507,
                       help='Scale factor (pixels per meter) for coordinate transformation (default: 0.3507, optimal for SiteA.jpg)')
    parser.add_argument('--custom-lanes', type=str, default=None,
                       help='Path to JSON file with custom marked lanes (from mark_lanes.py)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("MESA Traffic Simulation - Bounding Box Generator")
    print("="*60)
    print(f"Output: {args.output}")
    print(f"Frames: {args.frames}")
    print(f"FPS: {args.fps}")
    print(f"Max vehicles: {args.max_vehicles}")
    print(f"Image dimensions: {args.image_width}x{args.image_height}")
    print()
    
    # Create simulation model
    print("Creating simulation model...")
    model = TrafficSimulationModel(
        width=3000,  # World space width (not used directly, but needed)
        height=3000,  # World space height
        time_step=args.time_step,
        vehicle_spawn_rate=args.spawn_rate,
        max_vehicles=args.max_vehicles,
        custom_lanes_path=args.custom_lanes,
        model_boost=1
    )
    
    print(f"✓ Created road network with {len(model.road_network.all_lanes)} lanes")
    print(f"✓ Initial vehicles: {len(model.vehicles)}")
    print()
    
    # Create converter
    print("Creating CSV converter...")
    converter = SimulationToCSVConverter(
        model=model,
        image_width=args.image_width,
        image_height=args.image_height
    )
    
    print(f"✓ World bounds: {converter.world_bounds}")
    print()
    
    # Run simulation and save
    print("Running simulation...")
    df = converter.run_simulation_and_save(
        output_path=args.output,
        num_frames=args.frames,
        frames_per_second=args.fps
    )
    
    print()
    print("="*60)
    print("✅ Simulation complete!")
    print(f"   CSV file: {args.output}")
    print(f"   Total detections: {len(df)}")
    print(f"   Unique vehicles: {df['track_id'].nunique()}")
    print(f"   Frame range: {df['frame'].min()} to {df['frame'].max()}")
    print()
    print("Next steps:")
    print(f"   Use the visualization script to create video:")
    print(f"   python eda/vis/traffic_visualizer.py \\")
    print(f"       --image eda/data/media/SiteA.jpg \\")
    print(f"       --csv {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()

