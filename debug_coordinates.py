#!/usr/bin/env python3
"""
Debug script to check coordinate transformation
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.traffic_model import TrafficSimulationModel
from src.utils.simulation_to_csv import SimulationToCSVConverter


def debug_coordinates():
    """Debug coordinate transformation."""
    
    print("="*60)
    print("Coordinate Transformation Debug")
    print("="*60)
    
    # Create simulation model
    model = TrafficSimulationModel(
        width=3000,
        height=3000,
        time_step=0.1,
        vehicle_spawn_rate=0.1,
        max_vehicles=5
    )
    
    print(f"\nRoad network bounds:")
    min_x = min(lane.start_point.x for lane in model.road_network.all_lanes.values())
    max_x = max(lane.end_point.x for lane in model.road_network.all_lanes.values())
    min_y = min(lane.start_point.y for lane in model.road_network.all_lanes.values())
    max_y = max(lane.end_point.y for lane in model.road_network.all_lanes.values())
    print(f"  X: {min_x:.0f} to {max_x:.0f} meters")
    print(f"  Y: {min_y:.0f} to {max_y:.0f} meters")
    
    # Create converter with optimal scale
    converter = SimulationToCSVConverter(
        model=model,
        image_width=3840,
        image_height=2160,
        offset_x=2064.0,
        offset_y=526.0,
        scale=0.3507  # Optimal scale to fit roads in image
    )
    
    print(f"\nVehicle positions (world coordinates):")
    for vehicle in model.vehicles:
        world_x, world_y = vehicle.get_visual_position()
        print(f"  Vehicle {vehicle.unique_id}: ({world_x:.2f}, {world_y:.2f}) meters")
        
        # Transform to image coordinates
        from src.utils.bbox_utils import world_to_image_coordinates
        img_x, img_y = world_to_image_coordinates(
            world_x, world_y,
            converter.image_width, converter.image_height,
            converter.world_bounds,
            converter.offset_x, converter.offset_y,
            converter.scale
        )
        
        print(f"    -> Image: ({img_x:.2f}, {img_y:.2f}) pixels")
        
        # Check if in bounds
        in_bounds_x = 0 <= img_x <= converter.image_width
        in_bounds_y = 0 <= img_y <= converter.image_height
        print(f"    In bounds: X={in_bounds_x}, Y={in_bounds_y}")
    
    # Get frame data
    print(f"\nFrame data (first frame):")
    df = converter.get_frame_data(0)
    
    if len(df) > 0:
        print(f"  Vehicles in frame: {len(df)}")
        print(f"\n  First vehicle:")
        row = df.iloc[0]
        print(f"    center_x: {row['center_x']:.2f}")
        print(f"    center_y: {row['center_y']:.2f}")
        print(f"    x1, y1: ({row['x1']:.2f}, {row['y1']:.2f})")
        print(f"    x2, y2: ({row['x2']:.2f}, {row['y2']:.2f})")
        print(f"    x3, y3: ({row['x3']:.2f}, {row['y3']:.2f})")
        print(f"    x4, y4: ({row['x4']:.2f}, {row['y4']:.2f})")
        
        # Check bounds
        all_x = [row['center_x'], row['x1'], row['x2'], row['x3'], row['x4']]
        all_y = [row['center_y'], row['y1'], row['y2'], row['y3'], row['y4']]
        min_x_coord = min(all_x)
        max_x_coord = max(all_x)
        min_y_coord = min(all_y)
        max_y_coord = max(all_y)
        
        print(f"\n  Coordinate ranges:")
        print(f"    X: {min_x_coord:.2f} to {max_x_coord:.2f} (image width: 0-{converter.image_width})")
        print(f"    Y: {min_y_coord:.2f} to {max_y_coord:.2f} (image height: 0-{converter.image_height})")
        
        in_bounds = (min_x_coord >= 0 and max_x_coord <= converter.image_width and
                     min_y_coord >= 0 and max_y_coord <= converter.image_height)
        print(f"    All in bounds: {in_bounds}")
    else:
        print("  No vehicles in frame!")


if __name__ == "__main__":
    debug_coordinates()

