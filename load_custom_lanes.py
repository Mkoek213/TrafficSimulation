#!/usr/bin/env python3
"""
Load manually marked lanes and create road network for simulation

This script loads lanes marked with mark_lanes.py and creates a road network
that the MESA simulation can use.
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.road_network import RoadNetwork, Road, Lane, Point, Intersection


def load_lanes_from_json(json_path: str) -> dict:
    """Load lane data from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def create_road_network_from_lanes(lane_data: dict) -> RoadNetwork:
    """
    Create a road network from manually marked lanes.
    
    Args:
        lane_data: Dictionary with lane data from JSON file
    
    Returns:
        RoadNetwork object
    """
    network = RoadNetwork()
    
    lanes_list = lane_data['lanes']
    image_width = lane_data['image_width']
    image_height = lane_data['image_height']
    
    # Transformation parameters (same as simulation)
    offset_x = 2064.0  # ROI center X
    offset_y = 526.0   # ROI center Y
    scale = 0.3507     # pixels per meter (for positioning)
    
    def image_to_world(img_x: float, img_y: float) -> Tuple[float, float]:
        """Convert image coordinates to world coordinates."""
        world_x = (img_x - offset_x) / scale
        world_y = (offset_y - img_y) / scale  # Flip Y-axis
        return world_x, world_y
    
    # Group lanes into roads (assume each lane is a separate road for now)
    # Or group by direction if you want
    
    road_id = 0
    
    for lane_info in lanes_list:
        lane_id = lane_info['lane_id']
        points = lane_info['points']
        
        if len(points) < 2:
            print(f"⚠ Skipping lane {lane_id}: needs at least 2 points")
            continue
        
        # Convert points to world coordinates
        world_points = []
        for px, py in points:
            wx, wy = image_to_world(px, py)
            world_points.append(Point(wx, wy))
        
        # Create lane from start to end
        start_point = world_points[0]
        end_point = world_points[-1]
        
        # Create a road for this lane
        road = Road(road_id, f"Lane_{lane_id}")
        
        # Create lane with polygon points to detect missing edge
        lane = Lane(
            lane_id=lane_id,
            start_point=start_point,
            end_point=end_point,
            speed_limit=30.0,
            lane_width=3.5,
            polygon_points=world_points  # Pass polygon points to detect missing edge
        )
        
        road.add_lane(lane)
        network.add_road(road)
        
        print(f"Created lane {lane_id}: ({start_point.x:.1f}, {start_point.y:.1f}) -> ({end_point.x:.1f}, {end_point.y:.1f})")
        
        road_id += 1
    
    return network


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load marked lanes and create road network"
    )
    
    parser.add_argument('--lanes', required=True, help='Path to lanes JSON file')
    parser.add_argument('--output', help='Output road network JSON (optional)')
    
    args = parser.parse_args()
    
    # Load lanes
    print(f"Loading lanes from {args.lanes}...")
    lane_data = load_lanes_from_json(args.lanes)
    
    print(f"Found {len(lane_data['lanes'])} lanes")
    
    # Create road network
    print("\nCreating road network...")
    network = create_road_network_from_lanes(lane_data)
    
    print(f"\n✅ Created road network with {len(network.all_lanes)} lanes")
    
    # Save if requested
    if args.output:
        # Could save network to JSON here if needed
        print(f"Road network ready for simulation")


if __name__ == "__main__":
    main()

