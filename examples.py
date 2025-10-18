#!/usr/bin/env python3
"""
Example usage of the traffic simulation.

This script demonstrates how to use the traffic simulation
programmatically without the command-line interface.
"""

import sys
import os
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.traffic_model import TrafficSimulationModel
from src.models.road_network import RoadNetwork, Point
from src.utils.osm_loader import OSMNetworkLoader


def example_basic_simulation():
    """Example of running a basic simulation."""
    print("Basic Simulation Example")
    print("-" * 30)
    
    # Create simulation model
    model = TrafficSimulationModel(
        width=800,
        height=600,
        time_step=0.1,
        vehicle_spawn_rate=0.2,
        max_vehicles=20
    )
    
    # Run simulation for 10 seconds
    steps = 100  # 10 seconds at 0.1s time step
    for i in range(steps):
        model.step()
        
        # Print progress every 20 steps
        if i % 20 == 0:
            info = model.get_model_info()
            print(f"Step {i}: {info['num_vehicles']} vehicles, "
                  f"avg speed: {info['stats']['average_speed']:.1f} m/s")
    
    # Print final statistics
    final_info = model.get_model_info()
    print(f"\nFinal Statistics:")
    print(f"Total vehicles spawned: {final_info['stats']['total_vehicles_spawned']}")
    print(f"Total vehicles removed: {final_info['stats']['total_vehicles_removed']}")
    print(f"Average speed: {final_info['stats']['average_speed']:.1f} m/s")


def example_custom_road_network():
    """Example of creating a custom road network."""
    print("\nCustom Road Network Example")
    print("-" * 30)
    
    # Create a custom road network
    network = RoadNetwork()
    
    # Create a T-intersection
    # Main road (horizontal)
    main_start = Point(0, 0)
    main_end = Point(200, 0)
    main_road = network.create_simple_highway(main_start, main_end, num_lanes=2)
    
    # Side road (vertical)
    side_start = Point(100, 0)
    side_end = Point(100, 100)
    side_road = network.create_simple_highway(side_start, side_end, num_lanes=1)
    
    print(f"Created network with {len(network.all_lanes)} lanes")
    
    # Print lane information
    for lane_id, lane in network.all_lanes.items():
        print(f"Lane {lane_id}: {lane.length:.1f}m, "
              f"speed limit: {lane.speed_limit:.1f} m/s")


def example_osm_network():
    """Example of loading OSM network (if available)."""
    print("\nOpenStreetMap Network Example")
    print("-" * 30)
    
    try:
        # Try to load a small OSM network
        loader = OSMNetworkLoader()
        
        # Load a small area (this might take a while)
        print("Loading OSM network for a small area...")
        network = loader.load_network_from_bbox(
            north=40.7589,  # Small area in Manhattan
            south=40.7580,
            east=-73.9851,
            west=-73.9860
        )
        
        print(f"Loaded OSM network with {len(network.all_lanes)} lanes")
        
        # Print some statistics
        total_length = sum(lane.length for lane in network.all_lanes.values())
        print(f"Total road length: {total_length:.1f} meters")
        
    except Exception as e:
        print(f"OSM loading failed (this is expected if OSM data is not available): {e}")
        print("This is normal - OSM loading requires internet connection and proper setup.")


def example_vehicle_analysis():
    """Example of analyzing vehicle behavior."""
    print("\nVehicle Behavior Analysis Example")
    print("-" * 30)
    
    # Create simulation
    model = TrafficSimulationModel(max_vehicles=10)
    
    # Run simulation and collect data
    vehicle_data = []
    
    for step in range(50):
        model.step()
        
        # Collect data from each vehicle
        for vehicle in model.vehicles:
            vehicle_data.append({
                'step': step,
                'vehicle_id': vehicle.unique_id,
                'lane_id': vehicle.lane_id,
                'position': vehicle.position,
                'speed': vehicle.speed,
                'is_changing_lanes': vehicle.is_changing_lanes
            })
    
    # Analyze the data
    if vehicle_data:
        # Find vehicles that changed lanes
        lane_changes = [v for v in vehicle_data if v['is_changing_lanes']]
        print(f"Lane changes observed: {len(lane_changes)}")
        
        # Calculate average speeds
        speeds = [v['speed'] for v in vehicle_data]
        avg_speed = sum(speeds) / len(speeds)
        print(f"Average vehicle speed: {avg_speed:.1f} m/s")
        
        # Find fastest vehicle
        max_speed = max(speeds)
        print(f"Maximum vehicle speed: {max_speed:.1f} m/s")


def main():
    """Run all examples."""
    print("Traffic Simulation Examples")
    print("=" * 40)
    
    try:
        example_basic_simulation()
        example_custom_road_network()
        example_osm_network()
        example_vehicle_analysis()
        
        print("\n" + "=" * 40)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nExample failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
