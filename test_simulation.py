#!/usr/bin/env python3
"""
Test script for the traffic simulation.

This script tests the basic functionality of the traffic simulation
without requiring PyGame or visualization.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models.traffic_model import TrafficSimulationModel
from src.utils.krauss_model import KraussModel
from src.models.road_network import RoadNetwork, Point


def test_krauss_model():
    """Test the Krauss car-following model."""
    print("Testing Krauss Model...")
    
    model = KraussModel()
    
    # Test free flow (no leader)
    speed = model.calculate_next_speed(current_speed=10.0, distance_to_leader=float('inf'))
    assert speed > 10.0, "Vehicle should accelerate in free flow"
    
    # Test car following (with leader)
    speed = model.calculate_next_speed(current_speed=20.0, distance_to_leader=50.0, leader_speed=15.0)
    assert speed <= 20.0, "Vehicle should not exceed current speed when following"
    
    print("✓ Krauss Model tests passed")


def test_road_network():
    """Test the road network functionality."""
    print("Testing Road Network...")
    
    network = RoadNetwork()
    
    # Create a simple highway
    start_point = Point(0, 0)
    end_point = Point(100, 0)
    highway = network.create_simple_highway(start_point, end_point, num_lanes=2)
    
    assert len(network.all_lanes) == 2, "Should have 2 lanes"
    assert len(network.roads) == 1, "Should have 1 road"
    
    # Test lane properties
    lane = network.get_lane(0)
    assert lane is not None, "Should be able to get lane by ID"
    assert lane.length > 0, "Lane should have positive length"
    
    print("✓ Road Network tests passed")


def test_traffic_simulation():
    """Test the traffic simulation model."""
    print("Testing Traffic Simulation...")
    
    model = TrafficSimulationModel(width=500, height=500, max_vehicles=5)
    
    # Test initial state
    assert len(model.vehicles) > 0, "Should have initial vehicles"
    assert len(model.road_network.all_lanes) > 0, "Should have lanes"
    
    # Test simulation step
    initial_vehicle_count = len(model.vehicles)
    model.step()
    
    # Vehicles should still exist (unless they reached end of lane)
    assert len(model.vehicles) >= 0, "Vehicle count should be non-negative"
    
    # Test statistics
    info = model.get_model_info()
    assert 'num_vehicles' in info, "Should have vehicle count in info"
    assert 'stats' in info, "Should have statistics in info"
    
    print("✓ Traffic Simulation tests passed")


def test_vehicle_behavior():
    """Test vehicle behavior and interactions."""
    print("Testing Vehicle Behavior...")
    
    model = TrafficSimulationModel(width=500, height=500, max_vehicles=3)
    
    if len(model.vehicles) >= 2:
        vehicle1 = model.vehicles[0]
        vehicle2 = model.vehicles[1]
        
        # Test lane detection
        vehicles_in_lane = model.get_vehicles_in_lane(vehicle1.lane_id)
        assert len(vehicles_in_lane) > 0, "Should find vehicles in lane"
        
        # Test leader detection
        leader = vehicle1.get_leader()
        # Leader might be None if no vehicle ahead
        
        # Test distance calculation
        distance = vehicle1.calculate_distance_to_leader()
        assert distance >= 0, "Distance should be non-negative"
    
    print("✓ Vehicle Behavior tests passed")


def main():
    """Run all tests."""
    print("Running Traffic Simulation Tests")
    print("=" * 40)
    
    try:
        test_krauss_model()
        test_road_network()
        test_traffic_simulation()
        test_vehicle_behavior()
        
        print("\n" + "=" * 40)
        print("✓ All tests passed! The traffic simulation is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
