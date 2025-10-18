"""
Traffic Simulation Model

This module contains the main MESA Model class that manages the traffic simulation.
It coordinates vehicles, road networks, and simulation state.
"""

import numpy as np
import mesa
from mesa.space import AgentSet, ContinuousSpace
from typing import List, Dict, Optional
import random

from ..agents.vehicle import Vehicle
from ..models.road_network import RoadNetwork, Point
from ..utils.krauss_model import KraussModel


class TrafficSimulationModel(mesa.Model):
    """
    Main traffic simulation model using MESA framework.
    
    This model manages:
    - Vehicle agents
    - Road network
    - Simulation time and state
    - Vehicle spawning and removal
    - Statistics collection
    """
    
    def __init__(self, 
                 width: int = 1000,
                 height: int = 1000,
                 time_step: float = 0.1,  # seconds
                 vehicle_spawn_rate: float = 0.1,  # vehicles per second
                 max_vehicles: int = 100):
        """
        Initialize the traffic simulation model.
        
        Args:
            width: Width of the simulation space (pixels)
            height: Height of the simulation space (pixels)
            time_step: Time step for simulation (seconds)
            vehicle_spawn_rate: Rate of vehicle spawning (vehicles/second)
            max_vehicles: Maximum number of vehicles in simulation
        """
        super().__init__()
        
        # Simulation parameters
        self.width = width
        self.height = height
        self.time_step = time_step
        self.vehicle_spawn_rate = vehicle_spawn_rate
        self.max_vehicles = max_vehicles
        
        # MESA components
        self.vehicle_agents = AgentSet([])
        self.space = ContinuousSpace(width, height, False)
        
        # Road network
        self.road_network = RoadNetwork()
        
        # Vehicle management
        self.vehicles: List[Vehicle] = []
        self.vehicle_counter = 0
        self.spawn_timer = 0.0
        self.current_time = 0.0
        
        # Statistics
        self.stats = {
            'total_vehicles_spawned': 0,
            'total_vehicles_removed': 0,
            'average_speed': 0.0,
            'total_distance_traveled': 0.0,
            'collisions': 0
        }
        
        # Initialize road network
        self._create_test_road_network()
        
        # Spawn initial vehicles
        self._spawn_initial_vehicles()
    
    def _create_test_road_network(self):
        """Create a simple test road network."""
        # Create a simple highway (centered)
        start_point = Point(-400, 0)
        end_point = Point(400, 0)
        
        highway = self.road_network.create_simple_highway(
            start_point, end_point, num_lanes=3
        )
        
        # Create a perpendicular road (centered)
        start_point2 = Point(0, -300)
        end_point2 = Point(0, 300)
        
        highway2 = self.road_network.create_simple_highway(
            start_point2, end_point2, num_lanes=2
        )
        
        print(f"Created road network with {len(self.road_network.all_lanes)} lanes")
    
    def _spawn_initial_vehicles(self):
        """Spawn initial vehicles on the road network."""
        num_initial_vehicles = min(10, self.max_vehicles)
        
        for i in range(num_initial_vehicles):
            self._spawn_vehicle()
    
    def _spawn_vehicle(self):
        """Spawn a new vehicle on a random lane."""
        if len(self.vehicles) >= self.max_vehicles:
            return
        
        # Get available lanes
        available_lanes = list(self.road_network.all_lanes.keys())
        if not available_lanes:
            return
        
        # Select random lane
        lane_id = random.choice(available_lanes)
        
        # Create vehicle
        vehicle_id = self.vehicle_counter
        self.vehicle_counter += 1
        
        # Random vehicle properties
        max_speed = random.uniform(25.0, 35.0)  # m/s
        max_acceleration = random.uniform(1.5, 2.5)  # m/s²
        max_deceleration = random.uniform(-3.5, -4.5)  # m/s²
        
        # Random color
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        vehicle = Vehicle(
            model=self,
            unique_id=vehicle_id,
            lane_id=lane_id,
            position=random.uniform(0, 50),  # Start near beginning of lane
            speed=random.uniform(5, 15),  # Initial speed
            max_speed=max_speed,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration,
            color=color
        )
        
        # Add to simulation
        self.vehicles.append(vehicle)
        self.vehicle_agents.add(vehicle)
        
        # Update statistics
        self.stats['total_vehicles_spawned'] += 1
        
        print(f"Spawned vehicle {vehicle_id} on lane {lane_id}")
    
    def remove_vehicle(self, vehicle: Vehicle):
        """
        Remove a vehicle from the simulation.
        
        Args:
            vehicle: Vehicle to remove
        """
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
            self.vehicle_agents.remove(vehicle)
            self.stats['total_vehicles_removed'] += 1
            print(f"Removed vehicle {vehicle.unique_id}")
    
    def get_vehicles_in_lane(self, lane_id: int) -> List[Vehicle]:
        """
        Get all vehicles in a specific lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            List of vehicles in the lane
        """
        return [v for v in self.vehicles if v.lane_id == lane_id]
    
    def get_adjacent_lane(self, lane_id: int, direction: str) -> Optional[int]:
        """
        Get the adjacent lane ID in the specified direction.
        
        Args:
            lane_id: ID of the current lane
            direction: 'left' or 'right'
            
        Returns:
            Adjacent lane ID or None if not found
        """
        return self.road_network.get_adjacent_lane(lane_id, direction)
    
    def get_lane_length(self, lane_id: int) -> float:
        """
        Get the length of a lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Length of the lane in meters
        """
        return self.road_network.get_lane_length(lane_id)
    
    def get_lane_position(self, lane_id: int) -> tuple:
        """
        Get the starting position of a lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Tuple of (x, y) coordinates
        """
        return self.road_network.get_lane_position(lane_id)
    
    def get_lane_direction(self, lane_id: int) -> tuple:
        """
        Get the direction vector of a lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Tuple of (dx, dy) direction vector
        """
        return self.road_network.get_lane_direction(lane_id)
    
    def update_statistics(self):
        """Update simulation statistics."""
        if self.vehicles:
            # Calculate average speed
            total_speed = sum(v.speed for v in self.vehicles)
            self.stats['average_speed'] = total_speed / len(self.vehicles)
            
            # Calculate total distance traveled
            total_distance = sum(v.position for v in self.vehicles)
            self.stats['total_distance_traveled'] = total_distance
        else:
            self.stats['average_speed'] = 0.0
            self.stats['total_distance_traveled'] = 0.0
    
    def step(self):
        """Execute one step of the simulation."""
        # Update time
        self.current_time += self.time_step
        
        # Update spawn timer
        self.spawn_timer += self.time_step
        
        # Spawn new vehicles if needed
        if (self.spawn_timer >= 1.0 / self.vehicle_spawn_rate and 
            len(self.vehicles) < self.max_vehicles):
            self._spawn_vehicle()
            self.spawn_timer = 0.0
        
        # Update all agents
        self.vehicle_agents.do("step")
        
        # Update statistics
        self.update_statistics()
        
        # Update intersections (traffic lights)
        for intersection in self.road_network.intersections:
            intersection.update_traffic_lights(self.time_step)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the current model state.
        
        Returns:
            Dictionary with model information
        """
        return {
            'current_time': self.current_time,
            'num_vehicles': len(self.vehicles),
            'num_lanes': len(self.road_network.all_lanes),
            'num_roads': len(self.road_network.roads),
            'num_intersections': len(self.road_network.intersections),
            'stats': self.stats.copy()
        }
    
    def reset(self):
        """Reset the simulation to initial state."""
        # Clear all vehicles
        for vehicle in self.vehicles.copy():
            self.remove_vehicle(vehicle)
        
        # Reset counters
        self.vehicle_counter = 0
        self.spawn_timer = 0.0
        self.current_time = 0.0
        
        # Reset statistics
        self.stats = {
            'total_vehicles_spawned': 0,
            'total_vehicles_removed': 0,
            'average_speed': 0.0,
            'total_distance_traveled': 0.0,
            'collisions': 0
        }
        
        # Spawn initial vehicles
        self._spawn_initial_vehicles()
        
        print("Simulation reset")
