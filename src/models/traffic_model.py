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
                 vehicle_spawn_rate: float = 0.1,  # vehicles per second (one every 10 seconds)
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
        
        # Spawn vehicles gradually over time
        self._spawn_initial_vehicles()
    
    def _create_test_road_network(self):
        """Create a simple test road network with an intersection."""
        from ..models.road_network import Road, Lane, Intersection
        
        # Create intersection at origin
        center_point = Point(0, 0)
        intersection = Intersection(0, center_point)
        
        # Define road parameters
        road_length = 1500.0  # 1.5km in each direction
        
        # Create 4 roads: East, North, West, South
        directions = [
            ('East', Point(road_length, 0)),      # East
            ('North', Point(0, road_length)),     # North
            ('West', Point(-road_length, 0)),     # West
            ('South', Point(0, -road_length))     # South
        ]
        
        incoming_lanes = []
        outgoing_lanes = []
        
        for idx, (direction_name, direction_offset) in enumerate(directions):
            # Calculate far point for this direction
            far_point = Point(
                center_point.x + direction_offset.x,
                center_point.y + direction_offset.y
            )
            
            # Create incoming road (toward center)
            incoming_road = Road(len(self.road_network.roads), f"{direction_name}_Incoming")
            incoming_lane = Lane(
                lane_id=len(self.road_network.all_lanes),
                start_point=far_point,
                end_point=center_point,
                speed_limit=30.0,
                lane_width=3.5
            )
            incoming_road.add_lane(incoming_lane)
            self.road_network.add_road(incoming_road)
            incoming_lanes.append(incoming_lane.lane_id)
            intersection.add_lane(incoming_lane.lane_id)
            
            # Create outgoing road (away from center)
            outgoing_road = Road(len(self.road_network.roads), f"{direction_name}_Outgoing")
            outgoing_lane = Lane(
                lane_id=len(self.road_network.all_lanes),
                start_point=center_point,
                end_point=far_point,
                speed_limit=30.0,
                lane_width=3.5
            )
            outgoing_road.add_lane(outgoing_lane)
            self.road_network.add_road(outgoing_road)
            outgoing_lanes.append(outgoing_lane.lane_id)
        
        # Connect incoming lanes to opposite outgoing lanes (straight through)
        for i, incoming_lane_id in enumerate(incoming_lanes):
            incoming_lane = self.road_network.get_lane(incoming_lane_id)
            # Opposite direction is (i + 2) % 4
            opposite_idx = (i + 2) % 4
            outgoing_lane_id = outgoing_lanes[opposite_idx]
            incoming_lane.add_connected_lane(outgoing_lane_id)
        
        # Add intersection to network
        self.road_network.add_intersection(intersection)
        
        print(f"Created road network with {len(self.road_network.all_lanes)} lanes")
        print(f"Number of roads: {len(self.road_network.roads)}")
        print(f"Number of intersections: {len(self.road_network.intersections)}")
        
        # Print lane information
        for lane_id, lane in self.road_network.all_lanes.items():
            connected = f", connected to: {lane.connected_lanes}" if lane.connected_lanes else ""
            print(f"Lane {lane_id}: {lane.start_point.x:.0f},{lane.start_point.y:.0f} -> {lane.end_point.x:.0f},{lane.end_point.y:.0f} (length: {lane.length:.0f}m){connected}")
    
    def _get_incoming_lanes(self) -> List[int]:
        """
        Get all incoming lanes (lanes that have connected lanes).
        These are typically the lanes that approach intersections.
        
        Returns:
            List of incoming lane IDs
        """
        incoming_lanes = []
        for lane_id, lane in self.road_network.all_lanes.items():
            # If a lane has connected lanes, it's an incoming lane
            if lane.connected_lanes:
                incoming_lanes.append(lane_id)
        return incoming_lanes
    
    def _spawn_initial_vehicles(self):
        """Spawn initial vehicles on the road network."""
        # Only spawn one vehicle initially to avoid clustering
        if self.max_vehicles > 0:
            self._spawn_vehicle()
    
    def _spawn_vehicle(self):
        """Spawn a new vehicle with proper spacing."""
        if len(self.vehicles) >= self.max_vehicles:
            return
        
        # Get all incoming lanes (lanes that lead to the intersection)
        incoming_lanes = self._get_incoming_lanes()
        if not incoming_lanes:
            # Fallback to first lane if no incoming lanes found
            lane_id = list(self.road_network.all_lanes.keys())[0]
        else:
            # Randomly choose an incoming lane
            lane_id = random.choice(incoming_lanes)
        
        # Create vehicle
        vehicle_id = self.vehicle_counter
        self.vehicle_counter += 1
        
        # Random vehicle properties
        max_speed = random.uniform(30.0, 40.0)  # m/s (increased speeds)
        max_acceleration = random.uniform(2.0, 3.0)  # m/s² (faster acceleration)
        max_deceleration = random.uniform(-4.0, -5.0)  # m/s² (stronger braking)
        
        # Random color
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
        
        # Calculate safe starting position to avoid collisions
        safe_start_position = self._calculate_safe_start_position(lane_id)
        
        vehicle = Vehicle(
            model=self,
            unique_id=vehicle_id,
            lane_id=lane_id,
            position=safe_start_position,
            speed=random.uniform(15, 25),  # Higher initial speed to clear space faster
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
        
        print(f"Spawned vehicle {vehicle_id} at position {safe_start_position:.1f}m")
    
    def _calculate_safe_start_position(self, lane_id: int) -> float:
        """
        Calculate a safe starting position for a new vehicle at the start of the road.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Safe starting position along the lane (at the start)
        """
        # Get existing vehicles in this lane
        existing_vehicles = self.get_vehicles_in_lane(lane_id)
        
        if not existing_vehicles:
            # No vehicles in lane, start at beginning
            return random.uniform(0, 20)
        
        # Sort vehicles by position
        existing_vehicles.sort(key=lambda v: v.position)
        
        # Find vehicles near the start of the road (within first 100m)
        start_vehicles = [v for v in existing_vehicles if v.position < 100.0]
        
        if not start_vehicles:
            # No vehicles near start, spawn at beginning
            return random.uniform(0, 20)
        
        # Find the vehicle closest to the start
        closest_vehicle = min(start_vehicles, key=lambda v: v.position)
        
        # Calculate safe distance behind the closest vehicle
        min_safe_distance = 80.0  # meters - increased minimum safe distance
        safe_position = closest_vehicle.position - min_safe_distance - closest_vehicle.length
        
        # Ensure we don't go negative (stay at start of road)
        safe_position = max(0, safe_position)
        
        # Add some randomness to avoid perfect spacing
        safe_position += random.uniform(0, 10)
        
        # Ensure we don't go negative
        safe_position = max(0, safe_position)
        
        return safe_position
    
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
        
        # Spawn new vehicles if needed (very slowly)
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
        
        # Spawn initial vehicles gradually
        self._spawn_initial_vehicles()
        
        print("Simulation reset")
