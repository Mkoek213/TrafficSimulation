"""
Vehicle Agent Implementation

This module contains the Vehicle agent class that represents individual vehicles
in the traffic simulation. Each vehicle uses the Krauss car-following model
and can perform lane-changing maneuvers.
"""

import numpy as np
import mesa
from typing import Optional, List, Tuple, TYPE_CHECKING
from ..utils.krauss_model import KraussModel

if TYPE_CHECKING:
    from ..models.traffic_model import TrafficSimulationModel


class Vehicle(mesa.Agent):
    """
    A vehicle agent that moves along roads using the Krauss car-following model.
    
    Each vehicle has:
    - Position and velocity
    - Lane-changing capabilities
    - Car-following behavior using Krauss model
    - Unique characteristics (max speed, acceleration, etc.)
    """
    
    def __init__(self, 
                 model: 'TrafficSimulationModel',
                 unique_id: int,
                 lane_id: int,
                 position: float = 0.0,
                 speed: float = 0.0,
                 max_speed: float = 30.0,
                 max_acceleration: float = 2.0,
                 max_deceleration: float = -4.0,
                 length: float = 4.5,  # meters
                 width: float = 2.0,   # meters
                 color: Tuple[int, int, int] = (255, 0, 0)):  # Red by default
        """
        Initialize a vehicle agent.
        
        Args:
            unique_id: Unique identifier for the agent
            model: Reference to the simulation model
            lane_id: ID of the lane this vehicle is currently in
            position: Position along the lane (meters)
            speed: Current speed (m/s)
            max_speed: Maximum speed (m/s)
            max_acceleration: Maximum acceleration (m/s²)
            max_deceleration: Maximum deceleration (m/s²)
            length: Vehicle length (meters)
            width: Vehicle width (meters)
            color: RGB color tuple for visualization
        """
        super().__init__(model)
        
        # Physical properties
        self.length = length
        self.width = width
        self.color = color
        
        # Position and movement
        self.lane_id = lane_id
        self.position = position
        self.speed = speed
        self.target_speed = max_speed
        
        # Lane-changing properties
        self.desired_lane_change = None  # 'left', 'right', or None
        self.lane_change_timer = 0
        self.lane_change_duration = 2  # seconds
        
        # Car-following model
        self.krauss_model = KraussModel(
            max_speed=max_speed,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration
        )
        
        # State tracking
        self.is_changing_lanes = False
        self.lane_change_progress = 0.0  # 0.0 to 1.0
        
    def get_leader(self) -> Optional['Vehicle']:
        """
        Find the leading vehicle in the same lane.
        
        Returns:
            Leading vehicle or None if no leader found
        """
        lane_vehicles = self.model.get_vehicles_in_lane(self.lane_id)
        
        # Sort vehicles by position
        lane_vehicles.sort(key=lambda v: v.position)
        
        # Find this vehicle's index
        try:
            current_index = lane_vehicles.index(self)
        except ValueError:
            return None
        
        # Return the next vehicle (if any)
        if current_index + 1 < len(lane_vehicles):
            return lane_vehicles[current_index + 1]
        
        return None
    
    def get_follower(self) -> Optional['Vehicle']:
        """
        Find the following vehicle in the same lane.
        
        Returns:
            Following vehicle or None if no follower found
        """
        lane_vehicles = self.model.get_vehicles_in_lane(self.lane_id)
        
        # Sort vehicles by position
        lane_vehicles.sort(key=lambda v: v.position)
        
        # Find this vehicle's index
        try:
            current_index = lane_vehicles.index(self)
        except ValueError:
            return None
        
        # Return the previous vehicle (if any)
        if current_index > 0:
            return lane_vehicles[current_index - 1]
        
        return None
    
    def calculate_distance_to_leader(self) -> float:
        """
        Calculate distance to the leading vehicle.
        
        Returns:
            Distance to leader in meters, or float('inf') if no leader
        """
        leader = self.get_leader()
        if leader is None:
            return float('inf')
        
        # Distance is the gap between vehicles plus leader's length
        distance = leader.position - self.position - leader.length
        
        # Don't subtract safety margin here - it causes false collision detection
        # The safety margin is handled in the Krauss model instead
        return max(0.0, distance)
    
    def can_change_lane(self, direction: str) -> bool:
        """
        Check if the vehicle can safely change lanes in the given direction.
        
        Args:
            direction: 'left' or 'right'
            
        Returns:
            True if lane change is safe, False otherwise
        """
        if self.is_changing_lanes:
            return False
        
        # Get target lane
        target_lane_id = self.model.get_adjacent_lane(self.lane_id, direction)
        if target_lane_id is None:
            return False
        
        # Check for vehicles in target lane
        target_lane_vehicles = self.model.get_vehicles_in_lane(target_lane_id)
        
        # Check if there's enough space in target lane
        safe_distance = 20.0  # meters
        
        for vehicle in target_lane_vehicles:
            distance = abs(vehicle.position - self.position)
            if distance < safe_distance:
                return False
        
        return True
    
    def start_lane_change(self, direction: str):
        """
        Start a lane change maneuver.
        
        Args:
            direction: 'left' or 'right'
        """
        if self.can_change_lane(direction):
            self.desired_lane_change = direction
            self.is_changing_lanes = True
            self.lane_change_progress = 0.0
            self.lane_change_timer = 0
    
    def update_lane_change(self, dt: float):
        """
        Update lane change progress.
        
        Args:
            dt: Time step (seconds)
        """
        if not self.is_changing_lanes:
            return
        
        self.lane_change_timer += dt
        
        # Update progress
        self.lane_change_progress = min(1.0, self.lane_change_timer / self.lane_change_duration)
        
        # Complete lane change if finished
        if self.lane_change_progress >= 1.0:
            self.complete_lane_change()
    
    def complete_lane_change(self):
        """
        Complete the lane change maneuver.
        """
        if self.desired_lane_change:
            target_lane_id = self.model.get_adjacent_lane(self.lane_id, self.desired_lane_change)
            if target_lane_id is not None:
                self.lane_id = target_lane_id
        
        self.is_changing_lanes = False
        self.desired_lane_change = None
        self.lane_change_progress = 0.0
        self.lane_change_timer = 0
    
    def step(self):
        """
        Update the vehicle's state for one time step.
        """
        dt = self.model.time_step
        
        # Update lane change if in progress
        if self.is_changing_lanes:
            self.update_lane_change(dt)
        
        # Calculate distance to leader
        distance_to_leader = self.calculate_distance_to_leader()
        
        # Check for collision with leader (only if very close)
        if distance_to_leader < 3.0:  # Stop if within 3 meters (reduced from 10m)
            print(f"COLLISION DETECTED! Vehicle {self.unique_id} too close to leader")
            # Emergency stop
            self.speed = 0
            return
        
        # Get leader speed
        leader_speed = None
        leader = self.get_leader()
        if leader is not None:
            leader_speed = leader.speed
        
        # Calculate next speed using Krauss model
        next_speed = self.krauss_model.calculate_next_speed(
            self.speed,
            distance_to_leader,
            leader_speed
        )
        
        # Update speed
        self.speed = next_speed
        
        # Update position
        next_position = self.krauss_model.calculate_position_update(
            self.position,
            self.speed,
            dt
        )
        
        # Check for collision after position update
        if self._check_collision_after_move(next_position):
            print(f"COLLISION AVOIDED! Vehicle {self.unique_id} stopping")
            self.speed = 0
            return
        
        # Check if vehicle has reached end of lane
        lane_length = self.model.get_lane_length(self.lane_id)
        if next_position >= lane_length:
            # Vehicle has reached end of lane - try to transition to connected lane
            if not self._try_lane_transition():
                # No connected lane available, remove vehicle
                self.model.remove_vehicle(self)
        else:
            self.position = next_position
    
    def _try_lane_transition(self) -> bool:
        """
        Try to transition to a connected lane at the end of current lane.
        
        Returns:
            True if transition successful, False otherwise
        """
        current_lane = self.model.road_network.get_lane(self.lane_id)
        if not current_lane or not current_lane.connected_lanes:
            return False
        
        # Choose first connected lane (straight through for now)
        # In the future, this could support turns based on routing
        next_lane_id = current_lane.connected_lanes[0]
        
        # Check if the next lane has space at the beginning
        next_lane_vehicles = self.model.get_vehicles_in_lane(next_lane_id)
        
        # Check for vehicles blocking the entry to next lane
        min_start_distance = 20.0  # Need at least 20m clearance
        for vehicle in next_lane_vehicles:
            if vehicle.position < min_start_distance:
                # Too crowded at the start of next lane
                # Stop at end of current lane and wait
                self.speed = 0
                self.position = self.model.get_lane_length(self.lane_id)
                return False
        
        # Transition to next lane
        old_lane_id = self.lane_id
        self.lane_id = next_lane_id
        self.position = 0.0  # Start at beginning of new lane
        print(f"Vehicle {self.unique_id} transitioned from lane {old_lane_id} to lane {next_lane_id}")
        return True
    
    def _check_collision_after_move(self, new_position: float) -> bool:
        """
        Check if moving to new_position would cause a collision.
        
        Args:
            new_position: Proposed new position
            
        Returns:
            True if collision would occur, False otherwise
        """
        # Get all vehicles in the same lane
        lane_vehicles = self.model.get_vehicles_in_lane(self.lane_id)
        
        for vehicle in lane_vehicles:
            if vehicle == self:
                continue
            
            # Calculate distance to this vehicle
            distance = abs(vehicle.position - new_position)
            
            # Check if too close (considering vehicle lengths)
            min_distance = (self.length + vehicle.length) / 2 + 5.0  # 5m safety margin (reduced from 15m)
            
            if distance < min_distance:
                return True
        
        return False
    
    def get_visual_position(self) -> Tuple[float, float]:
        """
        Get the visual position for rendering.
        
        Returns:
            Tuple of (x, y) coordinates for visualization
        """
        # Get the lane and use its method to calculate position
        lane = self.model.road_network.get_lane(self.lane_id)
        if not lane:
            return (0.0, 0.0)
        
        # Use the lane's built-in method for accurate position calculation
        point = lane.get_position_at_distance(self.position)
        
        return point.x, point.y
    
    def get_visual_angle(self) -> float:
        """
        Get the visual angle for rendering.
        
        Returns:
            Angle in radians
        """
        lane_direction = self.model.get_lane_direction(self.lane_id)
        return np.arctan2(lane_direction[1], lane_direction[0])
