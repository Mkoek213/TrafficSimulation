"""
Vehicle Agent Implementation

This module contains the Vehicle agent class that represents individual vehicles
in the traffic simulation. Each vehicle uses the Krauss car-following model
to follow pre-defined lane centerlines.
"""

import numpy as np
import mesa
import random
from typing import Optional, List, Tuple, TYPE_CHECKING, cast
from ..utils.krauss_model import KraussModel
from ..models.road_network import Point

if TYPE_CHECKING:
    from ..models.traffic_model import TrafficSimulationModel
else:
    TrafficSimulationModel = None


class Vehicle(mesa.Agent):
    model: 'TrafficSimulationModel'
    """
    A vehicle agent that moves along roads using the Krauss car-following model.
    
    Each vehicle has:
    - Position and velocity
    - Lane-changing capabilities
    - Car-following behavior using Krauss model
    - Unique characteristics (max speed, acceleration, etc.)
    """
    
    def __init__(self, 
                 model: TrafficSimulationModel,
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
        self.model = cast('TrafficSimulationModel', model)
        
        # Physical properties
        self.length = length
        self.width = width
        self.color = color
        
        # Position and movement
        self.lane_id = lane_id
        self.position = position
        self.speed = speed
        self.target_speed = max_speed
        
        # Car-following model
        self.krauss_model = KraussModel(
            max_speed=max_speed,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration
        )
        
        # Route planning
        self.route_type = random.choice(['straight', 'left', 'right'])  # Desired route at intersection
        
    def get_leader(self) -> Optional['Vehicle']:
        """
        Find the leading vehicle in the same lane.
        
        Returns:
            Leading vehicle or None if no leader found
        """
        lane_vehicles = self.model.get_vehicles_in_lane(self.lane_id)
        
        if len(lane_vehicles) <= 1:
            return None
        
        # Sort vehicles by position
        lane_vehicles.sort(key=lambda v: v.position)
        
        # Find this vehicle's index
        try:
            current_index = lane_vehicles.index(self)
        except ValueError:
            return None
        
        # Return the next vehicle ahead (if any)
        if current_index + 1 < len(lane_vehicles):
            leader = lane_vehicles[current_index + 1]
            # Only return leader if they're actually ahead (considering wrap-around)
            lane_length = self.model.get_lane_length(self.lane_id)
            if leader.position > self.position:
                # Leader is ahead in normal order
                return leader
            elif leader.position < self.position - lane_length * 0.5:
                # Leader wrapped around and is ahead
                return leader
        
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
        
        # Distance is the gap between vehicles
        # Account for vehicle positions along the lane
        if leader.position > self.position:
            # Leader is ahead
            distance = leader.position - self.position - leader.length
        else:
            # Leader wrapped around (behind us in position but ahead on lane)
            lane_length = self.model.get_lane_length(self.lane_id)
            distance = (lane_length - self.position) + leader.position - leader.length
        
        # Don't subtract safety margin here - it causes false collision detection
        # The safety margin is handled in the Krauss model instead
        return max(0.0, distance)
    
    def _estimate_acceleration(self) -> float:
        """
        Estimate current acceleration based on speed and distance to leader.
        
        Returns:
            Estimated acceleration (m/s²)
        """
        leader = self.get_leader()
        if leader is None:
            # Free flow - accelerate towards max speed
            if self.speed < self.target_speed:
                return 2.0  # Max acceleration
            return 0.0
        
        distance_to_leader = self.calculate_distance_to_leader()
        
        # Simple estimation
        if distance_to_leader > 50:
            # Plenty of space
            speed_diff = leader.speed - self.speed
            if speed_diff > 0:
                return min(2.0, speed_diff * 0.1)
            return 0.0
        elif distance_to_leader < 20:
            # Too close - need to brake
            return -2.0
        
        # Normal following
        if leader.speed < self.speed:
            return -1.0
        return 0.0
    
    def step(self):
        """
        Update the vehicle's state for one time step.
        """
        dt = self.model.time_step
        
        # Calculate distance to leader
        distance_to_leader = self.calculate_distance_to_leader()
        
        # Get leader speed
        leader_speed = None
        leader = self.get_leader()
        if leader is not None:
            leader_speed = leader.speed
        
        # Calculate next speed using Krauss model
        # The Krauss model already handles safe speed calculation
        next_speed = self.krauss_model.calculate_next_speed(
            self.speed,
            distance_to_leader,
            leader_speed
        )
        
        # Only emergency stop if extremely close (less than 0.5m)
        if distance_to_leader < 0.5:
            self.speed = 0
            return
        
        # Update speed (Krauss model handles gradual deceleration)
        self.speed = next_speed
        
        # Ensure minimum speed to prevent complete stalling
        # Minimum speeds multiplied by 10 for 10x faster movement (much faster for demos)
        if distance_to_leader > 50.0 or leader is None:
            self.speed = max(self.speed, 20.0 * 10.0)  # Minimum 200 m/s (720 km/h) - 10x faster
        elif distance_to_leader > 20.0:
            # Medium distance - maintain at least 150 m/s (540 km/h) - 10x faster
            self.speed = max(self.speed, 15.0 * 10.0)
        elif distance_to_leader > 10.0:
            # Close but not too close - maintain at least 100 m/s (360 km/h) - 10x faster
            self.speed = max(self.speed, 10.0 * 10.0)
        # If very close, let Krauss model handle it
        
        # Update position
        next_position = self.krauss_model.calculate_position_update(
            self.position,
            self.speed,
            dt
        )
        
        # Check for collision after position update across all lanes
        collision_detected = self._check_collision_after_move(next_position)
        
        if collision_detected:
            # Collision ahead (including overlapping lanes) – wait in place
            self.speed = 0
            return
        
        # No collision or speed is very low - proceed normally
        
        # Check if vehicle has reached end of lane
        lane_length = self.model.get_lane_length(self.lane_id)
        if next_position >= lane_length:
            # Vehicle has reached end of lane - try to transition to connected lane
            transition_successful = False
            
            # CRITICAL: If lane ends at frame edge, vehicles should EXIT, not transition
            # Only try transitions if lane does NOT end at frame edge
            is_at_edge = self.model.is_lane_end_at_frame_edge(self.lane_id)
            
            if is_at_edge:
                # Lane ends at frame edge - vehicles should exit, not transition
                print(f"🚪 Vehicle {self.unique_id} reached end of lane {self.lane_id} at frame edge - removing (no transitions attempted)")
                self.position = lane_length
                self.speed = 0
                self.model.remove_vehicle(self)
                return  # Exit immediately
            
            # Lane does NOT end at edge - try transitions
            if not self._try_lane_transition():
                # No transition possible - stop at end of lane
                print(f"  ⏸️  Vehicle {self.unique_id} stopping at end of lane {self.lane_id} (not at edge, no transition available)")
                self.position = lane_length
                self.speed = 0
            # If transition was successful, position was already set in _try_lane_transition()
        else:
            self.position = next_position
    
    def _try_lane_transition(self) -> bool:
        """
        Try to transition to a connected lane at the end of current lane.
        Uses route planning to choose the correct lane based on desired route type.
        
        Returns:
            True if transition successful, False otherwise
        """
        current_lane = self.model.road_network.get_lane(self.lane_id)
        if not current_lane or not current_lane.connected_lanes:
            return False
        
        # Check traffic light if approaching intersection
        intersection = self.model._get_intersection_for_lane(self.lane_id)
        if intersection:
            if not intersection.can_proceed(self.lane_id):
                # Red light - stop at intersection
                self.speed = 0
                self.position = self.model.get_lane_length(self.lane_id)
                return False
        
        # Choose connected lane based on route type
        next_lane_id = self._choose_next_lane_by_route(current_lane)
        
        if next_lane_id is None:
            return False
        
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
        
        # Update route_type to match the lane we're entering
        # This ensures vehicles follow the lane's direction (e.g., if lane 3 turns left, vehicle follows left)
        next_lane = self.model.road_network.get_lane(next_lane_id)
        if next_lane and current_lane.route_types.get(next_lane_id):
            # Update route_type to match the route type of the lane we're entering
            self.route_type = current_lane.route_types.get(next_lane_id, 'straight')
        
        print(f"Vehicle {self.unique_id} transitioned from lane {old_lane_id} to lane {next_lane_id} (route: {self.route_type})")
        return True
    
    def _choose_next_lane_by_route(self, current_lane) -> Optional[int]:
        """
        Choose the next lane based on desired route type.
        Skips connection lanes and goes directly to target lanes.
        
        Args:
            current_lane: Current lane object
            
        Returns:
            Next lane ID or None if not found
        """
        if not current_lane.connected_lanes:
            return None
        
        # Helper function to resolve connection lanes to their target lanes
        def resolve_lane(lane_id: int) -> int:
            """If lane_id is a connection lane, return its target. Otherwise return lane_id."""
            lane = self.model.road_network.get_lane(lane_id)
            if lane and lane.is_lane_change_connection and lane.target_lane_id is not None:
                return lane.target_lane_id
            return lane_id
        
        # Find lane matching desired route type
        for connected_lane_id in current_lane.connected_lanes:
            route_type = current_lane.route_types.get(connected_lane_id, 'straight')
            if route_type == self.route_type:
                # Resolve connection lane to target lane if needed
                resolved_lane_id = resolve_lane(connected_lane_id)
                return resolved_lane_id
        
        # If exact match not found, prefer straight, then right, then left
        for preferred_type in ['straight', 'right', 'left']:
            for connected_lane_id in current_lane.connected_lanes:
                route_type = current_lane.route_types.get(connected_lane_id, 'straight')
                if route_type == preferred_type:
                    # Resolve connection lane to target lane if needed
                    resolved_lane_id = resolve_lane(connected_lane_id)
                    return resolved_lane_id
        
        # Fallback to first available (resolve connection lane if needed)
        if current_lane.connected_lanes:
            resolved_lane_id = resolve_lane(current_lane.connected_lanes[0])
            return resolved_lane_id
        
        return None
    
    def _check_collision_after_move(self, new_position: float) -> bool:
        """
        Check if moving to new_position would cause a bounding box collision.
        
        Args:
            new_position: Proposed new position
            
        Returns:
            True if collision would occur, False otherwise
        """
        # Calculate our new world position and angle
        lane = self.model.road_network.get_lane(self.lane_id)
        if not lane:
            return False
        
        # CRITICAL: ALWAYS use lane centerline position - vehicles cannot deviate
        new_point = lane.get_position_at_distance(new_position)
        new_world_x, new_world_y = new_point.x, new_point.y
        
        # Angle based on lane direction at the proposed position
        lane_dir = lane.get_direction_at_distance(new_position) if hasattr(lane, 'get_direction_at_distance') else lane.direction
        new_angle = np.arctan2(lane_dir[1], lane_dir[0]) if lane_dir is not None else self.get_visual_angle()
        
        # Check all vehicles in the simulation (not just current lane)
        for vehicle in self.model.vehicles:
            if vehicle == self:
                continue
            
            other_world_x, other_world_y = vehicle.get_visual_position()
            other_angle = vehicle.get_visual_angle()
            center_dist = np.sqrt((new_world_x - other_world_x)**2 + (new_world_y - other_world_y)**2)
            min_center_gap = (self.length + vehicle.length) / 2 + 36.0  # Six-times larger buffer beyond vehicle bodies
            if center_dist < min_center_gap:
                return True
            
            if self._bounding_boxes_overlap(
                new_world_x, new_world_y, self.length, self.width, new_angle,
                other_world_x, other_world_y, vehicle.length, vehicle.width, other_angle
            ):
                return True
        
        return False
    
    def _bounding_boxes_overlap(
        self,
        x1: float, y1: float, l1: float, w1: float, a1: float,
        x2: float, y2: float, l2: float, w2: float, a2: float
    ) -> bool:
        """
        Check if two rotated bounding boxes overlap using simplified distance check.
        
        Args:
            x1, y1: Center of first box
            l1, w1: Length and width of first box
            a1: Angle of first box
            x2, y2: Center of second box
            l2, w2: Length and width of second box
            a2: Angle of second box
            
        Returns:
            True if boxes overlap, False otherwise
        """
        # Calculate distance between centers
        center_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        # Calculate maximum extent (half-diagonal) of each box
        max_extent1 = np.sqrt((l1/2)**2 + (w1/2)**2)
        max_extent2 = np.sqrt((l2/2)**2 + (w2/2)**2)
        
        # Safety margin to prevent bounding box collisions - expanded for wider spacing
        safety_margin = 30.0  # meters
        
        # If distance between centers is less than sum of extents + safety margin, collision
        if center_dist < max_extent1 + max_extent2 + safety_margin:
            return True
        
        return False
    
    def get_visual_position(self) -> Tuple[float, float]:
        """
        Get the visual position for rendering.
        Vehicles always follow the lane centerline.
        
        Returns:
            Tuple of (x, y) coordinates for visualization (on lane centerline)
        """
        lane = self.model.road_network.get_lane(self.lane_id)
        if not lane:
            return (0.0, 0.0)
        point = lane.get_position_at_distance(self.position)
        return point.x, point.y
    
    def get_visual_angle(self) -> float:
        """
        Get the visual angle for rendering.
        Angle is based on the lane direction at the vehicle's position.
        
        Returns:
            Angle in radians
        """
        # If changing lanes, interpolate between current and target lane angles
        lane_direction = self.model.get_lane_direction(self.lane_id, self.position)
        return np.arctan2(lane_direction[1], lane_direction[0])
