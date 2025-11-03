"""
Vehicle Agent Implementation

This module contains the Vehicle agent class that represents individual vehicles
in the traffic simulation. Each vehicle uses the Krauss car-following model
and can perform lane-changing maneuvers.
"""

import numpy as np
import mesa
import random
from typing import Optional, List, Tuple, TYPE_CHECKING
from ..utils.krauss_model import KraussModel
from ..utils.mobil_model import MOBILModel
from ..models.road_network import Point

if TYPE_CHECKING:
    from ..models.traffic_model import TrafficSimulationModel
else:
    TrafficSimulationModel = None


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
        self.last_progress_printed = -1  # Track last printed progress percentage for debug
        self._target_lane_id = None  # Target lane ID for non-adjacent lane changes
        
        # Car-following model
        self.krauss_model = KraussModel(
            max_speed=max_speed,
            max_acceleration=max_acceleration,
            max_deceleration=max_deceleration
        )
        
        # MOBIL lane-changing model
        self.mobil_model = MOBILModel(
            politeness_factor=0.5,
            acceleration_threshold=0.2,
            safety_criterion=-2.0,
            right_lane_bias=0.1
        )
        
        # State tracking
        self.is_changing_lanes = False
        self.lane_change_progress = 0.0  # 0.0 to 1.0
        
        # Route planning
        self.route_type = random.choice(['straight', 'left', 'right'])  # Desired route at intersection
        self.route_planned = False  # Whether route has been planned
        
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
    
    def can_change_lane(self, direction: str) -> bool:
        """
        Check if the vehicle can safely change lanes in the given direction using MOBIL.
        Lane changes are ONLY allowed between lanes going in the same direction.
        
        Args:
            direction: 'left' or 'right'
            
        Returns:
            True if lane change is safe and beneficial, False otherwise
        """
        if self.is_changing_lanes:
            return False
        
        # Get target lane
        target_lane_id = self.model.get_adjacent_lane(self.lane_id, direction)
        if target_lane_id is None:
            return False
        
        # CRITICAL: Check if lanes go in the same direction
        # Lane changes should ONLY happen between parallel lanes going the same direction
        current_lane = self.model.road_network.get_lane(self.lane_id)
        target_lane = self.model.road_network.get_lane(target_lane_id)
        
        if not current_lane or not target_lane:
            return False
        
        # VERIFY: Check that the adjacent lane relationship is correct
        # If vehicle wants to go LEFT, the target should be the LEFT lane
        # If vehicle wants to go RIGHT, the target should be the RIGHT lane
        expected_left = current_lane.left_lane_id
        expected_right = current_lane.right_lane_id
        
        if direction == 'left' and target_lane_id != expected_left:
            print(f"  ⚠ WARNING: Vehicle {self.unique_id} trying to go LEFT, but target_lane_id={target_lane_id} != left_lane_id={expected_left}")
            print(f"     Current lane {self.lane_id}: left={expected_left}, right={expected_right}")
            return False
        
        if direction == 'right' and target_lane_id != expected_right:
            print(f"  ⚠ WARNING: Vehicle {self.unique_id} trying to go RIGHT, but target_lane_id={target_lane_id} != right_lane_id={expected_right}")
            print(f"     Current lane {self.lane_id}: left={expected_left}, right={expected_right}")
            return False
        
        # Check if lanes have similar directions (dot product close to 1.0)
        # This ensures they're parallel and going the same way
        current_dir = current_lane.direction
        target_dir = target_lane.direction
        
        # Calculate dot product of direction vectors
        dot_product = current_dir[0] * target_dir[0] + current_dir[1] * target_dir[1]
        
        # Require lanes to be nearly parallel (dot product > 0.7 means angle < 45 degrees)
        if dot_product < 0.7:
            print(f"  ⚠ WARNING: Vehicle {self.unique_id} on lane {self.lane_id} wants to change {direction} to lane {target_lane_id}, but lanes not parallel (dot={dot_product:.3f})")
            return False  # Lanes don't go in the same direction - cannot change lanes
        
        # Check basic safety: no vehicles too close
        target_lane_vehicles = self.model.get_vehicles_in_lane(target_lane_id)
        
        # Find leader and follower in target lane
        target_leader = None
        target_follower = None
        min_leader_distance = float('inf')
        min_follower_distance = float('inf')
        
        for vehicle in target_lane_vehicles:
            distance = vehicle.position - self.position
            if distance > 0:  # Ahead
                if distance < min_leader_distance:
                    min_leader_distance = distance
                    target_leader = vehicle
            else:  # Behind
                if abs(distance) < min_follower_distance:
                    min_follower_distance = abs(distance)
                    target_follower = vehicle
        
        # Safety check: need sufficient gap
        safe_gap_ahead = 30.0  # meters
        safe_gap_behind = 25.0  # meters
        
        if target_leader and min_leader_distance < safe_gap_ahead:
            return False
        if target_follower and min_follower_distance < safe_gap_behind:
            return False
        
        # Use MOBIL to evaluate if lane change is beneficial
        # Estimate accelerations
        current_accel = self._estimate_acceleration()
        
        # Estimate acceleration in target lane
        new_leader_distance = min_leader_distance if target_leader else float('inf')
        new_leader_speed = target_leader.speed if target_leader else None
        new_accel = self.mobil_model.estimate_new_acceleration(
            self.speed,
            self.target_speed,
            new_leader_distance,
            new_leader_speed
        )
        
        # Estimate impact on follower
        follower_accel_before = 0.0
        follower_accel_after = -2.0  # Conservative estimate
        
        if target_follower:
            # Estimate follower's acceleration before and after
            follower_accel_before = target_follower._estimate_acceleration()
            # After lane change, follower would be closer to leader
            follower_accel_after = -1.5  # Would need to brake more
        
        # Evaluate using MOBIL
        should_change, incentive = self.mobil_model.evaluate_lane_change(
            current_accel,
            new_accel,
            follower_accel_after,
            follower_accel_before,
            direction
        )
        
        return should_change
    
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
    
    def start_lane_change(self, direction: str):
        """
        Start a lane change maneuver.
        
        Args:
            direction: 'left' or 'right'
        """
        if self.can_change_lane(direction):
            target_lane_id = self.model.get_adjacent_lane(self.lane_id, direction)
            self.desired_lane_change = direction
            self.is_changing_lanes = True
            self.lane_change_progress = 0.0
            self.lane_change_timer = 0
            print(f"🔄 Vehicle {self.unique_id} STARTING lane change from lane {self.lane_id} to lane {target_lane_id} ({direction})")
    
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
        Uses _target_lane_id if set (for non-adjacent lane changes), otherwise uses adjacent lane.
        """
        if self.desired_lane_change:
            # Check if we have a specific target lane (for non-adjacent lane changes)
            if hasattr(self, '_target_lane_id') and self._target_lane_id is not None:
                target_lane_id = self._target_lane_id
            else:
                # Use standard adjacent lane lookup
                target_lane_id = self.model.get_adjacent_lane(self.lane_id, self.desired_lane_change)
            
            if target_lane_id is not None:
                old_lane_id = self.lane_id
                self.lane_id = target_lane_id
                print(f"✅ Vehicle {self.unique_id} COMPLETED lane change from lane {old_lane_id} to lane {target_lane_id} ({self.desired_lane_change})")
        
        self.is_changing_lanes = False
        self.desired_lane_change = None
        self.lane_change_progress = 0.0
        self.lane_change_timer = 0
        self.last_progress_printed = -1  # Reset progress tracking
        self._target_lane_id = None  # Clear target lane ID
    
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
        
        # Check for collision after position update
        # Be more lenient - only prevent movement if collision is truly imminent
        # The Krauss model should handle most collision avoidance
        collision_detected = self._check_collision_after_move(next_position)
        
        if collision_detected and self.speed > 0.1:
            # Collision detected - reduce speed but still allow movement
            leader = self.get_leader()
            if leader and leader.speed > 0:
                # Match leader's speed, but ensure minimum movement
                # Minimum speeds multiplied by 10 for 10x faster movement
                min_speed_lane2 = 8.0 * 10.0 if self.lane_id == 2 else 5.0 * 10.0
                min_speed = min_speed_lane2 if self.lane_id == 2 else 5.0 * 10.0
                self.speed = max(min_speed, min(self.speed, leader.speed * 0.95))
            else:
                # Slow down but don't stop completely
                # Minimum speeds multiplied by 10 for 10x faster movement
                min_speed = 10.0 * 10.0 if self.lane_id == 2 else 8.0 * 10.0
                self.speed = max(min_speed, self.speed * 0.85)
            
            # Still update position at reduced speed
            adjusted_position = self.position + self.speed * dt
            self.position = min(adjusted_position, next_position)
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
            # First, try direct connected lane transition (for turning lanes at intersections)
            if self._try_lane_transition():
                transition_successful = True
                # Position is already set in _try_lane_transition() to 0.0
            # If that fails, try adjacent lane transition (for lane-changing connections)
            elif self._try_adjacent_lane_transition():
                transition_successful = True
                # Position is already set in _try_adjacent_lane_transition() to 0.0
            # If that fails, try switching to adjacent lane if available (for access lanes)
            elif self._try_switch_to_adjacent_lane():
                transition_successful = True
                # Position is already set in _try_switch_to_adjacent_lane()
            
            if not transition_successful:
                # No transition possible - lane ends but not at edge (already checked above)
                # Stop and wait for space (this should be rare)
                print(f"  ⏸️  Vehicle {self.unique_id} stopping at end of lane {self.lane_id} (not at edge, no transition available)")
                self.position = lane_length
                self.speed = 0
            # If transition was successful, position was already set in the transition method
        else:
            self.position = next_position
        
        # Lane-changing decision (only if not already changing lanes)
        # Check every step for distance-based lane change detection
        if not self.is_changing_lanes:
            self._consider_lane_change()
        
        # Debug: show lane change progress if in progress (every 25% progress)
        if self.is_changing_lanes:
            progress_pct = int(self.lane_change_progress * 100)
            # Only print at 0%, 25%, 50%, 75% to avoid spam
            if progress_pct != self.last_progress_printed and progress_pct % 25 == 0:
                print(f"🔄 Vehicle {self.unique_id} lane change in progress: {progress_pct}% ({self.desired_lane_change})")
                self.last_progress_printed = progress_pct
    
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
    
    def _try_adjacent_lane_transition(self) -> bool:
        """
        Try to transition to an adjacent lane using lane-changing connections.
        This is used when a lane ends and there's no direct connection, but
        there's a lane-changing connection available (indicates end of access lane).
        
        Returns:
            True if transition successful, False otherwise
        """
        current_lane = self.model.road_network.get_lane(self.lane_id)
        if not current_lane:
            return False
        
        # Check all connected lanes for lane-changing connections
        for connected_lane_id in current_lane.connected_lanes:
            connected_lane = self.model.road_network.get_lane(connected_lane_id)
            if not connected_lane:
                continue
            
            # Check if this is a lane-changing connection lane
            if connected_lane.is_lane_change_connection and connected_lane.target_lane_id is not None:
                target_lane_id = connected_lane.target_lane_id
                
                # Check if the target lane has space at the beginning
                target_lane_vehicles = self.model.get_vehicles_in_lane(target_lane_id)
                min_start_distance = 20.0  # Need at least 20m clearance
                has_space = True
                for vehicle in target_lane_vehicles:
                    if vehicle.position < min_start_distance:
                        has_space = False
                        break
                
                if has_space:
                    # Transition through the connection lane first, then to target
                    # For simplicity, transition directly to target lane
                    old_lane_id = self.lane_id
                    self.lane_id = target_lane_id
                    self.position = 0.0  # Start at beginning of target lane
                    
                    print(f"Vehicle {self.unique_id} transitioned from lane {old_lane_id} to lane {target_lane_id} via adjacent lane connection")
                    return True
        
        return False
    
    def _try_switch_to_adjacent_lane(self) -> bool:
        """
        Try to switch to an adjacent lane when current lane ends.
        This is used for access lanes that end but have adjacent lanes available.
        Now has 50% probability to attempt adjacent lane switch.
        
        Returns:
            True if switch successful, False otherwise
        """
        # 50% chance to try adjacent lane switch (instead of always trying)
        if random.random() < 0.5:
            print(f"🚗 Vehicle {self.unique_id} attempting adjacent lane switch from lane {self.lane_id} (50% chance)")
        else:
            print(f"⏭️  Vehicle {self.unique_id} skipping adjacent lane switch from lane {self.lane_id} (50% chance)")
            return False
        
        current_lane = self.model.road_network.get_lane(self.lane_id)
        if not current_lane:
            return False
        
        # Try both left and right adjacent lanes
        for direction in ['left', 'right']:
            adjacent_lane_id = self.model.get_adjacent_lane(self.lane_id, direction)
            if adjacent_lane_id is None:
                continue
            
            adjacent_lane = self.model.road_network.get_lane(adjacent_lane_id)
            if not adjacent_lane:
                continue
            
            # Check if adjacent lane goes in similar direction (parallel lanes)
            current_dir = current_lane.direction
            adjacent_dir = adjacent_lane.direction
            
            # Calculate dot product to check if lanes are parallel
            dot_product = current_dir[0] * adjacent_dir[0] + current_dir[1] * adjacent_dir[1]
            if dot_product < 0.7:  # Not parallel enough
                print(f"  ⚠ Vehicle {self.unique_id}: Lane {adjacent_lane_id} not parallel enough (dot={dot_product:.2f})")
                continue
            
            # Find position on adjacent lane that corresponds to end of current lane
            # Use the end point of current lane to find closest point on adjacent lane
            current_end_point = current_lane.get_position_at_distance(self.model.get_lane_length(self.lane_id))
            adjacent_position = adjacent_lane.get_distance_from_start(current_end_point)
            
            # Check if there's space around that position on adjacent lane
            adjacent_lane_vehicles = self.model.get_vehicles_in_lane(adjacent_lane_id)
            min_gap = 30.0  # Need at least 30m gap
            has_space = True
            
            for vehicle in adjacent_lane_vehicles:
                distance = abs(vehicle.position - adjacent_position)
                if distance < min_gap:
                    has_space = False
                    print(f"  ⚠ Vehicle {self.unique_id}: Not enough space on lane {adjacent_lane_id} (gap={distance:.1f}m < {min_gap}m)")
                    break
            
            if has_space:
                # Switch to adjacent lane
                old_lane_id = self.lane_id
                self.lane_id = adjacent_lane_id
                self.position = adjacent_position
                
                print(f"✅ Vehicle {self.unique_id} switched from lane {old_lane_id} to adjacent lane {adjacent_lane_id} at position {adjacent_position:.1f}m")
                return True
            else:
                print(f"  ❌ Vehicle {self.unique_id}: Failed to switch to lane {adjacent_lane_id} ({direction}) - no space")
        
        print(f"  ❌ Vehicle {self.unique_id}: No suitable adjacent lane found")
        return False
    
    def _consider_lane_change(self):
        """
        Consider whether to change lanes using distance-based detection.
        Uses combined line functions for each lane and checks distance from point to line.
        Checks ALL lanes in the network, not just predefined adjacent lanes.
        This allows lane changes between lanes that are spatially close (e.g., lane 1 to lane 8)
        even if they don't have a direct adjacent relationship.
        For now, all cars change lanes if they can.
        """
        if self.is_changing_lanes:
            return
        
        # Get vehicle's current world position
        current_lane = self.model.road_network.get_lane(self.lane_id)
        if not current_lane:
            return
        
        vehicle_point = current_lane.get_position_at_distance(self.position)
        current_dir = current_lane.direction
        
        # Check ALL lanes in the network to find spatially close lanes
        # This allows lane changes even if lanes aren't marked as adjacent
        candidate_lanes = []
        lane_width = current_lane.lane_width if current_lane.lane_width else 3.5
        lane_change_threshold = lane_width * 3.0  # 3x lane width threshold for all lanes
        
        # Loop through all lanes in the network
        for lane_id, lane in self.model.road_network.all_lanes.items():
            # Skip current lane
            if lane_id == self.lane_id:
                continue
            
            # Check if lane goes in similar direction (parallel lanes)
            target_dir = lane.direction
            dot_product = current_dir[0] * target_dir[0] + current_dir[1] * target_dir[1]
            
            # Require lanes to be nearly parallel (dot product > 0.7 means angle < 45 degrees)
            if dot_product < 0.7:
                continue  # Lanes don't go in the same direction - skip
            
            # Calculate distance from vehicle point to this lane's centerline
            distance = lane.distance_to_centerline(vehicle_point)
            
            # If within threshold, this is a candidate for lane change
            if distance <= lane_change_threshold:
                # Determine direction (left or right) based on relative position
                # We can approximate this by checking which side of the current lane the target lane is on
                # For simplicity, we'll try both directions and let can_change_lane_with_target decide
                candidate_lanes.append((lane_id, lane, distance))
        
        # Sort candidates by distance (closest first)
        candidate_lanes.sort(key=lambda x: x[2])
        
        # Try to change to the closest lane
        for lane_id, lane, distance in candidate_lanes:
            # Determine direction by checking if we can change to this lane
            # Try to determine direction by checking adjacent lanes first
            left_lane_id = self.model.get_adjacent_lane(self.lane_id, 'left')
            right_lane_id = self.model.get_adjacent_lane(self.lane_id, 'right')
            
            # Determine direction: if target lane is on the left side, go left; otherwise right
            # We can approximate this by checking the cross product or relative position
            # For now, try both directions if we have adjacent lanes, or guess based on lane ID
            direction = None
            
            # If we have predefined adjacent lanes, check if target matches
            if left_lane_id == lane_id:
                direction = 'left'
            elif right_lane_id == lane_id:
                direction = 'right'
            else:
                # No predefined relationship - determine direction based on spatial position
                # Calculate perpendicular vector to current lane direction
                perp_x = -current_dir[1]
                perp_y = current_dir[0]
                
                # Get a point on the target lane near our position
                target_point = lane.get_position_at_distance(
                    lane.get_distance_from_start(vehicle_point)
                )
                
                # Vector from current position to target lane
                to_target = Point(target_point.x - vehicle_point.x, target_point.y - vehicle_point.y)
                
                # Dot product with perpendicular gives us which side
                side = perp_x * to_target.x + perp_y * to_target.y
                
                if side > 0:
                    direction = 'left'  # Target is on the left side
                else:
                    direction = 'right'  # Target is on the right side
            
            # Try to change to this lane using a modified can_change_lane that accepts target lane
            if direction and self._can_change_lane_to_target(lane_id, direction):
                print(f"🚗 Vehicle {self.unique_id} detected lane {lane_id} ({direction}) at distance {distance:.2f}m - initiating lane change")
                self._start_lane_change_to_target(lane_id, direction)
                return
        
        # Fallback: try predefined adjacent lanes if no spatial candidates found
        left_lane_id = self.model.get_adjacent_lane(self.lane_id, 'left')
        right_lane_id = self.model.get_adjacent_lane(self.lane_id, 'right')
        
        if left_lane_id is not None and self.can_change_lane('left'):
            print(f"🚗 Vehicle {self.unique_id} attempting lane change LEFT from lane {self.lane_id}")
            self.start_lane_change('left')
            return
        
        if right_lane_id is not None and self.can_change_lane('right'):
            print(f"🚗 Vehicle {self.unique_id} attempting lane change RIGHT from lane {self.lane_id}")
            self.start_lane_change('right')
            return
    
    def _can_change_lane_to_target(self, target_lane_id: int, direction: str) -> bool:
        """
        Check if the vehicle can safely change lanes to a specific target lane.
        This is similar to can_change_lane but accepts a specific target lane ID.
        
        Args:
            target_lane_id: ID of the target lane
            direction: 'left' or 'right' (for display purposes)
            
        Returns:
            True if lane change is safe, False otherwise
        """
        if self.is_changing_lanes:
            return False
        
        current_lane = self.model.road_network.get_lane(self.lane_id)
        target_lane = self.model.road_network.get_lane(target_lane_id)
        
        if not current_lane or not target_lane:
            return False
        
        # Check if lanes have similar directions (dot product close to 1.0)
        current_dir = current_lane.direction
        target_dir = target_lane.direction
        
        dot_product = current_dir[0] * target_dir[0] + current_dir[1] * target_dir[1]
        
        if dot_product < 0.7:
            return False  # Lanes don't go in the same direction
        
        # Check basic safety: no vehicles too close in target lane
        target_lane_vehicles = self.model.get_vehicles_in_lane(target_lane_id)
        
        safe_gap_ahead = 30.0  # meters
        safe_gap_behind = 25.0  # meters
        
        for vehicle in target_lane_vehicles:
            distance = vehicle.position - self.position
            if distance > 0 and distance < safe_gap_ahead:  # Ahead and too close
                return False
            if distance < 0 and abs(distance) < safe_gap_behind:  # Behind and too close
                return False
        
        return True
    
    def _start_lane_change_to_target(self, target_lane_id: int, direction: str):
        """
        Start a lane change maneuver to a specific target lane.
        
        Args:
            target_lane_id: ID of the target lane
            direction: 'left' or 'right'
        """
        if self._can_change_lane_to_target(target_lane_id, direction):
            # Temporarily set the target lane as adjacent so start_lane_change works
            # We'll override the lane change completion to use our target lane
            current_lane = self.model.road_network.get_lane(self.lane_id)
            if current_lane:
                # Temporarily modify adjacent lane to match target
                original_left = current_lane.left_lane_id
                original_right = current_lane.right_lane_id
                
                if direction == 'left':
                    current_lane.left_lane_id = target_lane_id
                else:
                    current_lane.right_lane_id = target_lane_id
                
                # Start the lane change
                self.desired_lane_change = direction
                self.is_changing_lanes = True
                self.lane_change_progress = 0.0
                self.lane_change_timer = 0
                
                # Store target lane ID for completion
                self._target_lane_id = target_lane_id
                
                # Restore original adjacent lanes
                current_lane.left_lane_id = original_left
                current_lane.right_lane_id = original_right
                
                print(f"🔄 Vehicle {self.unique_id} STARTING lane change from lane {self.lane_id} to lane {target_lane_id} ({direction})")
    
    def _lane_change_for_route(self):
        """
        Change lanes if needed to get into correct lane for desired route.
        """
        current_lane = self.model.road_network.get_lane(self.lane_id)
        if not current_lane:
            return
        
        # Check if current lane supports desired route
        route_supported = False
        for connected_lane_id in current_lane.connected_lanes:
            route_type = current_lane.route_types.get(connected_lane_id, 'straight')
            if route_type == self.route_type:
                route_supported = True
                break
        
        if route_supported:
            return  # Already in correct lane
        
        # Try to change to adjacent lane that supports route
        for direction in ['left', 'right']:
            target_lane_id = self.model.get_adjacent_lane(self.lane_id, direction)
            if target_lane_id is None:
                continue
            
            target_lane = self.model.road_network.get_lane(target_lane_id)
            if not target_lane:
                continue
            
            # Check if target lane supports desired route
            for connected_lane_id in target_lane.connected_lanes:
                route_type = target_lane.route_types.get(connected_lane_id, 'straight')
                if route_type == self.route_type:
                    # This lane supports our route - try to change
                    if self.can_change_lane(direction):
                        self.start_lane_change(direction)
                        return
    
    def _check_collision_after_move(self, new_position: float) -> bool:
        """
        Check if moving to new_position would cause a bounding box collision.
        
        Args:
            new_position: Proposed new position
            
        Returns:
            True if collision would occur, False otherwise
        """
        # Get all vehicles in the same lane
        lane_vehicles = self.model.get_vehicles_in_lane(self.lane_id)
        
        # Calculate our new world position and angle
        lane = self.model.road_network.get_lane(self.lane_id)
        if not lane:
            return False
        
        # CRITICAL: ALWAYS use lane centerline position - vehicles cannot deviate
        # This ensures collision checks use the exact same position calculation as rendering
        new_point = lane.get_position_at_distance(new_position)
        new_world_x, new_world_y = new_point.x, new_point.y
        
        # Angle is always based on lane direction
        new_angle = self.get_visual_angle()
        
        for vehicle in lane_vehicles:
            if vehicle == self:
                continue
            
            # Quick distance check first (much faster)
            lane_length = self.model.get_lane_length(self.lane_id)
            distance = abs(vehicle.position - new_position)
            
            # If distance is large, might be wrap-around - check actual gap
            if distance > lane_length * 0.5:
                if new_position > vehicle.position:
                    distance = new_position - vehicle.position
                else:
                    distance = (lane_length - vehicle.position) + new_position
            
            # Quick rejection: if too far along lane, no collision possible
            # Increased safety margin to prevent overlap
            min_distance_along_lane = (self.length + vehicle.length) / 2 + 5.0  # 5m safety margin
            if distance > min_distance_along_lane:
                continue  # Too far, no collision possible
            
            # Check bounding box collision for nearby vehicles
            other_world_x, other_world_y = vehicle.get_visual_position()
            other_angle = vehicle.get_visual_angle()
            
            # Check if bounding boxes overlap using simplified method
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
        
        # Safety margin to prevent bounding box collisions - increased to prevent overlap
        safety_margin = 2.5  # meters - larger margin to ensure no overlap
        
        # If distance between centers is less than sum of extents + safety margin, collision
        if center_dist < max_extent1 + max_extent2 + safety_margin:
            return True
        
        return False
    
    def get_visual_position(self) -> Tuple[float, float]:
        """
        Get the visual position for rendering.
        CRITICAL: ALWAYS returns position on lane centerline - vehicles CANNOT deviate from centerline
        except during lane changes, where they interpolate between centerlines.
        
        Vehicles can ONLY create x, y coordinates:
        1. On their current lane's centerline
        2. Between centerlines when changing lanes (interpolated)
        
        There is NO other way for vehicles to position themselves.
        
        Returns:
            Tuple of (x, y) coordinates for visualization (ALWAYS on centerline or between centerlines)
        """
        # Get the lane and use its method to calculate position
        lane = self.model.road_network.get_lane(self.lane_id)
        if not lane:
            return (0.0, 0.0)
        
        # If changing lanes, interpolate between current and target lane centerlines
        # This is the ONLY way vehicles can be between lanes
        if self.is_changing_lanes and self.desired_lane_change:
            # Check if we have a specific target lane (for non-adjacent lane changes)
            if hasattr(self, '_target_lane_id') and self._target_lane_id is not None:
                target_lane_id = self._target_lane_id
            else:
                # Use standard adjacent lane lookup
                target_lane_id = self.model.get_adjacent_lane(self.lane_id, self.desired_lane_change)
            
            if target_lane_id is not None:
                target_lane = self.model.road_network.get_lane(target_lane_id)
                if target_lane:
                    # Get position on current lane centerline (ALWAYS on centerline)
                    current_point = lane.get_position_at_distance(self.position)
                    # Get position on target lane centerline (at same distance along lane)
                    target_point = target_lane.get_position_at_distance(self.position)
                    
                    # Interpolate between centerlines based on lane change progress
                    # This creates a smooth transition between two centerlines
                    interp_x = current_point.x + (target_point.x - current_point.x) * self.lane_change_progress
                    interp_y = current_point.y + (target_point.y - current_point.y) * self.lane_change_progress
                    
                    return (interp_x, interp_y)
        
        # Normal case: ALWAYS on lane centerline - no deviation possible
        point = lane.get_position_at_distance(self.position)
        return point.x, point.y
    
    def get_visual_angle(self) -> float:
        """
        Get the visual angle for rendering.
        Angle is always based on the lane direction - vehicles follow lane direction.
        During lane changes, interpolate between current and target lane angles.
        
        Returns:
            Angle in radians
        """
        # If changing lanes, interpolate between current and target lane angles
        if self.is_changing_lanes and self.desired_lane_change:
            # Check if we have a specific target lane (for non-adjacent lane changes)
            if hasattr(self, '_target_lane_id') and self._target_lane_id is not None:
                target_lane_id = self._target_lane_id
            else:
                # Use standard adjacent lane lookup
                target_lane_id = self.model.get_adjacent_lane(self.lane_id, self.desired_lane_change)
            
            if target_lane_id is not None:
                # Get direction at current position for both lanes
                current_lane_dir = self.model.get_lane_direction(self.lane_id, self.position)
                target_lane_dir = self.model.get_lane_direction(target_lane_id, self.position)
                
                current_angle = np.arctan2(current_lane_dir[1], current_lane_dir[0])
                target_angle = np.arctan2(target_lane_dir[1], target_lane_dir[0])
                
                # Interpolate angle (handle wrap-around)
                angle_diff = target_angle - current_angle
                if angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                elif angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                return current_angle + angle_diff * self.lane_change_progress
        
        # Normal case: use current lane direction at current position
        # CRITICAL: Use direction at actual position, not overall lane direction
        # This ensures vehicles follow centerlines exactly, especially for vertical lanes
        lane_direction = self.model.get_lane_direction(self.lane_id, self.position)
        return np.arctan2(lane_direction[1], lane_direction[0])
