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
from ..models.road_network import Point, TrafficLights
import os

# Hardcoded lane pairs that must consider each other (mutual awareness)
# Pairs are bidirectional (we include both orders when checking)
MUTUAL_AWARE_LANE_PAIRS = {
    (1, 2), (2, 1),
    (3, 4), (4, 3),
}

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

    is_stopping_on_traffic_lights: bool
    current_acceleration: float # may be nagative
    reaction_time_counter: float
    
    def __init__(self, 
                 model: TrafficSimulationModel,
                 unique_id: int,
                 lane_id: int,
                 position: float = 0.0,
                 speed: float = 0.0,
                 max_speed: float = 30.0,
                 max_acceleration: float = 2.0,
                 max_deceleration: float = -10.0,
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
        self.unique_id = unique_id
        
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

        self.is_stopping_on_traffic_lights = False
        self.current_acceleration = 0.0
        self.reaction_time_counter = 0.0
        
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
    
    
    def step(self):
        """
        Update the vehicle's state for one time step.
        """
        dt = self.model.time_step

        # Calculate distance to leader
        distance_to_leader = self.calculate_distance_to_leader()

        # Calculate leader speed constraint
        leader_based_speed = self._calculate_leader_based_desired_speed(distance_to_leader)
        
        # Get traffic lights instance
        all_traffic_lights: List[TrafficLights] = self.model.get_traffic_lights_in_lane(self.lane_id)
        all_traffic_lights.sort(key=lambda x: x.position)
        applicable_traffic_lights = next((x for x in all_traffic_lights if x.position > self.position), None)

        # Take into account traffic lights
        # self._determine_if_stopping_on_traffic_lights(distance_to_leader, leader_based_speed, applicable_traffic_lights)
        
        # if self.is_stopping_on_traffic_lights: 
        #     self.speed = self.krauss_model.calculate_traffic_lights_based_next_speed(
        #         self.speed,
        #         applicable_traffic_lights.position - self.position - self.length / 2,
        #         dt
        #     )
        # else: # If not stopping on traffic lights
        #     self.speed = leader_based_speed
        current_speed = self.speed
        print(f"leader_based_speed: {leader_based_speed}")

        if applicable_traffic_lights is not None:

            distance_to_traffic_lights = applicable_traffic_lights.position - self.position - self.length / 2
            # traffic_lights_based_speed = self.krauss_model.calculate_traffic_lights_based_next_speed(
            #         self.speed,
            #         distance_to_traffic_lights,
            #         dt
            #     )
            # Determine traffic light state and pass appropriate leader speed to Krauss
            # If the light is green, do NOT treat it as a stationary leader (pass None)
            # so vehicles do not stop while it's green. If it's red or yellow, treat
            # it as a stopped leader (leader_speed = 0) which enforces stopping.
            tl_state = None
            try:
                tl_state = applicable_traffic_lights.get_state()
            except Exception:
                tl_state = None

            leader_for_tl = None if tl_state == 'green' else 0.0

            traffic_lights_based_speed = self.krauss_model.calculate_next_speed(
                self.speed,
                max(0.0, distance_to_traffic_lights - self.length / 2),
                self.model.time_step,
                leader_for_tl
            )

        
            self.speed = min(leader_based_speed, traffic_lights_based_speed)
            print(f"(veh: {self.unique_id}) traffic lights: distance: {distance_to_traffic_lights}, tl-based speed: {traffic_lights_based_speed}, speed: {self.speed}")
        else: # No traffic lights in front of
            leader_speed = self.get_leader().speed if self.get_leader() is not None else "None"
            self.speed = leader_based_speed
            print(f"(veh: {self.unique_id}) leader: distance: {distance_to_leader}, speed(l/f): {leader_speed}/{self.speed}, next_speed: {current_speed}")

        # Update position
        next_position = self._calculate_position_update(
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
            
            # Lanes 1, 2, 3, 7, and 8: vehicles should disappear at the end, no transitions
            # This prevents random stopping and unwanted lane changes
            if self.lane_id in [1, 2, 3, 7, 8]:
                print(f"🚪 Vehicle {self.unique_id} reached end of lane {self.lane_id} - removing (no transitions for lanes 1, 2, 3, 7, 8)")
                self.position = lane_length
                self.speed = 0
                self.model.remove_vehicle(self)
                return  # Exit immediately
            
            if is_at_edge:
                # Lane ends at frame edge - vehicles should exit, not transition
                print(f"🚪 Vehicle {self.unique_id} reached end of lane {self.lane_id} at frame edge - removing (no transitions attempted)")
                self.position = lane_length
                self.speed = 0
                self.model.remove_vehicle(self)
                return  # Exit immediately
            
            # Lane does NOT end at edge - try transitions
            # Lane does NOT end at edge - only allow transitions for true turning lanes
            current_lane = self.model.road_network.get_lane(self.lane_id)
            # route_types maps connected_lane_id -> route_type; lane_change connections are not turning
            has_turning_connection = False
            if current_lane and current_lane.route_types:
                for rt in current_lane.route_types.values():
                    if rt != 'lane_change':
                        has_turning_connection = True
                        break

            if has_turning_connection:
                # Only attempt transition if this lane is a turning lane
                if not self._try_lane_transition():
                    # No transition possible - stop at end of lane
                    print(f"  ⏸️  Vehicle {self.unique_id} stopping at end of lane {self.lane_id} (not at edge, no transition available)")
                    self.position = lane_length
                    self.speed = 0
            else:
                # Not a turning lane (or only lane_change connections) - remove vehicle at lane end
                print(f"🚪 Vehicle {self.unique_id} reached end of non-turning lane {self.lane_id} - removing")
                self.position = lane_length
                self.speed = 0
                self.model.remove_vehicle(self)
            # If transition was successful, position was already set in _try_lane_transition()
        else:
            self.position = next_position

    def _calculate_safe_turning_based_speed(self):
        current_position = self.position

        lane = self.model.road_network.get_lane(self.lane_id)
        sorted_by_distance = sorted(lane.centerline_points, key=lambda x: abs(current_position - lane.get_distance_from_start(x)))

        if len(sorted_by_distance) < 3:
            # Assuming it is a straight road
            return self.krauss_model.max_speed
        
        if 15 < abs(current_position - lane.get_distance_from_start(sorted_by_distance[2])):
            # Assuming it is a straight road
            return self.krauss_model.max_speed
        
        points = sorted_by_distance[:3]
        # Calculate radius of circle through 3 points
        p1, p2, p3 = points
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y

        # Using circumradius formula
        denom = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
        if abs(denom) < 1e-9:
            return self.krauss_model.max_speed

        a = ((x1**2 + y1**2) * (y2 - y3) + (x2**2 + y2**2) * (y3 - y1) + (x3**2 + y3**2) * (y1 - y2)) / denom
        b = ((x1**2 + y1**2) * (x3 - x2) + (x2**2 + y2**2) * (x1 - x3) + (x3**2 + y3**2) * (x2 - x1)) / denom

        radius = np.sqrt((x1 - a)**2 + (y1 - b)**2)
        
        return np.sqrt(0.8 * radius)


    def _determine_if_stopping_on_traffic_lights(self, distance_to_leader: int, leader_based_speed: float, applicable_traffic_lights: TrafficLights):

        # If no traffic lights in front of, do not stop
        if applicable_traffic_lights is None:
            self.is_stopping_on_traffic_lights = False
            return
        
        # Calculate distance to traffic lights
        distance = applicable_traffic_lights.position - self.position

        # Calculate leader possible stopping distance
        v_0_l = self.speed
        a_l = abs(self.krauss_model.max_deceleration) # non negative form
        min_leader_stopping_distance = v_0_l**2 / (2*a_l) # derived from v_0*t - 0.5*a*t**2 when t = v_0 / a

        # If leader is able to stop and is before traffic lights
        if min_leader_stopping_distance < distance - distance_to_leader and distance_to_leader < distance:
            # Do not try to stop on traffic lights on your own - stop with the leader
            self.is_stopping_on_traffic_lights = False
            return

        # Calculate possible stopping distance
        v_0 = self.speed
        a = abs(self.krauss_model.max_deceleration) # non negative form
        min_stopping_distance = v_0**2 / (2*a) # derived from v_0*t - 0.5*a*t**2 when t = v_0 / a

        # Handle the red light case
        if applicable_traffic_lights.get_state() == 'red' and distance < min_stopping_distance * 1.5:
            self.is_stopping_on_traffic_lights = True
            return

        # Handle the yellow light case
        necessary_time = distance / leader_based_speed # necessary time to reach traffic lights with leader-based speed
        if applicable_traffic_lights.get_state() == 'yellow' and necessary_time > 3:
            self.is_stopping_on_traffic_lights = True
            return
        
        # Handle the green light case
        if applicable_traffic_lights.get_state() == 'green':
            self.is_stopping_on_traffic_lights = False
            return

    def _calculate_leader_based_desired_speed(self, distance_to_leader: float) -> float:
        # Get leader speed from same-lane leader first
        leader_speed = None
        leader = self.get_leader()
        leader_length = self.length
        effective_gap = distance_to_leader - self.length / 2 - leader_length / 2
        if leader is not None:
            leader_speed = leader.speed
            leader_length = leader.length
            effective_gap = distance_to_leader - self.length / 2 - leader_length / 2

        # Additionally consider vehicles in adjacent lanes that are slightly
        # ahead and close laterally (e.g., where two lane centerlines split).
        # If such a vehicle is found within a forward search distance, treat
        # it as a virtual leader with its longitudinal gap and speed.
        try:
            lane = self.model.road_network.get_lane(self.lane_id)
            lane_dir = lane.get_direction_at_distance(self.position) if hasattr(lane, 'get_direction_at_distance') else lane.direction
            dir_x, dir_y = lane_dir
            # Search parameters
            # Reduce forward search and lateral threshold so adjacent lanes are
            # considered only when they are almost coincident. Make the
            # detection area smaller to avoid influencing cars after the split.
            forward_search_distance = 4.0   # meters ahead to consider (smaller)
            lateral_close_threshold = 0.1   # meters lateral separation to consider as blocking (smaller)

            closest_virtual_gap = float('inf')
            closest_virtual_speed = None

            for other in self.model.vehicles:
                if other is self:
                    continue
                # Get other's world position
                ox, oy = other.get_visual_position()
                # Vector from us to other
                dx = ox - lane.get_position_at_distance(self.position).x
                dy = oy - lane.get_position_at_distance(self.position).y
                # Project onto lane direction to get longitudinal distance
                longitudinal = dx * dir_x + dy * dir_y
                # Lateral separation (abs of projection onto perpendicular)
                perp_x, perp_y = -dir_y, dir_x
                lateral = abs(dx * perp_x + dy * perp_y)

                # Only consider vehicles in same lane or in explicitly configured
                # mutual-aware lane pairs. This disables cross-lane consideration
                # for all other neighboring lanes.
                is_mutual = (self.lane_id, other.lane_id) in MUTUAL_AWARE_LANE_PAIRS

                if not (other.lane_id == self.lane_id or is_mutual):
                    continue

                # Consider only vehicles ahead within forward_search_distance
                if not (longitudinal > 0 and longitudinal < forward_search_distance):
                    continue

                # Compute gap approximated by longitudinal minus half-lengths
                gap = longitudinal - (other.length/2) - (self.length/2)
                if gap < closest_virtual_gap:
                    closest_virtual_gap = gap
                    closest_virtual_speed = other.speed

            if closest_virtual_speed is not None and closest_virtual_gap >= 0:
                # If this virtual leader is closer than the same-lane leader gap, use it
                if closest_virtual_gap < effective_gap or leader is None:
                    effective_gap = closest_virtual_gap
                    leader_speed = closest_virtual_speed
        except Exception:
            # On any failure, fall back to same-lane leader behavior
            pass

        # Calculate next speed using Krauss model
        next_speed = self.krauss_model.calculate_next_speed(
            self.speed,
            max(0.0, effective_gap),
            self.model.time_step,
            leader_speed
        )

        return next_speed
    
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
        
        # Check vehicles in the same lane and immediate adjacent lanes.
        # For adjacent lanes use a smaller safety threshold so cars don't react
        # to vehicles that are merely driving side-by-side in neighboring lanes.
        # Only include adjacent lanes if they are explicitly configured as
        # mutual-aware pairs. This prevents vehicles in unrelated neighboring
        # lanes from blocking each other.
        lane_ids_to_check = {self.lane_id}
        left = self.model.get_adjacent_lane(self.lane_id, 'left')
        right = self.model.get_adjacent_lane(self.lane_id, 'right')
        if left is not None and (self.lane_id, left) in MUTUAL_AWARE_LANE_PAIRS:
            lane_ids_to_check.add(left)
        if right is not None and (self.lane_id, right) in MUTUAL_AWARE_LANE_PAIRS:
            lane_ids_to_check.add(right)

        # Also always include any explicitly configured mutual-aware lanes
        # even if they are not reported as adjacent by the road network.
        # Note: Lanes 1 and 2 are still checked for collisions, but blocking rules are relaxed below
        for a, b in MUTUAL_AWARE_LANE_PAIRS:
            if a == self.lane_id:
                lane_ids_to_check.add(b)

        # Gather vehicles in those lanes
        lane_vehicles = [v for v in self.model.vehicles if v.lane_id in lane_ids_to_check]
        for vehicle in lane_vehicles:
            if vehicle == self:
                continue
            # Current visual position of the other vehicle
            other_world_x, other_world_y = vehicle.get_visual_position()
            other_angle = vehicle.get_visual_angle()
            dx = other_world_x - new_world_x
            dy = other_world_y - new_world_y
            center_dist = np.sqrt(dx*dx + dy*dy)

            # Compute lateral separation relative to our lane direction.
            # This helps decide whether an adjacent lane is effectively the same
            # physical lane (very close centerlines) or a distinct lane.
            sin_a = np.sin(new_angle)
            cos_a = np.cos(new_angle)
            # Perpendicular unit vector to lane direction
            perp_x, perp_y = -sin_a, cos_a
            lateral_sep = abs(dx * perp_x + dy * perp_y)

            # Threshold for considering adjacent lane 'close' (meters).
            # Make this very small so vehicles in neighboring lanes do not
            # react unless centerlines are effectively coincident.
            lateral_close_threshold = 0.1

            # Compute longitudinal separation along our lane direction
            cos_a = np.cos(new_angle)
            sin_a = np.sin(new_angle)
            longitudinal_sep = dx * cos_a + dy * sin_a

            # If the other vehicle is clearly behind us (more than 2m), ignore it
            # - prevents being blocked by vehicles that are behind or beside us
            if longitudinal_sep < -2.0:
                continue

            # Base gap between centers (half lengths sum)
            base_min_center_gap = (self.length + vehicle.length) / 2

            # Compute longitudinal separation along our lane direction
            cos_a = np.cos(new_angle)
            sin_a = np.sin(new_angle)
            longitudinal_sep = dx * cos_a + dy * sin_a

            # If this pair is mutual-aware, enforce a minimum longitudinal gap
            # so vehicles on configured pairs do not occupy the same forward space.
            # Exception: lanes 1 and 2 should not block each other - they can drive side-by-side
            is_mutual_pair = (self.lane_id, vehicle.lane_id) in MUTUAL_AWARE_LANE_PAIRS
            # Skip strict mutual blocking for lanes 1 and 2 to prevent random stops
            is_lanes_1_2 = {self.lane_id, vehicle.lane_id} == {1, 2}
            mutual_min_longitudinal = 6.0
            debug = os.getenv('TRAFFIC_DEBUG')
            # Only enforce if the other vehicle is ahead (positive longitudinal)
            # Skip this for lanes 1 and 2 - they can drive parallel without blocking
            if is_mutual_pair and not is_lanes_1_2 and 0.0 < longitudinal_sep < mutual_min_longitudinal:
                if debug:
                    print(f"[DEBUG] mutual-block: {self.unique_id}(lane {self.lane_id}) <- {vehicle.unique_id}(lane {vehicle.lane_id}) long_sep={longitudinal_sep:.2f}")
                return True

            # If this pair is in the mutual-awareness set, treat like same-lane
            # Exception: lanes 1 and 2 should not use strict same-lane collision rules
            is_mutual_pair = (self.lane_id, vehicle.lane_id) in MUTUAL_AWARE_LANE_PAIRS
            is_lanes_1_2 = {self.lane_id, vehicle.lane_id} == {1, 2}
            # For lanes 1 and 2, only use strict rules if actually in same lane or very close laterally
            # Otherwise treat them as separate lanes that can drive side-by-side
            if vehicle.lane_id == self.lane_id or lateral_sep < lateral_close_threshold:
                # Same lane or very close lanes -> use full safety gap
                min_center_gap = base_min_center_gap
            elif is_mutual_pair and not is_lanes_1_2:
                # Mutual-aware pair (but not lanes 1/2) -> use full safety gap
                min_center_gap = base_min_center_gap
            else:
                # Adjacent lane but separated enough -> allow closer side-by-side
                adjacent_factor = 0.7
                min_center_gap = base_min_center_gap * adjacent_factor

            # If centers are too close to the other vehicle's current position, that's a collision/blocked
            if center_dist < min_center_gap:
                if debug:
                    print(f"[DEBUG] center-block: {self.unique_id}(lane {self.lane_id}) <- {vehicle.unique_id}(lane {vehicle.lane_id}) center_dist={center_dist:.2f} min_gap={min_center_gap:.2f}")
                return True

            # Also consider the other vehicle's predicted next position (so two vehicles
            # moving towards the same spot don't pass through each other). Estimate
            # other_next_position using its current speed and the model time step.
            try:
                dt = self.model.time_step
                other_lane = self.model.road_network.get_lane(vehicle.lane_id)
                other_next_pos = vehicle.position + vehicle.speed * dt
                # clamp to lane length
                other_lane_length = self.model.get_lane_length(vehicle.lane_id)
                if other_next_pos > other_lane_length:
                    other_next_pos = other_lane_length
                other_next_point = other_lane.get_position_at_distance(other_next_pos)
                other_next_x, other_next_y = other_next_point.x, other_next_point.y
                dx2 = other_next_x - new_world_x
                dy2 = other_next_y - new_world_y
                center_dist_next = np.sqrt(dx2*dx2 + dy2*dy2)
            except Exception:
                center_dist_next = float('inf')

            # Compute longitudinal separation to other's predicted next pos and skip
            # if other will be behind us.
            try:
                longitudinal_sep_next = dx2 * cos_a + dy2 * sin_a
            except Exception:
                longitudinal_sep_next = float('inf')

            if center_dist_next < min_center_gap and longitudinal_sep_next > -2.0:
                if debug:
                    print(f"[DEBUG] pred-block: {self.unique_id}(lane {self.lane_id}) <- {vehicle.unique_id}(lane {vehicle.lane_id}) center_dist_next={center_dist_next:.2f} min_gap={min_center_gap:.2f} long_next={longitudinal_sep_next:.2f}")
                return True

            # For bounding-box overlap check, decide adjacency based on lateral separation
            # to the other's predicted/ current position (use the smaller lateral sep)
            try:
                # lateral separation to current and next positions
                sin_a = np.sin(new_angle)
                cos_a = np.cos(new_angle)
                perp_x, perp_y = -sin_a, cos_a
                lateral_current = abs((other_world_x - new_world_x) * perp_x + (other_world_y - new_world_y) * perp_y)
                lateral_next = abs((other_next_x - new_world_x) * perp_x + (other_next_y - new_world_y) * perp_y)
                lateral_min = min(lateral_current, lateral_next)
            except Exception:
                lateral_min = lateral_sep

            considered_adjacent = (vehicle.lane_id != self.lane_id and lateral_min >= lateral_close_threshold)
            is_mutual_pair = (self.lane_id, vehicle.lane_id) in MUTUAL_AWARE_LANE_PAIRS
            is_lanes_1_2 = {self.lane_id, vehicle.lane_id} == {1, 2}
            # Lanes 1 and 2 should not use strict mutual collision rules - allow side-by-side driving
            considered_mutual = is_mutual_pair and not is_lanes_1_2

            # Check bounding boxes against current position. For configured mutual
            # lane pairs, force the larger safety margin so they don't run into each other.
            if self._bounding_boxes_overlap(
                new_world_x, new_world_y, self.length, self.width, new_angle,
                other_world_x, other_world_y, vehicle.length, vehicle.width, other_angle,
                considered_adjacent=considered_adjacent,
                considered_mutual=considered_mutual
            ):
                return True

            # And check bounding boxes against other's predicted position
            if self._bounding_boxes_overlap(
                new_world_x, new_world_y, self.length, self.width, new_angle,
                other_next_x, other_next_y, vehicle.length, vehicle.width, other_angle,
                considered_adjacent=considered_adjacent,
                considered_mutual=considered_mutual
            ):
                return True
        
        return False
    
    def _bounding_boxes_overlap(
        self,
        x1: float, y1: float, l1: float, w1: float, a1: float,
        x2: float, y2: float, l2: float, w2: float, a2: float,
        considered_adjacent: bool = False,
        considered_mutual: bool = False
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
        # Use Separating Axis Theorem (SAT) for robust oriented bounding box overlap.
        # Compute corner points for each box.
        def corners(x, y, length, width, angle):
            # Half-dimensions
            hl = length / 2.0
            hw = width / 2.0
            ca = np.cos(angle)
            sa = np.sin(angle)
            # local axes
            ux = (ca, sa)  # along length
            uy = (-sa, ca)  # along width (perp)
            # corners in order
            return [
                (x + ux[0]*hl + uy[0]*hw, y + ux[1]*hl + uy[1]*hw),
                (x - ux[0]*hl + uy[0]*hw, y - ux[1]*hl + uy[1]*hw),
                (x - ux[0]*hl - uy[0]*hw, y - ux[1]*hl - uy[1]*hw),
                (x + ux[0]*hl - uy[0]*hw, y + ux[1]*hl - uy[1]*hw),
            ]

        c1 = corners(x1, y1, l1, w1, a1)
        c2 = corners(x2, y2, l2, w2, a2)

        # Axes to test are the normals of all edges (two unique axes per box)
        def axes_from_corners(c):
            axes = []
            for i in range(2):
                p1 = c[i]
                p2 = c[(i+1) % 4]
                edge = (p2[0]-p1[0], p2[1]-p1[1])
                # normal
                norm = (-edge[1], edge[0])
                # normalize
                norm_len = np.hypot(norm[0], norm[1])
                if norm_len > 1e-8:
                    axes.append((norm[0]/norm_len, norm[1]/norm_len))
            return axes

        axes = axes_from_corners(c1) + axes_from_corners(c2)

        # Safety margin: smaller for adjacent lanes, larger for same-lane.
        # If this is an explicitly configured mutual pair, increase the
        # safety margin so vehicles never overlap.
        safety_margin_same = 0.5
        safety_margin_adjacent = 0.2
        if considered_mutual:
            margin = safety_margin_same * 1.6
        else:
            margin = safety_margin_adjacent if considered_adjacent else safety_margin_same

        # Project corners onto each axis and check for gaps
        for ax in axes:
            proj1 = [p[0]*ax[0] + p[1]*ax[1] for p in c1]
            proj2 = [p[0]*ax[0] + p[1]*ax[1] for p in c2]
            min1, max1 = min(proj1) - margin, max(proj1) + margin
            min2, max2 = min(proj2) - margin, max(proj2) + margin
            # if separated on this axis -> no overlap
            if max1 < min2 or max2 < min1:
                return False

        # No separating axis found -> boxes overlap (considering margin)
        return True
    
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

    def _calculate_position_update(self, 
                                current_position: float,
                                current_speed: float,
                                dt: float = 1.0) -> float:
        """
        Calculate the next position based on current speed.
        
        Args:
            current_position: Current position (m)
            current_speed: Current speed (m/s)
            dt: Time step (seconds)
            
        Returns:
            Next position (m)
        """
        return current_position + current_speed * dt