"""
Road Network Implementation

This module contains classes for representing roads, lanes, and intersections
in the traffic simulation. It provides the infrastructure for vehicles to
move along defined paths.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Point:
    """Represents a 2D point with x, y coordinates."""
    x: float
    y: float
    
    def distance_to(self, other: 'Point') -> float:
        """Calculate distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __add__(self, other: 'Point') -> 'Point':
        """Add two points."""
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        """Subtract two points."""
        return Point(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Point':
        """Multiply point by scalar."""
        return Point(self.x * scalar, self.y * scalar)


class Lane:
    """
    Represents a single lane in the road network.
    
    A lane is defined by:
    - Start and end points
    - Direction vector
    - Length
    - Adjacent lanes (left/right)
    - Speed limit
    - Centerline points (the path vehicles follow exactly)
    """
    
    def __init__(self, 
                 lane_id: int,
                 start_point: Point,
                 end_point: Point,
                 speed_limit: float = 30.0,  # m/s
                 lane_width: float = 3.5,   # meters
                 centerline_points: Optional[List[Point]] = None):  # Optional centerline path points
        """
        Initialize a lane.
        
        Args:
            lane_id: Unique identifier for the lane
            start_point: Starting point of the lane
            end_point: Ending point of the lane
            speed_limit: Speed limit for the lane (m/s)
            lane_width: Width of the lane (meters)
            centerline_points: Optional list of centerline points for vehicles to follow exactly
        """
        self.lane_id = lane_id
        self.start_point = start_point
        self.end_point = end_point
        self.speed_limit = speed_limit
        self.lane_width = lane_width
        self.centerline_points = centerline_points if centerline_points else []
        
        # Lane-changing connection metadata (for special transition lanes)
        self.source_lane_id: Optional[int] = None
        self.target_lane_id: Optional[int] = None
        self.is_lane_change_connection: bool = False
        
        # Calculate lane properties
        # If centerline points exist, use them to calculate length and direction
        if self.centerline_points and len(self.centerline_points) >= 2:
            # Calculate total length along centerline
            total_length = 0.0
            for i in range(len(self.centerline_points) - 1):
                total_length += self.centerline_points[i].distance_to(self.centerline_points[i + 1])
            self.length = total_length
            # Direction is from first to last centerline point
            if len(self.centerline_points) >= 2:
                dx = self.centerline_points[-1].x - self.centerline_points[0].x
                dy = self.centerline_points[-1].y - self.centerline_points[0].y
                length = np.sqrt(dx**2 + dy**2)
                self.direction = (dx / length, dy / length) if length > 0 else (0, 0)
            else:
                self.length = start_point.distance_to(end_point)
                self.direction = self._calculate_direction()
        else:
            self.length = start_point.distance_to(end_point)
            self.direction = self._calculate_direction()
        
        # Spawn position (set from JSON if available)
        self.spawn_position: Optional[Point] = None
        self.spawn_direction: Optional[Tuple[float, float]] = None
        
        # Adjacent lanes
        self.left_lane_id = None
        self.right_lane_id = None
        
        # Connected lanes (for intersections)
        self.connected_lanes: List[int] = []
        
        # Route types for connected lanes (for navigation)
        # Maps connected_lane_id -> route_type ('straight', 'left', 'right')
        self.route_types: Dict[int, str] = {}
        
        # Lane purpose (for intersections)
        self.allows_left_turn = False
        self.allows_right_turn = False
        self.allows_straight = True
    
    def _calculate_direction(self) -> Tuple[float, float]:
        """Calculate the direction vector of the lane."""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return (0, 0)
        
        return (dx / length, dy / length)
    
    def get_spawn_position(self) -> Optional[Point]:
        """
        Get the spawn position for this lane.
        
        Returns:
            Spawn position point, or None if not available
        """
        return self.spawn_position
    
    def get_spawn_direction(self) -> Tuple[float, float]:
        """
        Get the spawn direction for this lane.
        
        Returns:
            Direction vector (dx, dy) normalized
        """
        if self.spawn_direction:
            return self.spawn_direction
        # Fallback to lane direction
        return self.direction
    
    def lanes_connect_at_endpoints(self, other_lane: 'Lane', threshold: float = 5.0) -> bool:
        """
        Check if this lane's end point coincides with another lane's start point (turning lane).
        
        Args:
            other_lane: Another lane to check against
            threshold: Maximum distance in meters to consider endpoints as coinciding
            
        Returns:
            True if lanes physically connect (turning lane scenario)
        """
        # Check if end point of this lane is close to start point of other lane
        endpoint_distance = self.end_point.distance_to(other_lane.start_point)
        return endpoint_distance < threshold
    
    def get_lane_type(self) -> str:
        """
        Determine if this is an access lane (ends without connection) or turning lane (connects to another).
        This is determined by checking if the lane has connected lanes.
        
        Returns:
            'access' if lane ends without connection, 'turning' if connected, 'unknown' if unsure
        """
        if self.connected_lanes:
            return 'turning'
        return 'access'
    
    def get_position_at_distance(self, distance: float) -> Point:
        """
        Get the position along the lane at a given distance.
        CRITICAL: If centerline_points exist, vehicles MUST follow the centerline path exactly.
        There is NO fallback - vehicles can ONLY move on centerlines.
        
        Args:
            distance: Distance along the lane from start (meters)
            
        Returns:
            Point at the specified distance (ALWAYS on centerline if centerline exists)
        """
        # Clamp distance to lane length
        distance = max(0, min(distance, self.length))
        
        # CRITICAL: If centerline points exist, vehicles MUST follow them exactly
        # No vehicle can deviate from centerlines - this is the ONLY allowed path
        if self.centerline_points and len(self.centerline_points) >= 2:
            # Calculate cumulative distances along centerline segments
            cumulative_distances = [0.0]
            for i in range(len(self.centerline_points) - 1):
                seg_length = self.centerline_points[i].distance_to(self.centerline_points[i + 1])
                cumulative_distances.append(cumulative_distances[-1] + seg_length)
            
            # Find which segment contains the target distance
            for i in range(len(cumulative_distances) - 1):
                if distance <= cumulative_distances[i + 1]:
                    # Distance is within this segment
                    seg_start_dist = cumulative_distances[i]
                    seg_end_dist = cumulative_distances[i + 1]
                    seg_length = seg_end_dist - seg_start_dist
                    
                    if seg_length > 0:
                        # Interpolate within this segment - vehicle stays ON the centerline
                        t = (distance - seg_start_dist) / seg_length
                        p1 = self.centerline_points[i]
                        p2 = self.centerline_points[i + 1]
                        return Point(
                            p1.x + t * (p2.x - p1.x),
                            p1.y + t * (p2.y - p1.y)
                        )
                    else:
                        return self.centerline_points[i]
            
            # If distance exceeds all segments, return last point
            return self.centerline_points[-1]
        
        # FALLBACK: Only used if centerlines don't exist (shouldn't happen in production)
        # This creates a straight line path - vehicles should always have centerlines
        if self.length > 0:
            t = distance / self.length
            x = self.start_point.x + t * (self.end_point.x - self.start_point.x)
            y = self.start_point.y + t * (self.end_point.y - self.start_point.y)
            return Point(x, y)
        
        return self.start_point
    
    def get_direction_at_distance(self, distance: float) -> Tuple[float, float]:
        """
        Get the direction vector at a specific distance along the lane.
        If centerline_points exist, returns direction from current position to a point ahead.
        This ensures vehicles face forward along the path.
        
        Args:
            distance: Distance along the lane from start (meters)
            
        Returns:
            Normalized direction vector (dx, dy)
        """
        # Clamp distance to lane length
        distance = max(0, min(distance, self.length))
        
        # If centerline points exist, get direction from current position to point ahead
        if self.centerline_points and len(self.centerline_points) >= 2:
            # Calculate cumulative distances along centerline segments
            cumulative_distances = [0.0]
            for i in range(len(self.centerline_points) - 1):
                seg_length = self.centerline_points[i].distance_to(self.centerline_points[i + 1])
                cumulative_distances.append(cumulative_distances[-1] + seg_length)
            
            # Get current position on centerline
            current_pos = self.get_position_at_distance(distance)
            
            # Look ahead a reasonable distance (e.g., 5 meters) to get forward direction
            look_ahead_distance = min(5.0, self.length - distance)
            if look_ahead_distance > 0.1:  # Only if we have room ahead
                ahead_pos = self.get_position_at_distance(distance + look_ahead_distance)
                dx = ahead_pos.x - current_pos.x
                dy = ahead_pos.y - current_pos.y
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    return (dx / length, dy / length)
            
            # Fallback: use segment direction
            # Find which segment contains the target distance
            for i in range(len(cumulative_distances) - 1):
                if distance <= cumulative_distances[i + 1]:
                    # Distance is within this segment
                    p1 = self.centerline_points[i]
                    p2 = self.centerline_points[i + 1]
                    dx = p2.x - p1.x
                    dy = p2.y - p1.y
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        return (dx / length, dy / length)
                    else:
                        # Degenerate segment, use next segment or overall direction
                        if i + 1 < len(self.centerline_points) - 1:
                            p1 = self.centerline_points[i + 1]
                            p2 = self.centerline_points[i + 2]
                            dx = p2.x - p1.x
                            dy = p2.y - p1.y
                            length = np.sqrt(dx**2 + dy**2)
                            if length > 0:
                                return (dx / length, dy / length)
                        break
            
            # Fallback: use overall direction
            return self.direction
        
        # Fallback: use overall lane direction
        return self.direction
    
    def get_distance_from_start(self, point: Point) -> float:
        """
        Get the distance along the lane from the start point.
        If centerline_points exist, finds closest point on centerline.
        Otherwise, projects onto straight line.
        
        Args:
            point: Point to measure distance from
            
        Returns:
            Distance along the lane (meters)
        """
        # If centerline points exist, find closest point on centerline
        if self.centerline_points and len(self.centerline_points) >= 2:
            # Find closest segment and interpolate
            min_dist = float('inf')
            best_distance = 0.0
            cumulative_dist = 0.0
            
            for i in range(len(self.centerline_points) - 1):
                p1 = self.centerline_points[i]
                p2 = self.centerline_points[i + 1]
                seg_length = p1.distance_to(p2)
                
                # Project point onto this segment
                seg_vec = Point(p2.x - p1.x, p2.y - p1.y)
                point_vec = Point(point.x - p1.x, point.y - p1.y)
                
                if seg_length > 0:
                    t = (seg_vec.x * point_vec.x + seg_vec.y * point_vec.y) / (seg_length * seg_length)
                    t = max(0, min(1, t))  # Clamp to segment
                    
                    # Closest point on segment
                    closest = Point(
                        p1.x + t * seg_vec.x,
                        p1.y + t * seg_vec.y
                    )
                    
                    dist_to_seg = point.distance_to(closest)
                    if dist_to_seg < min_dist:
                        min_dist = dist_to_seg
                        best_distance = cumulative_dist + t * seg_length
                
                cumulative_dist += seg_length
            
            return max(0, min(best_distance, self.length))
        
        # Fallback: Project point onto lane direction (straight line)
        lane_vector = Point(self.end_point.x - self.start_point.x, self.end_point.y - self.start_point.y)
        point_vector = Point(point.x - self.start_point.x, point.y - self.start_point.y)
        
        if self.length > 0:
            # Project onto lane direction
            t = (lane_vector.x * point_vector.x + lane_vector.y * point_vector.y) / (self.length * self.length)
            t = max(0, min(1, t))  # Clamp to lane
            return t * self.length
        
        return 0.0
    
    def distance_to_centerline(self, point: Point) -> float:
        """
        Calculate the minimum distance from a point to this lane's centerline.
        This creates a "combined line function" representation of the lane.
        
        Args:
            point: Point to measure distance from
            
        Returns:
            Minimum distance to the centerline (meters)
        """
        if self.centerline_points and len(self.centerline_points) >= 2:
            # Find minimum distance to any segment of the centerline
            min_dist = float('inf')
            
            for i in range(len(self.centerline_points) - 1):
                p1 = self.centerline_points[i]
                p2 = self.centerline_points[i + 1]
                seg_length = p1.distance_to(p2)
                
                if seg_length == 0:
                    # Zero-length segment, just use distance to point
                    dist = point.distance_to(p1)
                    min_dist = min(min_dist, dist)
                    continue
                
                # Project point onto this segment
                seg_vec = Point(p2.x - p1.x, p2.y - p1.y)
                point_vec = Point(point.x - p1.x, point.y - p1.y)
                
                t = (seg_vec.x * point_vec.x + seg_vec.y * point_vec.y) / (seg_length * seg_length)
                t = max(0, min(1, t))  # Clamp to segment
                
                # Closest point on segment
                closest = Point(
                    p1.x + t * seg_vec.x,
                    p1.y + t * seg_vec.y
                )
                
                dist_to_seg = point.distance_to(closest)
                min_dist = min(min_dist, dist_to_seg)
            
            return min_dist
        
        # Fallback: project onto straight line
        lane_vector = Point(self.end_point.x - self.start_point.x, self.end_point.y - self.start_point.y)
        point_vector = Point(point.x - self.start_point.x, point.y - self.start_point.y)
        
        if self.length > 0:
            t = (lane_vector.x * point_vector.x + lane_vector.y * point_vector.y) / (self.length * self.length)
            t = max(0, min(1, t))
            closest_point = Point(
                self.start_point.x + t * lane_vector.x,
                self.start_point.y + t * lane_vector.y
            )
            return point.distance_to(closest_point)
        
        return point.distance_to(self.start_point)
    
    def snap_point_to_centerline(self, point: Point) -> Point:
        """
        Snap a point to the nearest point on this lane's centerline.
        
        Args:
            point: Point to snap
            
        Returns:
            Closest point on the centerline
        """
        if self.centerline_points and len(self.centerline_points) >= 2:
            # Find closest point on centerline
            min_dist = float('inf')
            best_point = None
            
            for i in range(len(self.centerline_points) - 1):
                p1 = self.centerline_points[i]
                p2 = self.centerline_points[i + 1]
                seg_length = p1.distance_to(p2)
                
                if seg_length == 0:
                    continue
                
                # Project point onto this segment
                seg_vec = Point(p2.x - p1.x, p2.y - p1.y)
                point_vec = Point(point.x - p1.x, point.y - p1.y)
                
                t = (seg_vec.x * point_vec.x + seg_vec.y * point_vec.y) / (seg_length * seg_length)
                t = max(0, min(1, t))  # Clamp to segment
                
                # Closest point on segment
                closest = Point(
                    p1.x + t * seg_vec.x,
                    p1.y + t * seg_vec.y
                )
                
                dist_to_seg = point.distance_to(closest)
                if dist_to_seg < min_dist:
                    min_dist = dist_to_seg
                    best_point = closest
            
            if best_point:
                return best_point
        
        # Fallback: project onto straight line
        lane_vector = Point(self.end_point.x - self.start_point.x, self.end_point.y - self.start_point.y)
        point_vector = Point(point.x - self.start_point.x, point.y - self.start_point.y)
        
        if self.length > 0:
            t = (lane_vector.x * point_vector.x + lane_vector.y * point_vector.y) / (self.length * self.length)
            t = max(0, min(1, t))
            return Point(
                self.start_point.x + t * lane_vector.x,
                self.start_point.y + t * lane_vector.y
            )
        
        return self.start_point
    
    def point_lies_on_lane(self, point: Point, tolerance: float = 10.0) -> bool:
        """
        Check if a point lies on this lane's centerline (within tolerance).
        Uses interpolation to check if point is on any segment of the centerline.
        
        Args:
            point: Point to check
            tolerance: Maximum distance from centerline to consider point as "on" the lane (meters)
            
        Returns:
            True if point lies on the lane centerline within tolerance, False otherwise
        """
        if not self.centerline_points or len(self.centerline_points) < 2:
            # Fallback to straight line check
            snapped = self.snap_point_to_centerline(point)
            return point.distance_to(snapped) <= tolerance
        
        # Check all segments of the centerline
        for i in range(len(self.centerline_points) - 1):
            p1 = self.centerline_points[i]
            p2 = self.centerline_points[i + 1]
            seg_length = p1.distance_to(p2)
            
            if seg_length == 0:
                # Zero-length segment, just check distance to point
                if point.distance_to(p1) <= tolerance:
                    return True
                continue
            
            # Project point onto this segment
            seg_vec = Point(p2.x - p1.x, p2.y - p1.y)
            point_vec = Point(point.x - p1.x, point.y - p1.y)
            
            t = (seg_vec.x * point_vec.x + seg_vec.y * point_vec.y) / (seg_length * seg_length)
            t = max(0, min(1, t))  # Clamp to segment
            
            # Closest point on segment
            closest = Point(
                p1.x + t * seg_vec.x,
                p1.y + t * seg_vec.y
            )
            
            dist_to_seg = point.distance_to(closest)
            if dist_to_seg <= tolerance:
                return True
        
        return False
    
    def set_adjacent_lanes(self, left_lane_id: Optional[int] = None, 
                          right_lane_id: Optional[int] = None):
        """
        Set the adjacent lanes.
        
        Args:
            left_lane_id: ID of the left adjacent lane
            right_lane_id: ID of the right adjacent lane
        """
        self.left_lane_id = left_lane_id
        self.right_lane_id = right_lane_id
    
    def add_connected_lane(self, lane_id: int, route_type: str = 'straight'):
        """
        Add a connected lane (for intersections).
        
        Args:
            lane_id: ID of the connected lane
            route_type: Type of route ('straight', 'left', 'right')
        """
        if lane_id not in self.connected_lanes:
            self.connected_lanes.append(lane_id)
            self.route_types[lane_id] = route_type
            
            # Update lane capabilities based on route types
            if route_type == 'left':
                self.allows_left_turn = True
            elif route_type == 'right':
                self.allows_right_turn = True
            elif route_type == 'straight':
                self.allows_straight = True


class Road:
    """
    Represents a road consisting of multiple lanes.
    
    A road can have multiple lanes in the same direction or
    lanes in opposite directions (bidirectional).
    """
    
    def __init__(self, road_id: int, name: str = ""):
        """
        Initialize a road.
        
        Args:
            road_id: Unique identifier for the road
            name: Name of the road (optional)
        """
        self.road_id = road_id
        self.name = name
        self.lanes: List[Lane] = []
        self.lane_map: Dict[int, Lane] = {}
    
    def add_lane(self, lane: Lane):
        """
        Add a lane to the road.
        
        Args:
            lane: Lane to add
        """
        self.lanes.append(lane)
        self.lane_map[lane.lane_id] = lane
    
    def get_lane(self, lane_id: int) -> Optional[Lane]:
        """
        Get a lane by its ID.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Lane object or None if not found
        """
        return self.lane_map.get(lane_id)
    
    def get_adjacent_lane(self, lane_id: int, direction: str) -> Optional[int]:
        """
        Get the adjacent lane ID in the specified direction.
        
        Args:
            lane_id: ID of the current lane
            direction: 'left' or 'right'
            
        Returns:
            Adjacent lane ID or None if not found
        """
        lane = self.get_lane(lane_id)
        if lane is None:
            return None
        
        if direction == 'left':
            return lane.left_lane_id
        elif direction == 'right':
            return lane.right_lane_id
        
        return None


class Intersection:
    """
    Represents an intersection where multiple roads meet.
    
    An intersection manages:
    - Traffic light states
    - Right-of-way rules
    - Lane connections
    """
    
    def __init__(self, intersection_id: int, center_point: Point):
        """
        Initialize an intersection.
        
        Args:
            intersection_id: Unique identifier for the intersection
            center_point: Center point of the intersection
        """
        self.intersection_id = intersection_id
        self.center_point = center_point
        self.connected_lanes: List[int] = []
        # Traffic light system with phases
        self.traffic_lights: Dict[int, str] = {}  # lane_id -> state
        self.traffic_light_timer = 0.0
        self.current_phase = 0
        
        # Traffic light phases: list of tuples (duration, {lane_id: state})
        # Each phase defines which lanes have green/yellow/red
        self.traffic_light_phases: List[Tuple[float, Dict[int, str]]] = []
        
        # Default phase durations (seconds)
        self.green_duration = 25.0
        self.yellow_duration = 3.0
        self.red_duration = 2.0
    
    def add_lane(self, lane_id: int):
        """
        Add a lane that connects to this intersection.
        
        Args:
            lane_id: ID of the connecting lane
        """
        if lane_id not in self.connected_lanes:
            self.connected_lanes.append(lane_id)
            self.traffic_lights[lane_id] = 'red'  # Default to red
    
    def set_traffic_light_phases(self, phases: List[Tuple[float, Dict[int, str]]]):
        """
        Set traffic light phases programmatically.
        
        Args:
            phases: List of tuples (duration_seconds, {lane_id: state})
        """
        self.traffic_light_phases = phases
        if phases:
            # Initialize with first phase
            _, first_phase = phases[0]
            self.traffic_lights = first_phase.copy()
    
    def update_traffic_lights(self, dt: float):
        """
        Update traffic light states based on phases.
        
        Args:
            dt: Time step (seconds)
        """
        self.traffic_light_timer += dt
        
        if not self.traffic_light_phases:
            # Fallback to simple cycling if no phases defined
            if self.traffic_light_timer >= 30.0:
                for lane_id in self.traffic_lights:
                    if self.traffic_lights[lane_id] == 'red':
                        self.traffic_lights[lane_id] = 'green'
                    elif self.traffic_lights[lane_id] == 'green':
                        self.traffic_lights[lane_id] = 'yellow'
                    elif self.traffic_lights[lane_id] == 'yellow':
                        self.traffic_lights[lane_id] = 'red'
                self.traffic_light_timer = 0
            return
        
        # Use phase-based system
        if self.current_phase >= len(self.traffic_light_phases):
            self.current_phase = 0
        
        phase_duration, phase_states = self.traffic_light_phases[self.current_phase]
        
        if self.traffic_light_timer >= phase_duration:
            # Move to next phase
            self.current_phase = (self.current_phase + 1) % len(self.traffic_light_phases)
            self.traffic_light_timer = 0.0
            
            # Update light states
            _, new_phase_states = self.traffic_light_phases[self.current_phase]
            self.traffic_lights = new_phase_states.copy()
    
    def can_proceed(self, lane_id: int) -> bool:
        """
        Check if a vehicle can proceed through the intersection.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            True if vehicle can proceed, False otherwise
        """
        light_state = self.traffic_lights.get(lane_id, 'red')
        return light_state == 'green'


class RoadNetwork:
    """
    Manages the entire road network including roads, lanes, and intersections.
    
    This class provides the main interface for:
    - Creating and managing roads
    - Finding paths between points
    - Managing intersections
    - Querying lane information
    """
    
    def __init__(self):
        """Initialize the road network."""
        self.roads: List[Road] = []
        self.intersections: List[Intersection] = []
        self.all_lanes: Dict[int, Lane] = {}
        self.road_map: Dict[int, Road] = {}
        self.intersection_map: Dict[int, Intersection] = {}
    
    def add_road(self, road: Road):
        """
        Add a road to the network.
        
        Args:
            road: Road to add
        """
        self.roads.append(road)
        self.road_map[road.road_id] = road
        
        # Add all lanes to the global lane map
        for lane in road.lanes:
            self.all_lanes[lane.lane_id] = lane
    
    def add_intersection(self, intersection: Intersection):
        """
        Add an intersection to the network.
        
        Args:
            intersection: Intersection to add
        """
        self.intersections.append(intersection)
        self.intersection_map[intersection.intersection_id] = intersection
    
    def get_lane(self, lane_id: int) -> Optional[Lane]:
        """
        Get a lane by its ID.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Lane object or None if not found
        """
        return self.all_lanes.get(lane_id)
    
    def get_adjacent_lane(self, lane_id: int, direction: str) -> Optional[int]:
        """
        Get the adjacent lane ID in the specified direction.
        
        Args:
            lane_id: ID of the current lane
            direction: 'left' or 'right'
            
        Returns:
            Adjacent lane ID or None if not found
        """
        lane = self.get_lane(lane_id)
        if lane is None:
            return None
        
        if direction == 'left':
            return lane.left_lane_id
        elif direction == 'right':
            return lane.right_lane_id
        
        return None
    
    def find_lane_for_point(self, point: Point, tolerance: float = 10.0) -> Optional[int]:
        """
        Find which lane a point belongs to by checking if it lies on any lane's centerline.
        Uses interpolation to check all lane segments.
        
        Args:
            point: Point to check (in world coordinates)
            tolerance: Maximum distance from centerline to consider point as "on" the lane (meters)
            
        Returns:
            Lane ID if point lies on a lane, None otherwise
        """
        best_lane_id = None
        min_distance = float('inf')
        
        for lane_id, lane in self.all_lanes.items():
            if lane.point_lies_on_lane(point, tolerance):
                # Point lies on this lane, check distance to be sure
                snapped = lane.snap_point_to_centerline(point)
                dist = point.distance_to(snapped)
                if dist < min_distance:
                    min_distance = dist
                    best_lane_id = lane_id
        
        return best_lane_id
    
    def get_lane_length(self, lane_id: int) -> float:
        """
        Get the length of a lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Length of the lane in meters, or 0 if not found
        """
        lane = self.get_lane(lane_id)
        return lane.length if lane else 0.0
    
    def get_lane_position(self, lane_id: int) -> Tuple[float, float]:
        """
        Get the starting position of a lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Tuple of (x, y) coordinates
        """
        lane = self.get_lane(lane_id)
        if lane:
            return (lane.start_point.x, lane.start_point.y)
        return (0.0, 0.0)
    
    def get_lane_direction(self, lane_id: int, distance: Optional[float] = None) -> Tuple[float, float]:
        """
        Get the direction vector of a lane.
        If distance is provided and lane has centerline, returns direction at that distance.
        
        Args:
            lane_id: ID of the lane
            distance: Optional distance along lane to get direction at
            
        Returns:
            Tuple of (dx, dy) direction vector
        """
        lane = self.get_lane(lane_id)
        if lane:
            if distance is not None:
                return lane.get_direction_at_distance(distance)
            return lane.direction
        return (1.0, 0.0)  # Default direction
    
    def create_simple_highway(self, 
                            start_point: Point, 
                            end_point: Point, 
                            num_lanes: int = 2) -> Road:
        """
        Create a simple highway with multiple lanes.
        
        Args:
            start_point: Starting point of the highway
            end_point: Ending point of the highway
            num_lanes: Number of lanes in each direction
            
        Returns:
            Created road
        """
        road_id = len(self.roads)
        road = Road(road_id, f"Highway_{road_id}")
        
        # Calculate lane spacing
        lane_width = 3.5  # meters
        total_width = num_lanes * lane_width
        
        # Create lanes
        lane_ids = []
        for i in range(num_lanes):
            lane_id = len(self.all_lanes) + i
            lane_ids.append(lane_id)
            
            # Calculate lane start and end points
            offset = (i - num_lanes/2 + 0.5) * lane_width
            
            # Calculate perpendicular direction
            dx = end_point.x - start_point.x
            dy = end_point.y - start_point.y
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                perp_x = -dy / length
                perp_y = dx / length
            else:
                perp_x, perp_y = 0, 1
            
            lane_start = Point(
                start_point.x + perp_x * offset,
                start_point.y + perp_y * offset
            )
            lane_end = Point(
                end_point.x + perp_x * offset,
                end_point.y + perp_y * offset
            )
            
            lane = Lane(lane_id, lane_start, lane_end)
            road.add_lane(lane)
        
        # Set adjacent lanes after all lanes are created
        for i, lane_id in enumerate(lane_ids):
            lane = road.get_lane(lane_id)
            if lane:
                if i > 0:
                    lane.left_lane_id = lane_ids[i - 1]
                if i < num_lanes - 1:
                    lane.right_lane_id = lane_ids[i + 1]
        
        self.add_road(road)
        return road
