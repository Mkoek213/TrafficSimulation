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
    """
    
    def __init__(self, 
                 lane_id: int,
                 start_point: Point,
                 end_point: Point,
                 speed_limit: float = 30.0,  # m/s
                 lane_width: float = 3.5):   # meters
        """
        Initialize a lane.
        
        Args:
            lane_id: Unique identifier for the lane
            start_point: Starting point of the lane
            end_point: Ending point of the lane
            speed_limit: Speed limit for the lane (m/s)
            lane_width: Width of the lane (meters)
        """
        self.lane_id = lane_id
        self.start_point = start_point
        self.end_point = end_point
        self.speed_limit = speed_limit
        self.lane_width = lane_width
        
        # Calculate lane properties
        self.length = start_point.distance_to(end_point)
        self.direction = self._calculate_direction()
        
        # Adjacent lanes
        self.left_lane_id = None
        self.right_lane_id = None
        
        # Connected lanes (for intersections)
        self.connected_lanes: List[int] = []
    
    def _calculate_direction(self) -> Tuple[float, float]:
        """Calculate the direction vector of the lane."""
        dx = self.end_point.x - self.start_point.x
        dy = self.end_point.y - self.start_point.y
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return (0, 0)
        
        return (dx / length, dy / length)
    
    def get_position_at_distance(self, distance: float) -> Point:
        """
        Get the position along the lane at a given distance.
        
        Args:
            distance: Distance along the lane from start (meters)
            
        Returns:
            Point at the specified distance
        """
        # Clamp distance to lane length
        distance = max(0, min(distance, self.length))
        
        # Calculate position using linear interpolation
        t = distance / self.length if self.length > 0 else 0
        x = self.start_point.x + t * (self.end_point.x - self.start_point.x)
        y = self.start_point.y + t * (self.end_point.y - self.start_point.y)
        
        return Point(x, y)
    
    def get_distance_from_start(self, point: Point) -> float:
        """
        Get the distance along the lane from the start point.
        
        Args:
            point: Point to measure distance from
            
        Returns:
            Distance along the lane (meters)
        """
        # Project point onto lane direction
        lane_vector = Point(self.end_point.x - self.start_point.x, 
                          self.end_point.y - self.start_point.y)
        point_vector = Point(point.x - self.start_point.x, 
                           point.y - self.start_point.y)
        
        # Calculate dot product
        dot_product = lane_vector.x * point_vector.x + lane_vector.y * point_vector.y
        
        # Calculate distance
        distance = dot_product / self.length if self.length > 0 else 0
        
        return max(0, min(distance, self.length))
    
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
    
    def add_connected_lane(self, lane_id: int):
        """
        Add a connected lane (for intersections).
        
        Args:
            lane_id: ID of the connected lane
        """
        if lane_id not in self.connected_lanes:
            self.connected_lanes.append(lane_id)


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
        self.traffic_lights: Dict[int, str] = {}  # lane_id -> state
        self.traffic_light_timer = 0
        self.traffic_light_duration = 30  # seconds
    
    def add_lane(self, lane_id: int):
        """
        Add a lane that connects to this intersection.
        
        Args:
            lane_id: ID of the connecting lane
        """
        if lane_id not in self.connected_lanes:
            self.connected_lanes.append(lane_id)
            self.traffic_lights[lane_id] = 'red'  # Default to red
    
    def update_traffic_lights(self, dt: float):
        """
        Update traffic light states.
        
        Args:
            dt: Time step (seconds)
        """
        self.traffic_light_timer += dt
        
        if self.traffic_light_timer >= self.traffic_light_duration:
            # Cycle through traffic light states
            for lane_id in self.traffic_lights:
                if self.traffic_lights[lane_id] == 'red':
                    self.traffic_lights[lane_id] = 'green'
                elif self.traffic_lights[lane_id] == 'green':
                    self.traffic_lights[lane_id] = 'yellow'
                elif self.traffic_lights[lane_id] == 'yellow':
                    self.traffic_lights[lane_id] = 'red'
            
            self.traffic_light_timer = 0
    
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
    
    def get_lane_direction(self, lane_id: int) -> Tuple[float, float]:
        """
        Get the direction vector of a lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Tuple of (dx, dy) direction vector
        """
        lane = self.get_lane(lane_id)
        if lane:
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
            if i > 0:
                lane.left_lane_id = lane_ids[i - 1]
            if i < num_lanes - 1:
                lane.right_lane_id = lane_ids[i + 1]
        
        self.add_road(road)
        return road
