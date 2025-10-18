"""
OpenStreetMap Integration

This module provides functionality to load and parse OpenStreetMap data
for creating road networks in the traffic simulation.
"""

import osmnx as ox
import networkx as nx
from typing import List, Tuple, Dict, Optional
import numpy as np

from ..models.road_network import RoadNetwork, Point, Lane, Road


class OSMNetworkLoader:
    """
    Loads road networks from OpenStreetMap data.
    
    This class handles:
    - Downloading OSM data for a given area
    - Converting OSM data to our road network format
    - Creating lanes and intersections
    """
    
    def __init__(self):
        """Initialize the OSM network loader."""
        self.network_type = 'drive'  # Type of network to download
        self.simplify = True  # Simplify the network
    
    def load_network_from_place(self, 
                               place: str,
                               network_type: str = 'drive') -> RoadNetwork:
        """
        Load road network from a place name.
        
        Args:
            place: Place name (e.g., "Manhattan, New York, USA")
            network_type: Type of network ('drive', 'walk', 'bike')
            
        Returns:
            RoadNetwork object
        """
        print(f"Loading OSM network for: {place}")
        
        try:
            # Download the network
            G = ox.graph_from_place(place, network_type=network_type, simplify=self.simplify)
            
            # Convert to our road network format
            road_network = self._convert_osm_to_road_network(G)
            
            print(f"Loaded network with {len(road_network.all_lanes)} lanes")
            return road_network
            
        except Exception as e:
            print(f"Error loading OSM network: {e}")
            # Return empty network
            return RoadNetwork()
    
    def load_network_from_bbox(self, 
                              north: float, 
                              south: float, 
                              east: float, 
                              west: float,
                              network_type: str = 'drive') -> RoadNetwork:
        """
        Load road network from bounding box coordinates.
        
        Args:
            north: Northern boundary (latitude)
            south: Southern boundary (latitude)
            east: Eastern boundary (longitude)
            west: Western boundary (longitude)
            network_type: Type of network ('drive', 'walk', 'bike')
            
        Returns:
            RoadNetwork object
        """
        print(f"Loading OSM network for bbox: {north}, {south}, {east}, {west}")
        
        try:
            # Download the network
            G = ox.graph_from_bbox(north, south, east, west, network_type=network_type, simplify=self.simplify)
            
            # Convert to our road network format
            road_network = self._convert_osm_to_road_network(G)
            
            print(f"Loaded network with {len(road_network.all_lanes)} lanes")
            return road_network
            
        except Exception as e:
            print(f"Error loading OSM network: {e}")
            # Return empty network
            return RoadNetwork()
    
    def _convert_osm_to_road_network(self, G: nx.MultiDiGraph) -> RoadNetwork:
        """
        Convert OSM NetworkX graph to our RoadNetwork format.
        
        Args:
            G: NetworkX MultiDiGraph from OSM
            
        Returns:
            RoadNetwork object
        """
        road_network = RoadNetwork()
        
        # Get node and edge data
        nodes = list(G.nodes(data=True))
        edges = list(G.edges(data=True, keys=True))
        
        print(f"Converting {len(nodes)} nodes and {len(edges)} edges...")
        
        # Create roads from edges
        road_id = 0
        lane_id = 0
        
        for (u, v, key, data) in edges:
            # Get node coordinates
            u_node = dict(nodes)[u]
            v_node = dict(nodes)[v]
            
            # Convert coordinates to our Point format
            start_point = Point(u_node['x'], u_node['y'])
            end_point = Point(v_node['x'], v_node['y'])
            
            # Get road properties
            road_name = data.get('name', f'Road_{road_id}')
            maxspeed = self._parse_maxspeed(data.get('maxspeed', '50'))
            lanes = self._parse_lanes(data.get('lanes', '1'))
            
            # Create road
            road = Road(road_id, road_name)
            
            # Create lanes for this road
            for i in range(lanes):
                # Calculate lane offset
                lane_offset = self._calculate_lane_offset(i, lanes, data.get('width', 3.5))
                
                # Create lane start and end points with offset
                lane_start = self._offset_point(start_point, lane_offset, u_node, v_node)
                lane_end = self._offset_point(end_point, lane_offset, u_node, v_node)
                
                # Create lane
                lane = Lane(
                    lane_id=lane_id,
                    start_point=lane_start,
                    end_point=lane_end,
                    speed_limit=maxspeed
                )
                
                # Set adjacent lanes
                if i > 0:
                    lane.left_lane_id = lane_id - 1
                if i < lanes - 1:
                    lane.right_lane_id = lane_id + 1
                
                road.add_lane(lane)
                lane_id += 1
            
            road_network.add_road(road)
            road_id += 1
        
        return road_network
    
    def _parse_maxspeed(self, maxspeed_str: str) -> float:
        """
        Parse maxspeed string to float.
        
        Args:
            maxspeed_str: Maxspeed string from OSM
            
        Returns:
            Maxspeed in m/s
        """
        try:
            # Handle different formats
            if isinstance(maxspeed_str, (int, float)):
                return float(maxspeed_str)
            
            maxspeed_str = str(maxspeed_str).lower()
            
            # Remove common suffixes
            maxspeed_str = maxspeed_str.replace('mph', '').replace('km/h', '').replace('kph', '')
            
            # Extract number
            import re
            numbers = re.findall(r'\d+', maxspeed_str)
            if numbers:
                speed = float(numbers[0])
                
                # Convert to m/s if needed
                if 'mph' in str(maxspeed_str).lower():
                    speed = speed * 0.44704  # mph to m/s
                elif 'km' in str(maxspeed_str).lower():
                    speed = speed / 3.6  # km/h to m/s
                
                return speed
            
            return 30.0  # Default speed limit
            
        except:
            return 30.0  # Default speed limit
    
    def _parse_lanes(self, lanes_str: str) -> int:
        """
        Parse lanes string to integer.
        
        Args:
            lanes_str: Lanes string from OSM
            
        Returns:
            Number of lanes
        """
        try:
            if isinstance(lanes_str, (int, float)):
                return int(lanes_str)
            
            lanes_str = str(lanes_str)
            
            # Extract number
            import re
            numbers = re.findall(r'\d+', lanes_str)
            if numbers:
                return int(numbers[0])
            
            return 1  # Default to single lane
            
        except:
            return 1  # Default to single lane
    
    def _calculate_lane_offset(self, lane_index: int, total_lanes: int, road_width: float) -> float:
        """
        Calculate the offset for a lane.
        
        Args:
            lane_index: Index of the lane (0-based)
            total_lanes: Total number of lanes
            road_width: Width of the road
            
        Returns:
            Lane offset in meters
        """
        lane_width = 3.5  # Standard lane width
        total_width = total_lanes * lane_width
        
        # Calculate offset from center
        offset = (lane_index - (total_lanes - 1) / 2) * lane_width
        
        return offset
    
    def _offset_point(self, point: Point, offset: float, start_node: dict, end_node: dict) -> Point:
        """
        Calculate offset point perpendicular to the road direction.
        
        Args:
            point: Original point
            offset: Offset distance
            start_node: Start node data
            end_node: End node data
            
        Returns:
            Offset point
        """
        # Calculate road direction
        dx = end_node['x'] - start_node['x']
        dy = end_node['y'] - start_node['y']
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return point
        
        # Calculate perpendicular direction
        perp_x = -dy / length
        perp_y = dx / length
        
        # Apply offset
        offset_x = point.x + perp_x * offset
        offset_y = point.y + perp_y * offset
        
        return Point(offset_x, offset_y)


def create_sample_network() -> RoadNetwork:
    """
    Create a sample road network for testing.
    
    Returns:
        RoadNetwork object
    """
    road_network = RoadNetwork()
    
    # Create a simple intersection
    center = Point(0, 0)
    
    # Create roads
    road1 = Road(0, "Main Street")
    road2 = Road(1, "Cross Street")
    
    # Create lanes for road1 (horizontal)
    lane1 = Lane(0, Point(-100, -3.5), Point(100, -3.5))
    lane2 = Lane(1, Point(-100, 0), Point(100, 0))
    lane3 = Lane(2, Point(-100, 3.5), Point(100, 3.5))
    
    # Set adjacent lanes
    lane1.right_lane_id = 1
    lane2.left_lane_id = 0
    lane2.right_lane_id = 2
    lane3.left_lane_id = 1
    
    road1.add_lane(lane1)
    road1.add_lane(lane2)
    road1.add_lane(lane3)
    
    # Create lanes for road2 (vertical)
    lane4 = Lane(3, Point(-3.5, -100), Point(-3.5, 100))
    lane5 = Lane(4, Point(0, -100), Point(0, 100))
    lane6 = Lane(5, Point(3.5, -100), Point(3.5, 100))
    
    # Set adjacent lanes
    lane4.right_lane_id = 4
    lane5.left_lane_id = 3
    lane5.right_lane_id = 5
    lane6.left_lane_id = 4
    
    road2.add_lane(lane4)
    road2.add_lane(lane5)
    road2.add_lane(lane6)
    
    road_network.add_road(road1)
    road_network.add_road(road2)
    
    return road_network
