"""
Traffic Simulation Model

This module contains the main MESA Model class that manages the traffic simulation.
It coordinates vehicles, road networks, and simulation state.
"""

import numpy as np
import mesa
from mesa.space import AgentSet, ContinuousSpace
from typing import List, Dict, Optional, Tuple
from ..models.road_network import Intersection
import random

from ..agents.vehicle import Vehicle
from ..models.road_network import RoadNetwork, Point, Lane, Road, TrafficLights, TrafficLightsNode
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

                 vehicle_spawn_rate: float | Dict[int, float] = 0.5,  # vehicles per second (more frequent spawning)
                 max_vehicles: int = 150,
                 custom_lanes_path: Optional[str] = None,
                 model_boost: float = 10):
        """
        Initialize the traffic simulation model.
        
        Args:
            width: Width of the simulation space (pixels)
            height: Height of the simulation space (pixels)
            time_step: Time step for simulation (seconds)

            vehicle_spawn_rate: Rate of vehicle spawning (vehicles/second). Can be a float (global rate) or dict {lane_id: rate}.
            max_vehicles: Maximum number of vehicles in simulation
            custom_lanes_path: Path to JSON file with custom lanes (if None, uses default network)
            model_boot (float): How much faster the vehicles are moving than in reality.
        """
        super().__init__()
        
        # Simulation parameters
        self.width = width
        self.height = height
        self.time_step = time_step
        self.vehicle_spawn_rate = vehicle_spawn_rate
        self.max_vehicles = max_vehicles
        self.model_boost = model_boost
        
        # Default transformation parameters (will be overridden if custom lanes loaded)
        self.offset_x = 2064.0
        self.offset_y = 526.0
        self.scale = 0.3507
        
        # MESA components
        self.vehicle_agents = AgentSet([])
        self.space = ContinuousSpace(width, height, False)
        
        # Road network
        self.road_network = RoadNetwork()
        
        # Vehicle management
        self.vehicles: List[Vehicle] = []
        self.vehicle_counter = 0
        
        # Initialize spawn timers with staggered start times
        if isinstance(self.vehicle_spawn_rate, dict):
            # Per-lane spawning with staggered start times
            # Stagger by 0.3 seconds per lane to prevent simultaneous spawns on adjacent lanes
            self.spawn_timers = {}
            stagger_interval = 0.3  # seconds between lane spawn starts
            for idx, lane_id in enumerate(sorted(self.vehicle_spawn_rate.keys())):
                # Start each lane at a negative time (delay first spawn)
                self.spawn_timers[lane_id] = -idx * stagger_interval
            print(f"🚗 Initialized staggered spawn timers: {self.spawn_timers}")
            self.global_spawn_timer = None
        else:
            # Global spawning
            self.spawn_timers = None
            self.global_spawn_timer = 0.0
        self.current_time = 0.0
        
        # Image dimensions (will be set when loading custom lanes, defaults for test network)
        self.image_width = 3840  # Default, will be overridden if custom lanes loaded
        self.image_height = 2160  # Default, will be overridden if custom lanes loaded
        
        # Statistics
        self.stats = {
            'total_vehicles_spawned': 0,
            'total_vehicles_removed': 0,
            'average_speed': 0.0,
            'total_distance_traveled': 0.0,
            'collisions': 0
        }
        
        # Initialize road network
        if custom_lanes_path:
            self._create_custom_road_network(custom_lanes_path)
        else:
            self._create_test_road_network()
        
        # Spawn vehicles gradually over time
        self._spawn_initial_vehicles()
    
    def _get_intersection_for_lane(self, lane_id: int) -> Optional['Intersection']:
        """
        Get the intersection that a lane connects to.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Intersection object or None if not found
        """
        for intersection in self.road_network.intersections:
            if lane_id in intersection.connected_lanes:
                return intersection
        return None
    
    def _create_test_road_network(self):
        """Create Korean crossing layout matching the image:
        - Horizontal: 4 lanes (2 going left, 2 going right)
        - Vertical: 4 lanes (2 going top-to-bottom, 2 going bottom-to-top)
        """
        from ..models.road_network import Road, Lane, Intersection
        
        # Create intersection at origin
        center_point = Point(0, 0)
        intersection = Intersection(0, center_point)
        
        # Define road parameters
        road_length = 1500.0  # 1.5km in each direction
        lane_width = 3.5  # meters
        lane_spacing = lane_width  # Spacing between lanes
        
        # Lane IDs tracking
        lane_id_counter = 0
        
        # Dictionary to store lanes by direction and type
        lanes_dict = {
            'East': {'incoming': [], 'outgoing': []},
            'West': {'incoming': [], 'outgoing': []},
            'North': {'incoming': [], 'outgoing': []},
            'South': {'incoming': [], 'outgoing': []}
        }
        
        # === HORIZONTAL ROADS: 4 lanes total (2 East, 2 West) ===
        # East incoming (2 lanes going West)
        east_incoming_road = Road(len(self.road_network.roads), "East_Incoming")
        for lane_idx in range(2):
            lane_offset = (lane_idx - 1.0) * lane_spacing  # -1.0, 0.0 spacing
            start_point = Point(road_length, lane_offset)
            end_point = Point(0, lane_offset)
            
            lane = Lane(
                lane_id=lane_id_counter,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=lane_width
            )
            east_incoming_road.add_lane(lane)
            lanes_dict['East']['incoming'].append(lane_id_counter)
            intersection.add_lane(lane_id_counter)
            lane_id_counter += 1
        
        # Set adjacent lanes for East incoming
        for i, lane in enumerate(east_incoming_road.lanes):
            if i > 0:
                lane.left_lane_id = east_incoming_road.lanes[i-1].lane_id
            if i < len(east_incoming_road.lanes) - 1:
                lane.right_lane_id = east_incoming_road.lanes[i+1].lane_id
        
        self.road_network.add_road(east_incoming_road)
        
        # East outgoing (2 lanes going East)
        east_outgoing_road = Road(len(self.road_network.roads), "East_Outgoing")
        for lane_idx in range(2):
            lane_offset = (lane_idx - 1.0) * lane_spacing
            start_point = Point(0, lane_offset)
            end_point = Point(road_length, lane_offset)
            
            lane = Lane(
                lane_id=lane_id_counter,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=lane_width
            )
            east_outgoing_road.add_lane(lane)
            lanes_dict['East']['outgoing'].append(lane_id_counter)
            lane_id_counter += 1
        
        # Set adjacent lanes for East outgoing
        for i, lane in enumerate(east_outgoing_road.lanes):
            if i > 0:
                lane.left_lane_id = east_outgoing_road.lanes[i-1].lane_id
            if i < len(east_outgoing_road.lanes) - 1:
                lane.right_lane_id = east_outgoing_road.lanes[i+1].lane_id
        
        self.road_network.add_road(east_outgoing_road)
        
        # West incoming (2 lanes going East)
        west_incoming_road = Road(len(self.road_network.roads), "West_Incoming")
        for lane_idx in range(2):
            lane_offset = (lane_idx - 1.0) * lane_spacing
            start_point = Point(-road_length, lane_offset)
            end_point = Point(0, lane_offset)
            
            lane = Lane(
                lane_id=lane_id_counter,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=lane_width
            )
            west_incoming_road.add_lane(lane)
            lanes_dict['West']['incoming'].append(lane_id_counter)
            intersection.add_lane(lane_id_counter)
            lane_id_counter += 1
        
        # Set adjacent lanes for West incoming
        for i, lane in enumerate(west_incoming_road.lanes):
            if i > 0:
                lane.left_lane_id = west_incoming_road.lanes[i-1].lane_id
            if i < len(west_incoming_road.lanes) - 1:
                lane.right_lane_id = west_incoming_road.lanes[i+1].lane_id
        
        self.road_network.add_road(west_incoming_road)
        
        # West outgoing (2 lanes going West)
        west_outgoing_road = Road(len(self.road_network.roads), "West_Outgoing")
        for lane_idx in range(2):
            lane_offset = (lane_idx - 1.0) * lane_spacing
            start_point = Point(0, lane_offset)
            end_point = Point(-road_length, lane_offset)
            
            lane = Lane(
                lane_id=lane_id_counter,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=lane_width
            )
            west_outgoing_road.add_lane(lane)
            lanes_dict['West']['outgoing'].append(lane_id_counter)
            lane_id_counter += 1
        
        # Set adjacent lanes for West outgoing
        for i, lane in enumerate(west_outgoing_road.lanes):
            if i > 0:
                lane.left_lane_id = west_outgoing_road.lanes[i-1].lane_id
            if i < len(west_outgoing_road.lanes) - 1:
                lane.right_lane_id = west_outgoing_road.lanes[i+1].lane_id
        
        self.road_network.add_road(west_outgoing_road)
        
        # === VERTICAL ROADS: 4 lanes total (2 North, 2 South) ===
        # North incoming (2 lanes going South - top to bottom)
        north_incoming_road = Road(len(self.road_network.roads), "North_Incoming")
        for lane_idx in range(2):
            lane_offset = (lane_idx - 1.0) * lane_spacing
            start_point = Point(lane_offset, road_length)
            end_point = Point(lane_offset, 0)
            
            lane = Lane(
                lane_id=lane_id_counter,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=lane_width
            )
            north_incoming_road.add_lane(lane)
            lanes_dict['North']['incoming'].append(lane_id_counter)
            intersection.add_lane(lane_id_counter)
            lane_id_counter += 1
        
        # Set adjacent lanes for North incoming
        for i, lane in enumerate(north_incoming_road.lanes):
            if i > 0:
                lane.left_lane_id = north_incoming_road.lanes[i-1].lane_id
            if i < len(north_incoming_road.lanes) - 1:
                lane.right_lane_id = north_incoming_road.lanes[i+1].lane_id
        
        self.road_network.add_road(north_incoming_road)
        
        # North outgoing (2 lanes going North - bottom to top)
        north_outgoing_road = Road(len(self.road_network.roads), "North_Outgoing")
        for lane_idx in range(2):
            lane_offset = (lane_idx - 1.0) * lane_spacing
            start_point = Point(lane_offset, 0)
            end_point = Point(lane_offset, road_length)
            
            lane = Lane(
                lane_id=lane_id_counter,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=lane_width
            )
            north_outgoing_road.add_lane(lane)
            lanes_dict['North']['outgoing'].append(lane_id_counter)
            lane_id_counter += 1
        
        # Set adjacent lanes for North outgoing
        for i, lane in enumerate(north_outgoing_road.lanes):
            if i > 0:
                lane.left_lane_id = north_outgoing_road.lanes[i-1].lane_id
            if i < len(north_outgoing_road.lanes) - 1:
                lane.right_lane_id = north_outgoing_road.lanes[i+1].lane_id
        
        self.road_network.add_road(north_outgoing_road)
        
        # South incoming (2 lanes going North - bottom to top)
        south_incoming_road = Road(len(self.road_network.roads), "South_Incoming")
        for lane_idx in range(2):
            lane_offset = (lane_idx - 1.0) * lane_spacing
            start_point = Point(lane_offset, -road_length)
            end_point = Point(lane_offset, 0)
            
            lane = Lane(
                lane_id=lane_id_counter,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=lane_width
            )
            south_incoming_road.add_lane(lane)
            lanes_dict['South']['incoming'].append(lane_id_counter)
            intersection.add_lane(lane_id_counter)
            lane_id_counter += 1
        
        # Set adjacent lanes for South incoming
        for i, lane in enumerate(south_incoming_road.lanes):
            if i > 0:
                lane.left_lane_id = south_incoming_road.lanes[i-1].lane_id
            if i < len(south_incoming_road.lanes) - 1:
                lane.right_lane_id = south_incoming_road.lanes[i+1].lane_id
        
        self.road_network.add_road(south_incoming_road)
        
        # South outgoing (2 lanes going South - top to bottom)
        south_outgoing_road = Road(len(self.road_network.roads), "South_Outgoing")
        for lane_idx in range(2):
            lane_offset = (lane_idx - 1.0) * lane_spacing
            start_point = Point(lane_offset, 0)
            end_point = Point(lane_offset, -road_length)
            
            lane = Lane(
                lane_id=lane_id_counter,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=lane_width
            )
            south_outgoing_road.add_lane(lane)
            lanes_dict['South']['outgoing'].append(lane_id_counter)
            lane_id_counter += 1
        
        # Set adjacent lanes for South outgoing
        for i, lane in enumerate(south_outgoing_road.lanes):
            if i > 0:
                lane.left_lane_id = south_outgoing_road.lanes[i-1].lane_id
            if i < len(south_outgoing_road.lanes) - 1:
                lane.right_lane_id = south_outgoing_road.lanes[i+1].lane_id
        
        self.road_network.add_road(south_outgoing_road)
        
        # === CONNECT LANES AT INTERSECTION ===
        # East incoming -> West outgoing (straight)
        for i, east_in_lane_id in enumerate(lanes_dict['East']['incoming']):
            east_lane = self.road_network.get_lane(east_in_lane_id)
            if east_lane:
                west_out_lane_id = lanes_dict['West']['outgoing'][i]
                east_lane.add_connected_lane(west_out_lane_id, 'straight')
        
        # West incoming -> East outgoing (straight)
        for i, west_in_lane_id in enumerate(lanes_dict['West']['incoming']):
            west_lane = self.road_network.get_lane(west_in_lane_id)
            if west_lane:
                east_out_lane_id = lanes_dict['East']['outgoing'][i]
                west_lane.add_connected_lane(east_out_lane_id, 'straight')
        
        # North incoming -> South outgoing (straight)
        for i, north_in_lane_id in enumerate(lanes_dict['North']['incoming']):
            north_lane = self.road_network.get_lane(north_in_lane_id)
            if north_lane:
                south_out_lane_id = lanes_dict['South']['outgoing'][i]
                north_lane.add_connected_lane(south_out_lane_id, 'straight')
        
        # South incoming -> North outgoing (straight)
        for i, south_in_lane_id in enumerate(lanes_dict['South']['incoming']):
            south_lane = self.road_network.get_lane(south_in_lane_id)
            if south_lane:
                north_out_lane_id = lanes_dict['North']['outgoing'][i]
                south_lane.add_connected_lane(north_out_lane_id, 'straight')
        
        # === SET UP TRAFFIC LIGHT PHASES ===
        # Phase 1: East-West green (horizontal)
        phase1_duration = 25.0
        phase1_states = {}
        for lane_id in lanes_dict['East']['incoming'] + lanes_dict['West']['incoming']:
            phase1_states[lane_id] = 'green'
        for lane_id in lanes_dict['North']['incoming'] + lanes_dict['South']['incoming']:
            phase1_states[lane_id] = 'red'
        
        # Phase 2: Yellow for East-West
        phase2_duration = 3.0
        phase2_states = {}
        for lane_id in lanes_dict['East']['incoming'] + lanes_dict['West']['incoming']:
            phase2_states[lane_id] = 'yellow'
        for lane_id in lanes_dict['North']['incoming'] + lanes_dict['South']['incoming']:
            phase2_states[lane_id] = 'red'
        
        # Phase 3: North-South green (vertical)
        phase3_duration = 20.0
        phase3_states = {}
        for lane_id in lanes_dict['East']['incoming'] + lanes_dict['West']['incoming']:
            phase3_states[lane_id] = 'red'
        for lane_id in lanes_dict['North']['incoming'] + lanes_dict['South']['incoming']:
            phase3_states[lane_id] = 'green'
        
        # Phase 4: Yellow for North-South
        phase4_duration = 3.0
        phase4_states = {}
        for lane_id in lanes_dict['East']['incoming'] + lanes_dict['West']['incoming']:
            phase4_states[lane_id] = 'red'
        for lane_id in lanes_dict['North']['incoming'] + lanes_dict['South']['incoming']:
            phase4_states[lane_id] = 'yellow'
        
        intersection.set_traffic_light_phases([
            (phase1_duration, phase1_states),
            (phase2_duration, phase2_states),
            (phase3_duration, phase3_states),
            (phase4_duration, phase4_states)
        ])
        
        # Add intersection to network
        self.road_network.add_intersection(intersection)
        
        # Initialize TrafficLightsNode for test network (empty list of lights for now as they are managed by Intersection)
        self.traffic_lights_node = TrafficLightsNode(
            [],
            int(7 / self.time_step),
            int(1 / self.time_step),
            int(3 / self.time_step)
        )
        
        print(f"Created Korean crossing layout with {len(self.road_network.all_lanes)} lanes")
        print(f"Number of roads: {len(self.road_network.roads)}")
        print(f"Number of intersections: {len(self.road_network.intersections)}")
        
        # Print lane information
        for lane_id, lane in self.road_network.all_lanes.items():
            connected = f", connected to: {lane.connected_lanes}" if lane.connected_lanes else ""
            route_types = f", routes: {lane.route_types}" if lane.route_types else ""
            print(f"Lane {lane_id}: {lane.start_point.x:.0f},{lane.start_point.y:.0f} -> {lane.end_point.x:.0f},{lane.end_point.y:.0f} (length: {lane.length:.0f}m){connected}{route_types}")
    
    def _create_custom_road_network(self, lanes_json_path: str):
        """Create road network from custom marked lanes."""
        import json
        from pathlib import Path
        from ..models.road_network import Road, Lane, Point
        
        # Load lane data
        with open(lanes_json_path) as f:
            lane_data = json.load(f)
        
        # Store lanes_data for later use (checking manual spawn points)
        self.lanes_data = lane_data
        
        # Store image dimensions for edge detection
        self.image_width = lane_data.get('image_width', 3840)
        self.image_height = lane_data.get('image_height', 2160)
        
        # Store image points for each lane (for spawn validation)
        self.lane_image_points: Dict[int, List[Tuple[int, int]]] = {}
        
        # Track which lanes have manual spawn points
        self.manual_spawn_lanes: set = set()
        
        # Transformation parameters
        # Use the original offset values to ensure vehicles spawn at correct positions
        # The offset is necessary to properly align image coordinates with world coordinates
        offset_x = lane_data.get('offset_x', 2064.0)  # ROI center X
        self.offset_x = offset_x
        offset_y = lane_data.get('offset_y', 526.0)   # ROI center Y
        self.offset_y = offset_y
        scale = lane_data.get('scale', 11)  # pixels per meter
        self.scale = scale
        
        def image_to_world(img_x: float, img_y: float) -> Point:
            """Convert image coordinates to world coordinates."""
            world_x = (img_x - offset_x) / scale
            world_y = (offset_y - img_y) / scale  # Flip Y-axis
            return Point(world_x, world_y)
        
        def world_to_image(world_x: float, world_y: float) -> Tuple[float, float]:
            """Convert world coordinates to image coordinates."""
            img_x = world_x * scale + offset_x
            img_y = offset_y - world_y * scale  # Flip Y-axis
            return (img_x, img_y)
        
        lanes_list = lane_data['lanes']
        road_id = 0
        
        print(f"Creating custom road network from {len(lanes_list)} marked lanes...")
        created_traffic_lights = []

        # First, create all regular lanes
        for lane_info in lanes_list:
            lane_id = lane_info['lane_id']
            
            # Use centerline_points if available, otherwise use points
            if 'centerline_points' in lane_info and len(lane_info['centerline_points']) > 0:
                points = lane_info['centerline_points']
            else:
                points = lane_info.get('points', [])
            
            if len(points) < 2:
                print(f"⚠ Skipping lane {lane_id}: needs at least 2 points")
                continue
            
            # Convert points to world coordinates
            centerline_world_points = []
            for px, py in points:
                centerline_world_points.append(image_to_world(px, py))
            
            # Determine start and end points (first and last centerline points)
            start_point = centerline_world_points[0]
            end_point = centerline_world_points[-1]

            # Prepare traffic lights
            traffic_lights_info = lane_info.get('traffic_lights', [])

            lane_traffic_lights = []
            for lights_instance_info in traffic_lights_info:
                # Convert points to world coordinates
                px, py = lights_instance_info["position"]
                lights = TrafficLights(image_to_world(px, py), lights_instance_info["green_stages"])
                lane_traffic_lights.append(lights)

            created_traffic_lights.extend(lane_traffic_lights)
            # Create lane with centerline points
            lane = Lane(
                lane_id=lane_id,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=3.5,
                centerline_points=centerline_world_points,
                traffic_lights=lane_traffic_lights
            )
            
            # Check if spawn is enabled for this lane (spawn at first point)
            spawn_enabled = lane_info.get('spawn_enabled', False)
            if spawn_enabled and centerline_world_points and len(centerline_world_points) > 0:
                # Use first point of centerline as spawn position
                first_point_world = centerline_world_points[0]
                lane.spawn_position = first_point_world
                # Calculate spawn direction from first to second point (if available)
                if len(centerline_world_points) >= 2:
                    second_point_world = centerline_world_points[1]
                    dx = second_point_world.x - first_point_world.x
                    dy = second_point_world.y - first_point_world.y
                    length = np.sqrt(dx**2 + dy**2)
                    if length > 0:
                        lane.spawn_direction = (dx / length, dy / length)
                    else:
                        lane.spawn_direction = lane.direction
                else:
                    lane.spawn_direction = lane.direction
                self.manual_spawn_lanes.add(lane_id)
                first_img_x, first_img_y = world_to_image(first_point_world.x, first_point_world.y)
                print(f"  ✓ Spawn enabled for lane {lane_id} at first centerline point: ({first_img_x:.0f}, {first_img_y:.0f})")
            
            # Store original image points for edge detection
            self.lane_image_points[lane_id] = points
            
            # Load adjacent lane connections from JSON if available
            # These take precedence over automatic detection
            left_lane_id = lane_info.get('left_lane_id')
            right_lane_id = lane_info.get('right_lane_id')
            if left_lane_id is not None or right_lane_id is not None:
                lane.set_adjacent_lanes(left_lane_id, right_lane_id)
                if left_lane_id is not None:
                    print(f"  ✓ Lane {lane_id} manually set LEFT to lane {left_lane_id} (from JSON)")
                if right_lane_id is not None:
                    print(f"  ✓ Lane {lane_id} manually set RIGHT to lane {right_lane_id} (from JSON)")
            
            road = Road(road_id, f"Custom_Lane_{lane_id}")
            road.add_lane(lane)
            self.road_network.add_road(road)
            
            # Print spawn information if available
            spawn_info = ""
            if lane.spawn_position:
                spawn_info = f", spawn at ({lane.spawn_position.x:.1f}, {lane.spawn_position.y:.1f})"
            
            print(f"  Created lane {lane_id}: ({start_point.x:.1f}, {start_point.y:.1f}) -> ({end_point.x:.1f}, {end_point.y:.1f}){spawn_info}")
            
            road_id += 1
        
        # Create TrafficLightsNode
        # Use 7s stage length and 1s yellow, plus a 1s all-red buffer between stages
        self.traffic_lights_node = TrafficLightsNode(
            created_traffic_lights,
            int(7 / self.time_step),            # steps per stage (~7 seconds)
            int(1 / self.time_step),            # yellow/indication length (~1 second)
            interstage_buffer_steps=int(3 / self.time_step)  # all-red buffer (~3 seconds)
        )

        # Create lane-changing connection lanes
        self._create_lane_change_connections(lane_data, image_to_world, road_id)
        
        # After all lanes are created, detect turning lanes (lanes that physically connect)
        print("\nDetecting turning lanes (physically connected lanes)...")
        self._detect_turning_lanes()
        
        # Automatically detect and set adjacent lanes from lane connections
        print("\nDetecting adjacent lanes from lane connections...")
        self._detect_adjacent_lanes_from_connections(lane_data)
        
        print(f"✅ Created custom road network with {len(self.road_network.all_lanes)} lanes")
    
    def _create_lane_change_connections(self, lane_data: Dict, image_to_world, start_road_id: int):
        """Create lanes for lane-changing connections."""
        connections = lane_data.get('lane_connections', [])
        if not connections:
            return
        
        print(f"\nCreating {len(connections)} lane-changing connection(s)...")
        road_id = start_road_id
        
        for conn in connections:
            source_lane_id = conn['source_lane_id']
            target_lane_id = conn['target_lane_id']
            connection_points = conn.get('points', [])
            
            if len(connection_points) < 2:
                print(f"⚠ Skipping connection {source_lane_id} -> {target_lane_id}: needs at least 2 points")
                continue
            
            # Get source and target lanes
            source_lane = self.road_network.get_lane(source_lane_id)
            target_lane = self.road_network.get_lane(target_lane_id)
            
            if not source_lane or not target_lane:
                print(f"⚠ Skipping connection {source_lane_id} -> {target_lane_id}: lane not found")
                continue
            
            # Convert connection points to world coordinates
            connection_world_points = []
            for px, py in connection_points:
                world_point = image_to_world(px, py)
                connection_world_points.append(world_point)
            
            # Use interpolation to find which lane each point belongs to and snap to nearest lane
            # For high-resolution images (5k x 3k), use more lenient tolerance
            # 10 meters ≈ 3.5 pixels, which is reasonable for point matching
            tolerance = 10.0  # meters - maximum distance to consider point as "on" a lane
            
            # Process first point - should be on source lane
            if connection_world_points:
                first_point = connection_world_points[0]
                # Check if point lies on source lane using interpolation
                if source_lane.point_lies_on_lane(first_point, tolerance):
                    # Point is on source lane, snap it
                    snapped_first = source_lane.snap_point_to_centerline(first_point)
                    connection_world_points[0] = snapped_first
                else:
                    # Point is not exactly on source lane, find nearest lane and snap
                    nearest_lane_id = self.road_network.find_lane_for_point(first_point, tolerance)
                    if nearest_lane_id is not None:
                        nearest_lane = self.road_network.get_lane(nearest_lane_id)
                        if nearest_lane:
                            snapped_first = nearest_lane.snap_point_to_centerline(first_point)
                            connection_world_points[0] = snapped_first
                            print(f"  ⚠ Adjusted first connection point to snap to lane {nearest_lane_id} (was {source_lane_id})")
                    else:
                        # Fallback: snap to source lane anyway
                        snapped_first = source_lane.snap_point_to_centerline(first_point)
                        connection_world_points[0] = snapped_first
                
                # Process last point - should be on target lane
                last_point = connection_world_points[-1]
                # Check if point lies on target lane using interpolation
                if target_lane.point_lies_on_lane(last_point, tolerance):
                    # Point is on target lane, snap it
                    snapped_last = target_lane.snap_point_to_centerline(last_point)
                    connection_world_points[-1] = snapped_last
                else:
                    # Point is not exactly on target lane, find nearest lane and snap
                    nearest_lane_id = self.road_network.find_lane_for_point(last_point, tolerance)
                    if nearest_lane_id is not None:
                        nearest_lane = self.road_network.get_lane(nearest_lane_id)
                        if nearest_lane:
                            snapped_last = nearest_lane.snap_point_to_centerline(last_point)
                            connection_world_points[-1] = snapped_last
                            print(f"  ⚠ Adjusted last connection point to snap to lane {nearest_lane_id} (was {target_lane_id})")
                    else:
                        # Fallback: snap to target lane anyway
                        snapped_last = target_lane.snap_point_to_centerline(last_point)
                        connection_world_points[-1] = snapped_last
            
            # Also snap intermediate points that are close to lanes using interpolation
            for i in range(1, len(connection_world_points) - 1):
                point = connection_world_points[i]
                
                # Check which lane this point belongs to using interpolation
                nearest_lane_id = self.road_network.find_lane_for_point(point, tolerance)
                
                if nearest_lane_id is not None:
                    nearest_lane = self.road_network.get_lane(nearest_lane_id)
                    if nearest_lane:
                        # Snap to the nearest lane
                        snapped_point = nearest_lane.snap_point_to_centerline(point)
                        connection_world_points[i] = snapped_point
                else:
                    # If not on any lane, check distance to source and target lanes
                    dist_to_source = point.distance_to(source_lane.snap_point_to_centerline(point))
                    dist_to_target = point.distance_to(target_lane.snap_point_to_centerline(point))
                    
                    # If point is very close to a lane, snap it to that lane
                    snap_threshold = 5.0  # meters
                    if dist_to_source < snap_threshold and dist_to_source < dist_to_target:
                        connection_world_points[i] = source_lane.snap_point_to_centerline(point)
                    elif dist_to_target < snap_threshold:
                        connection_world_points[i] = target_lane.snap_point_to_centerline(point)
            
            start_point = connection_world_points[0]
            end_point = connection_world_points[-1]
            
            # Create a connection lane (for lane-changing)
            # Use a special lane ID (negative or high number to avoid conflicts)
            connection_lane_id = 10000 + road_id  # Use high ID to avoid conflicts
            
            connection_lane = Lane(
                lane_id=connection_lane_id,
                start_point=start_point,
                end_point=end_point,
                speed_limit=30.0,
                lane_width=3.5,
                centerline_points=connection_world_points
            )
            
            # Store connection metadata
            connection_lane.source_lane_id = source_lane_id
            connection_lane.target_lane_id = target_lane_id
            connection_lane.is_lane_change_connection = True
            
            road = Road(road_id, f"LaneChange_{source_lane_id}_to_{target_lane_id}")
            road.add_lane(connection_lane)
            self.road_network.add_road(road)
            
            # Add connection to source lane's connected lanes
            source_lane.add_connected_lane(connection_lane_id, 'lane_change')
            
            # Also add direct connection from source to target lane (for automatic transition)
            if target_lane_id not in source_lane.connected_lanes:
                source_lane.add_connected_lane(target_lane_id, 'lane_change')
            
            print(f"  ✓ Created lane-changing connection: Lane {source_lane_id} -> Lane {target_lane_id} (connection lane {connection_lane_id})")
            
            road_id += 1
    
    def _detect_turning_lanes(self):
        """
        Automatically detect and connect turning lanes that physically connect at endpoints.
        Access lanes (that just end) remain unconnected.
        """
        connection_threshold = 5.0  # meters - max distance to consider lanes as connecting
        connections_made = 0
        
        # Check all pairs of lanes
        lane_ids = list(self.road_network.all_lanes.keys())
        for i, lane_id_a in enumerate(lane_ids):
            lane_a = self.road_network.get_lane(lane_id_a)
            if not lane_a:
                continue
            
            for lane_id_b in lane_ids[i+1:]:
                lane_b = self.road_network.get_lane(lane_id_b)
                if not lane_b:
                    continue
                
                # Check if lane A's end connects to lane B's start (A -> B turning lane)
                if lane_a.lanes_connect_at_endpoints(lane_b, connection_threshold):
                    # Determine route type based on angle between lanes
                    route_type = self._determine_route_type(lane_a, lane_b)
                    
                    # Connect lane A to lane B
                    if lane_id_b not in lane_a.connected_lanes:
                        lane_a.add_connected_lane(lane_id_b, route_type)
                        connections_made += 1
                        print(f"  Connected lane {lane_id_a} -> lane {lane_id_b} ({route_type} turn)")
                
                # Check if lane B's end connects to lane A's start (B -> A turning lane)
                if lane_b.lanes_connect_at_endpoints(lane_a, connection_threshold):
                    # Determine route type based on angle between lanes
                    route_type = self._determine_route_type(lane_b, lane_a)
                    
                    # Connect lane B to lane A
                    if lane_id_a not in lane_b.connected_lanes:
                        lane_b.add_connected_lane(lane_id_a, route_type)
                        connections_made += 1
                        print(f"  Connected lane {lane_id_b} -> lane {lane_id_a} ({route_type} turn)")
        
        if connections_made > 0:
            print(f"✅ Automatically connected {connections_made} turning lane(s)")
        else:
            print("  No turning lanes detected (all lanes are access lanes)")
    
    def _detect_adjacent_lanes_from_connections(self, lane_data: Dict):
        """
        Automatically detect adjacent lanes from lane_connections.
        If two lanes have connection points between them, they are likely adjacent.
        Uses interpolation to verify that connection points lie on the lanes.
        
        Args:
            lane_data: Dictionary containing lane data from JSON
        """
        connections = lane_data.get('lane_connections', [])
        if not connections:
            print("  No lane connections found for adjacent lane detection")
            return
        
        # Build a mapping of which lanes connect to which
        # Key: (source_lane_id, target_lane_id), Value: count of connections
        # Also track bidirectional connections (A->B and B->A) which strongly indicate adjacency
        lane_pairs = {}
        bidirectional_pairs = set()  # Track pairs that have connections in both directions
        
        # First pass: collect all connections
        for conn in connections:
            source_lane_id = conn['source_lane_id']
            target_lane_id = conn['target_lane_id']
            connection_points = conn.get('points', [])
            
            if len(connection_points) < 2:
                continue
            
            pair_key = (source_lane_id, target_lane_id)
            if pair_key not in lane_pairs:
                lane_pairs[pair_key] = []
            lane_pairs[pair_key].extend(connection_points)
        
        # Second pass: detect bidirectional connections
        for pair_key in lane_pairs.keys():
            source_lane_id, target_lane_id = pair_key
            reverse_key = (target_lane_id, source_lane_id)
            if reverse_key in lane_pairs:
                bidirectional_pairs.add(tuple(sorted([source_lane_id, target_lane_id])))
        
        print(f"  Found {len(bidirectional_pairs)} bidirectional lane pairs (strong candidates for adjacency)")
        
        # Transformation parameters (same as in _create_custom_road_network)
        offset_x = lane_data.get('offset_x', 2064.0)
        offset_y = lane_data.get('offset_y', 526.0)
        scale = lane_data.get('scale', 0.3507)
        
        def image_to_world(img_x: float, img_y: float) -> Point:
            """Convert image coordinates to world coordinates."""
            world_x = (img_x - offset_x) / scale
            world_y = (offset_y - img_y) / scale
            return Point(world_x, world_y)
        
        # For each pair of lanes that have connections, check if they should be adjacent
        # For high-resolution images, use more lenient tolerance
        tolerance = 10.0  # meters - maximum distance to consider lanes as adjacent
        adjacent_lanes_detected = 0
        
        print(f"  Checking {len(lane_pairs)} lane pairs for adjacent relationships...")
        
        for (source_lane_id, target_lane_id), points in lane_pairs.items():
            source_lane = self.road_network.get_lane(source_lane_id)
            target_lane = self.road_network.get_lane(target_lane_id)
            
            if not source_lane or not target_lane:
                print(f"  ⚠ Skipping pair ({source_lane_id}, {target_lane_id}): lane not found")
                continue
            
            # Check if lanes are parallel (similar direction)
            source_dir = source_lane.direction
            target_dir = target_lane.direction
            dot_product = source_dir[0] * target_dir[0] + source_dir[1] * target_dir[1]
            
            print(f"  Checking lanes {source_lane_id} <-> {target_lane_id}: dot_product={dot_product:.3f}, parallel={dot_product >= 0.7}")
            
            # Lanes must be nearly parallel (dot product > 0.7)
            if dot_product < 0.7:
                print(f"    ❌ Not parallel enough (dot={dot_product:.3f} < 0.7)")
                continue  # Not parallel enough to be adjacent
            
            # Check if this is a bidirectional connection (strong indicator of adjacency)
            is_bidirectional = tuple(sorted([source_lane_id, target_lane_id])) in bidirectional_pairs
            
            # Check average distance between lanes by sampling connection points
            total_distance = 0.0
            valid_points = 0
            
            for px, py in points[:5]:  # Sample first 5 points
                world_point = image_to_world(px, py)
                
                # Find closest points on both lanes
                source_closest = source_lane.snap_point_to_centerline(world_point)
                target_closest = target_lane.snap_point_to_centerline(world_point)
                
                # Calculate distance between lanes at this point
                dist = source_closest.distance_to(target_closest)
                
                # For bidirectional connections, be more lenient (points don't need to be exactly on lanes)
                if is_bidirectional:
                    # Accept if distance is reasonable (adjacent lanes are typically 3.5-7m apart)
                    if dist < 15.0:  # More lenient for bidirectional
                        total_distance += dist
                        valid_points += 1
                else:
                    # For unidirectional, check if point lies on either lane
                    on_source = source_lane.point_lies_on_lane(world_point, tolerance)
                    on_target = target_lane.point_lies_on_lane(world_point, tolerance)
                    
                    if (on_source or on_target) and dist < 10.0:
                        total_distance += dist
                        valid_points += 1
            
            if valid_points == 0:
                print(f"    ❌ No valid points found (points checked: {len(points[:5])}, bidirectional={is_bidirectional})")
                continue
            
            avg_distance = total_distance / valid_points
            print(f"    Avg distance: {avg_distance:.2f}m, valid_points: {valid_points}, bidirectional={is_bidirectional}")
            
            # If average distance is reasonable (typical lane width is 3.5m, so adjacent lanes should be ~3.5-7m apart)
            # For bidirectional connections, be more lenient with distance range
            max_distance = 15.0 if is_bidirectional else 10.0
            min_distance = 1.0 if is_bidirectional else 2.0
            
            if min_distance <= avg_distance <= max_distance:
                # Determine which lane is left and which is right
                # Use cross product to determine relative position
                # For parallel lanes going same direction, cross product determines left/right
                
                # Get a sample point from each lane (use multiple points for better accuracy)
                # Use points along the lane to determine spatial relationship
                source_positions = [
                    source_lane.get_position_at_distance(source_lane.length * 0.25),
                    source_lane.get_position_at_distance(source_lane.length * 0.5),
                    source_lane.get_position_at_distance(source_lane.length * 0.75)
                ]
                target_positions = [
                    target_lane.get_position_at_distance(target_lane.length * 0.25),
                    target_lane.get_position_at_distance(target_lane.length * 0.5),
                    target_lane.get_position_at_distance(target_lane.length * 0.75)
                ]
                
                # Calculate average cross product to determine left/right more reliably
                cross_sum = 0.0
                for i in range(len(source_positions)):
                    source_pos = source_positions[i]
                    target_pos = target_positions[i]
                    
                    # Vector from source to target
                    vec_to_target = Point(target_pos.x - source_pos.x, target_pos.y - source_pos.y)
                    
                    # Cross product: dir × vec_to_target
                    # In 2D: cross = dir.x * vec.y - dir.y * vec.x
                    # If cross > 0: target is to the LEFT of the direction of travel
                    # If cross < 0: target is to the RIGHT of the direction of travel
                    cross = source_dir[0] * vec_to_target.y - source_dir[1] * vec_to_target.x
                    cross_sum += cross
                
                avg_cross = cross_sum / len(source_positions)
                
                print(f"    Cross product: {avg_cross:.3f} (for lanes {source_lane_id} <-> {target_lane_id})")
                
                # CRITICAL: Only set adjacent lanes if they haven't been manually set from JSON
                # This prevents overwriting correct manual assignments
                if avg_cross > 0:
                    # Target is to the LEFT of source (relative to direction of travel)
                    # BUT: In standard coordinate systems, we need to verify this is correct
                    # For now, check if already set manually - if so, don't overwrite
                    if source_lane.left_lane_id is None and target_lane.right_lane_id is None:
                        source_lane.left_lane_id = target_lane_id
                        target_lane.right_lane_id = source_lane_id
                        adjacent_lanes_detected += 1
                        print(f"  ✓ Detected adjacent lanes: {source_lane_id} (left) <-> {target_lane_id} (right) [cross={avg_cross:.3f}]")
                    elif source_lane.left_lane_id is not None:
                        print(f"  ⏭️  Skipping - lane {source_lane_id} already has left_lane_id={source_lane.left_lane_id} (manual from JSON?)")
                    elif target_lane.right_lane_id is not None:
                        print(f"  ⏭️  Skipping - lane {target_lane_id} already has right_lane_id={target_lane.right_lane_id} (manual from JSON?)")
                else:
                    # Target is to the RIGHT of source (relative to direction of travel)
                    if source_lane.right_lane_id is None and target_lane.left_lane_id is None:
                        source_lane.right_lane_id = target_lane_id
                        target_lane.left_lane_id = source_lane_id
                        adjacent_lanes_detected += 1
                        print(f"  ✓ Detected adjacent lanes: {source_lane_id} (right) <-> {target_lane_id} (left) [cross={avg_cross:.3f}]")
                    elif source_lane.right_lane_id is not None:
                        print(f"  ⏭️  Skipping - lane {source_lane_id} already has right_lane_id={source_lane.right_lane_id} (manual from JSON?)")
                    elif target_lane.left_lane_id is not None:
                        print(f"  ⏭️  Skipping - lane {target_lane_id} already has left_lane_id={target_lane.left_lane_id} (manual from JSON?)")
        
        if adjacent_lanes_detected > 0:
            print(f"✅ Detected {adjacent_lanes_detected} adjacent lane relationship(s)")
        else:
            print("  ❌ No adjacent lanes detected from connections")
            print("  This might be because:")
            print("    - Lanes are not parallel enough (need dot_product > 0.7)")
            print("    - Average distance between lanes is not in range 2-10m")
            print("    - Connection points don't lie on the lanes")
            
            # Fallback: For bidirectional pairs, assume they're adjacent if they have many connections
            print(f"\n  Trying fallback: Setting bidirectional pairs as adjacent...")
            fallback_count = 0
            for lane_pair in bidirectional_pairs:
                lane_a_id, lane_b_id = lane_pair
                lane_a = self.road_network.get_lane(lane_a_id)
                lane_b = self.road_network.get_lane(lane_b_id)
                
                if not lane_a or not lane_b:
                    continue
                
                # Check if they're at least somewhat parallel (relaxed requirement)
                source_dir = lane_a.direction
                target_dir = lane_b.direction
                dot_product = source_dir[0] * target_dir[0] + source_dir[1] * target_dir[1]
                
                # More lenient: only require > 0.5 (approximately same general direction)
                if dot_product > 0.5:
                    # Determine left/right using cross product (use multiple points for accuracy)
                    source_positions = [
                        lane_a.get_position_at_distance(lane_a.length * 0.25),
                        lane_a.get_position_at_distance(lane_a.length * 0.5),
                        lane_a.get_position_at_distance(lane_a.length * 0.75)
                    ]
                    target_positions = [
                        lane_b.get_position_at_distance(lane_b.length * 0.25),
                        lane_b.get_position_at_distance(lane_b.length * 0.5),
                        lane_b.get_position_at_distance(lane_b.length * 0.75)
                    ]
                    
                    cross_sum = 0.0
                    for i in range(len(source_positions)):
                        vec_to_target = Point(target_positions[i].x - source_positions[i].x, 
                                             target_positions[i].y - source_positions[i].y)
                        cross = source_dir[0] * vec_to_target.y - source_dir[1] * vec_to_target.x
                        cross_sum += cross
                    
                    avg_cross = cross_sum / len(source_positions)
                    
                    # Only set if not already manually set
                    if avg_cross > 0:
                        # Target is to the LEFT of source
                        if lane_a.left_lane_id is None and lane_b.right_lane_id is None:
                            lane_a.left_lane_id = lane_b_id
                            lane_b.right_lane_id = lane_a_id
                            fallback_count += 1
                            print(f"    ✓ Fallback: Set lanes {lane_a_id} (left) <-> {lane_b_id} (right) [dot={dot_product:.3f}, cross={avg_cross:.3f}]")
                        else:
                            print(f"    ⏭️  Fallback skipped - already set manually (lane {lane_a_id}.left={lane_a.left_lane_id}, lane {lane_b_id}.right={lane_b.right_lane_id})")
                    else:
                        # Target is to the RIGHT of source
                        if lane_a.right_lane_id is None and lane_b.left_lane_id is None:
                            lane_a.right_lane_id = lane_b_id
                            lane_b.left_lane_id = lane_a_id
                            fallback_count += 1
                            print(f"    ✓ Fallback: Set lanes {lane_a_id} (right) <-> {lane_b_id} (left) [dot={dot_product:.3f}, cross={avg_cross:.3f}]")
                        else:
                            print(f"    ⏭️  Fallback skipped - already set manually (lane {lane_a_id}.right={lane_a.right_lane_id}, lane {lane_b_id}.left={lane_b.left_lane_id})")
                else:
                    print(f"    ⚠ Skipping fallback for {lane_a_id}<->{lane_b_id}: not parallel enough (dot={dot_product:.3f})")
            
            if fallback_count > 0:
                print(f"  ✅ Fallback set {fallback_count} adjacent lane relationship(s)")
            else:
                print(f"  ❌ Fallback also failed - no adjacent lanes set")
    
    def _determine_route_type(self, from_lane, to_lane) -> str:
        """
        Determine the route type (straight, left, right) based on the angle between lanes.
        
        Args:
            from_lane: Source lane
            to_lane: Target lane
            
        Returns:
            Route type: 'straight', 'left', or 'right'
        """
        # Calculate angle between lane directions
        from_dir = from_lane.direction
        to_dir = to_lane.direction
        
        # Calculate angle using dot product
        dot_product = from_dir[0] * to_dir[0] + from_dir[1] * to_dir[1]
        # Clamp to [-1, 1] for acos
        dot_product = max(-1.0, min(1.0, dot_product))
        angle = np.arccos(dot_product)
        
        # Calculate cross product to determine left/right
        cross_product = from_dir[0] * to_dir[1] - from_dir[1] * to_dir[0]
        
        # Convert angle to degrees for easier thresholding
        angle_deg = np.degrees(angle)
        
        # Straight: angle < 30 degrees
        if angle_deg < 30:
            return 'straight'
        # Left turn: angle > 30 degrees and cross product > 0 (counter-clockwise)
        elif cross_product > 0:
            return 'left'
        # Right turn: angle > 30 degrees and cross product < 0 (clockwise)
        else:
            return 'right'
    
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
    
    def _is_lane_at_edge(self, lane_id: int, image_points: List[Tuple[int, int]]) -> bool:
        """
        Check if a lane starts at the edge of the image (within threshold).
        Lanes that start at edges are entry points, not split lanes.
        
        Args:
            lane_id: ID of the lane
            image_points: List of (x, y) tuples in image coordinates
            
        Returns:
            True if lane starts at an edge, False otherwise
        """
        if not image_points:
            return False
        
        # Get the start point (first point) in image coordinates
        start_x, start_y = image_points[0]
        
        # Threshold for considering a point "at the edge" (in pixels)
        edge_threshold = 100  # pixels from edge
        
        # Check if start point is near any edge
        at_left_edge = start_x <= edge_threshold
        at_right_edge = start_x >= self.image_width - edge_threshold
        at_top_edge = start_y <= edge_threshold
        at_bottom_edge = start_y >= self.image_height - edge_threshold
        
        return at_left_edge or at_right_edge or at_top_edge or at_bottom_edge
    
    def _get_spawnable_lanes(self) -> List[int]:
        """
        Get lanes where vehicles can spawn.
        Simply returns all lanes that have spawn_position set (spawn_enabled in JSON).
        
        Returns:
            List of lane IDs where vehicles can spawn
        """
        available_lanes = []
        
        for lane_id, lane in self.road_network.all_lanes.items():
            # Lane MUST have centerlines for vehicles to spawn
            if not lane.centerline_points or len(lane.centerline_points) < 2:
                continue
            
            # Must have spawn_position (set from spawn_enabled in JSON)
            if not lane.spawn_position:
                continue
            
            # That's it! All lanes with spawn_position are spawnable
            available_lanes.append(lane_id)
        
        return available_lanes
    
    def _check_spawn_position_safe(self, lane_id: int, position: float) -> bool:
        """
        Check if a spawn position is safe (no bounding box overlap with existing vehicles).
        CRITICAL: This checks position 0 (beginning) to ensure no vehicles spawn on top of each other.
        
        Args:
            lane_id: ID of the lane
            position: Proposed spawn position along the lane (should be 0.0)
            
        Returns:
            True if position is safe (no overlap), False otherwise
        """
        lane = self.road_network.get_lane(lane_id)
        if not lane:
            return False
        
        # CRITICAL: Position must be 0.0 (beginning) - vehicles can only spawn at the start
        if abs(position) > 0.1:  # Allow small tolerance for floating point
            print(f"  ⚠ Warning: Spawn position {position:.2f} is not at beginning (0.0)")
        
        # Get spawn point in world coordinates (at position 0)
        spawn_point = lane.get_position_at_distance(0.0)
        spawn_x, spawn_y = spawn_point.x, spawn_point.y
        
        # Get direction at position 0 for angle calculation
        lane_dir = lane.get_direction_at_distance(0.0) if lane.centerline_points else lane.direction
        spawn_angle = np.arctan2(lane_dir[1], lane_dir[0])
        
        # Check against vehicles on the SAME lane only
        vehicle_length = 4.5
        vehicle_width = 2.0
        spawn_max_extent = np.sqrt((vehicle_length/2)**2 + (vehicle_width/2)**2)
        safety_margin = 0.0  # No safety margin - just check for actual overlap
        
        for vehicle in self.vehicles:
            # CRITICAL: Only check vehicles on the same lane
            if vehicle.lane_id != lane_id:
                continue
            
            # Only check vehicles within 10m of spawn point (position 0.0)
            if vehicle.position > 10.0:
                continue
            
            veh_x, veh_y = vehicle.get_visual_position()
            center_dist = np.sqrt((spawn_x - veh_x)**2 + (spawn_y - veh_y)**2)
            
            vehicle_extent = np.sqrt((vehicle.length/2)**2 + (vehicle.width/2)**2)
            min_allowed = spawn_max_extent + vehicle_extent + safety_margin
            if center_dist < min_allowed:
                return False
        
        return True  # No overlap, position is safe
    
    def _get_spawn_position_for_lane(self, lane_id: int) -> float:
        """
        Get the spawn position for a lane.
        DEPRECATED: Vehicles now always spawn at position 0.0 (beginning).
        This method is kept for compatibility but always returns 0.0.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Always returns 0.0 (spawn point at beginning)
        """
        # CRITICAL: Vehicles MUST spawn at position 0.0 (beginning)
        # Position 0.0 corresponds to the first centerline point where spawn_enabled is true
        return 0.0
    
    def _spawn_initial_vehicles(self):
        """Spawn initial vehicles on the road network."""
        # Spawn very few vehicles initially to avoid overlap
        # Let the spawn timer handle gradual spawning
        if self.max_vehicles > 0:
            # Spawn only 1-2 vehicles initially, well spaced
            num_initial = min(2, self.max_vehicles)
            for _ in range(num_initial):
                if len(self.vehicles) < self.max_vehicles:
                    self._spawn_vehicle()
                    # Small delay between initial spawns to ensure spacing
                    if self.global_spawn_timer is not None:
                        self.global_spawn_timer = 0.5  # Reset timer so next spawn is delayed
    
    def _spawn_vehicle(self, lane_id: Optional[int] = None):
        """Spawn a new vehicle with proper spacing."""
        if len(self.vehicles) >= self.max_vehicles:
            return
        
        # Get spawnable lanes (only lanes 1, 2, 4, 5, 6, 7, 8, 9)
        spawnable_lanes = self._get_spawnable_lanes()
        if not spawnable_lanes:
            # Fallback to all lanes if no spawnable lanes found
            spawnable_lanes = list(self.road_network.all_lanes.keys())
        
        if not spawnable_lanes:
            return
        
        # Randomly choose a spawnable lane if not specified
        if lane_id is None:
            # If per-lane spawn rates are defined, restrict to those lanes
            if isinstance(self.vehicle_spawn_rate, dict):
                allowed_lanes = list(self.vehicle_spawn_rate.keys())
                # Filter spawnable_lanes to only include allowed_lanes
                valid_lanes = [l for l in spawnable_lanes if l in allowed_lanes]
                if not valid_lanes:
                    return
                lane_id = random.choice(valid_lanes)
            else:
                lane_id = random.choice(spawnable_lanes)
        elif lane_id not in spawnable_lanes:
            # If specified lane is not spawnable (e.g. not safe), abort
            # But we should check if it's just temporarily unsafe or fundamentally unspawnable
            # For now, just check if it's in the list of valid lanes
            if lane_id not in self.road_network.all_lanes:
                return
            # If it exists but wasn't returned by _get_spawnable_lanes, it might be unsafe or not an entry lane
            # We'll proceed to safety check below

        
        # Create vehicle
        vehicle_id = self.vehicle_counter
        self.vehicle_counter += 1
        
        # Random vehicle properties - various speeds (some fast, some slower)
        # Speeds are multiplied by 10 for 10x faster movement (much faster for demos)
        # Create more variation: 60% fast, 30% medium, 10% slow
        speed_roll = random.random()
        if speed_roll < 0.6:
            # Fast vehicles (highway speeds) - 10x faster
            max_speed = random.uniform(16.6, 20) * self.model_boost  # m/s (900-1260 km/h equivalent)
        elif speed_roll < 0.9:
            # Medium speed vehicles (city speeds) - 10x faster
            max_speed = random.uniform(13, 16.6) * self.model_boost  # m/s (540-790 km/h equivalent)
        else:
            # Slow vehicles (traffic/slow drivers) - 10x faster
            max_speed = random.uniform(10.0, 13)  * self.model_boost  # m/s (360-540 km/h equivalent)
        
        max_acceleration = random.uniform(2.5 * self.model_boost, 4.0 * self.model_boost)  # m/s² (varied acceleration)
        max_deceleration = random.uniform(-10.0 * self.model_boost, -20.0 * self.model_boost)  # m/s² (varied braking)
        
        # Random color - but default to yellow for visibility
        color = (255, 220, 0)  # Yellow by default (like in the image)
        # Optionally add slight variation
        if random.random() < 0.2:  # 20% chance of slight variation
            color = (
                min(255, color[0] + random.randint(-20, 20)),
                min(255, color[1] + random.randint(-20, 20)),
                min(255, color[2] + random.randint(-20, 20))
            )
        
        # CRITICAL: Vehicles MUST spawn at position 0.0 (beginning of lane)
        # They can ONLY spawn at the spawn point (first point of centerline)
        # Position 0.0 corresponds to the first centerline point where spawn_enabled is true
        spawn_position = 0.0
        
        # Verify no bounding box overlap before spawning at position 0.0
        if not self._check_spawn_position_safe(lane_id, spawn_position):
            # Position not safe, skip spawning this vehicle
            print(f"⚠ Skipped spawning vehicle on lane {lane_id} - position {spawn_position:.1f}m not safe (overlap detected)")
            return
        
        # Random initial speed - varied based on max speed
        # Start at 60-80% of max speed for variety (already 5x faster)
        initial_speed = random.uniform(max_speed * 0.6, max_speed * 0.8)
        
        vehicle = Vehicle(
            model=self,
            unique_id=vehicle_id,
            lane_id=lane_id,
            position=spawn_position,  # Always spawn at position 0.0 (spawn point)
            speed=initial_speed,
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
        
        # Verify vehicle orientation matches lane direction
        lane = self.road_network.get_lane(lane_id)
        if lane:
            lane_dir = lane.direction
            # CRITICAL: Use direction at position 0 (beginning) to match spawn position
            if lane.centerline_points and len(lane.centerline_points) >= 2:
                # Use direction from first to second centerline point for accurate direction
                lane_dir = lane.get_direction_at_distance(0.0)
            
            vehicle_angle = vehicle.get_visual_angle()
            lane_angle = np.arctan2(lane_dir[1], lane_dir[0])
            angle_diff = abs(vehicle_angle - lane_angle)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            
            # Debug for vertical lanes
            if lane_id in [6, 7, 8, 9]:
                print(f"Spawned vehicle {vehicle_id} on lane {lane_id} at position {spawn_position:.1f}m")
                print(f"  Lane direction: ({lane_dir[0]:.3f}, {lane_dir[1]:.3f}), angle: {np.degrees(lane_angle):.1f}°")
                print(f"  Vehicle angle: {np.degrees(vehicle_angle):.1f}° (diff: {np.degrees(angle_diff):.2f}°)")
                if abs(angle_diff) > 0.1:  # More than ~6 degrees difference
                    print(f"  ⚠ WARNING: Vehicle angle doesn't match lane direction!")
        else:
            print(f"Spawned vehicle {vehicle_id} on lane {lane_id} at position {spawn_position:.1f}m")
    
    def _calculate_safe_start_position_at(self, lane_id: int, preferred_position: float) -> float:
        """
        Calculate a safe starting position for a new vehicle near a preferred position.
        
        Args:
            lane_id: ID of the lane
            preferred_position: Preferred spawn position (left/right/top beginning)
            
        Returns:
            Safe starting position along the lane
        """
        # Get existing vehicles in this lane
        existing_vehicles = self.get_vehicles_in_lane(lane_id)
        
        if not existing_vehicles:
            # No vehicles in lane, spawn at preferred position
            return preferred_position
        
        # Sort vehicles by position
        existing_vehicles.sort(key=lambda v: v.position)
        
        # Get lane length
        lane_length = self.get_lane_length(lane_id)
        
        # Check if spawning at beginning (position 0) or end (position = lane_length)
        spawn_at_beginning = preferred_position < lane_length * 0.1
        
        if spawn_at_beginning:
            # Spawn at beginning - check vehicles near start
            check_distance = min(300.0, lane_length * 0.2)
            nearby_vehicles = [v for v in existing_vehicles if v.position < check_distance]
            
            if not nearby_vehicles:
                return preferred_position
            
            # Find closest vehicle to start
            closest_vehicle = min(nearby_vehicles, key=lambda v: v.position)
            
            # Safe distance - ensure vehicles don't overlap (minimum = vehicle length + safety margin)
            # For vehicles spawning at beginning, they need space ahead of closest vehicle
            vehicle_length = 4.5  # Average vehicle length
            min_safe_distance = vehicle_length + 10.0  # At least vehicle length + 10m safety margin
            safe_position = closest_vehicle.position - min_safe_distance - vehicle_length
            
            # Ensure safe position is at least at spawn position
            safe_position = max(preferred_position, safe_position)
            
            # If safe position is negative, try to find a gap
            if safe_position < 0:
                positions_used = [v.position for v in nearby_vehicles]
                positions_used.sort()
                
                # Look for gaps - need at least vehicle length + safety margin
                vehicle_length = 4.5
                gap_requirement = vehicle_length + 10.0  # Minimum gap between vehicles
                for i in range(len(positions_used) - 1):
                    gap_start = positions_used[i] + gap_requirement
                    gap_end = positions_used[i + 1] - gap_requirement
                    if gap_end > gap_start and gap_end > preferred_position:
                        # Found a gap, use it
                        return max(preferred_position, gap_start)
                
                # No gap found - spawn after the last vehicle with proper spacing
                max_position = max(positions_used) + vehicle_length + 10.0
                return min(max_position, check_distance)
            
            return max(0, safe_position)
        else:
            # Spawn at end - check vehicles near end
            check_distance = min(300.0, lane_length * 0.2)
            nearby_vehicles = [v for v in existing_vehicles if v.position > lane_length - check_distance]
            
            if not nearby_vehicles:
                return preferred_position
            
            # Find closest vehicle to end
            closest_vehicle = max(nearby_vehicles, key=lambda v: v.position)
            
            # Safe distance - ensure vehicles don't overlap
            vehicle_length = 4.5  # Average vehicle length
            min_safe_distance = vehicle_length + 10.0  # At least vehicle length + 10m safety margin
            safe_position = closest_vehicle.position + min_safe_distance + vehicle_length
            
            # Ensure safe position doesn't exceed preferred position (for end spawns)
            safe_position = min(preferred_position, safe_position)
            
            # If safe position exceeds lane length, try to find a gap
            if safe_position > lane_length:
                positions_used = [v.position for v in nearby_vehicles]
                positions_used.sort()
                
                # Look for gaps - need at least vehicle length + safety margin
                vehicle_length = 4.5
                gap_requirement = vehicle_length + 10.0  # Minimum gap between vehicles
                for i in range(len(positions_used) - 1):
                    gap_start = positions_used[i] + gap_requirement
                    gap_end = positions_used[i + 1] - gap_requirement
                    if gap_end > gap_start and gap_start < preferred_position:
                        # Found a gap, use it
                        return min(preferred_position, gap_end)
                
                # No gap found - spawn before the first vehicle with proper spacing
                vehicle_length = 4.5
                min_position = min(positions_used) - vehicle_length - 10.0
                return max(min_position, lane_length - check_distance)
            
            return min(lane_length, safe_position)
    
    def _calculate_safe_start_position(self, lane_id: int) -> float:
        """
        Calculate a safe starting position for a new vehicle at the start of the road.
        Deprecated: Use _calculate_safe_start_position_at instead.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Safe starting position along the lane (at the start)
        """
        return self._calculate_safe_start_position_at(lane_id, 0.0)
    
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
    
    def get_traffic_lights_in_lane(self, lane_id: int) -> List[TrafficLights]:
        """
        Get all traffic lights in a specific lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            List of traffic lights in the lane
        """
        lane = self.road_network.get_lane(lane_id)
        return [] if lane is None else lane.traffic_lights
    
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
    
    def is_lane_end_at_frame_edge(self, lane_id: int) -> bool:
        """
        Check if a lane's end point is near the frame edge.
        Vehicles should only be removed if they reach the end of lanes that end at frame edges.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            True if lane end is near frame edge, False otherwise
        """
        # Get lane and its end point
        lane = self.road_network.get_lane(lane_id)
        if not lane:
            return False
        
        # Get end point in world coordinates
        end_point_world = lane.end_point
        
        # If lane has centerline points, use the last centerline point as end
        if lane.centerline_points and len(lane.centerline_points) > 0:
            end_point_world = lane.centerline_points[-1]
        
        # Transformation parameters (same as in _create_custom_road_network)
        offset_x = 2064.0  # ROI center X
        offset_y = 526.0   # ROI center Y
        scale = 0.3507     # pixels per meter
        
        # Convert world coordinates to image coordinates
        img_x = end_point_world.x * scale + offset_x
        img_y = offset_y - end_point_world.y * scale  # Flip Y-axis back
        
        # Threshold for considering a point "at the edge" (in pixels)
        edge_threshold = 100  # pixels from edge
        
        # Check if end point is near any edge
        at_left_edge = img_x <= edge_threshold
        at_right_edge = img_x >= self.image_width - edge_threshold
        at_top_edge = img_y <= edge_threshold
        at_bottom_edge = img_y >= self.image_height - edge_threshold
        
        return at_left_edge or at_right_edge or at_top_edge or at_bottom_edge
    
    def get_lane_position(self, lane_id: int) -> tuple:
        """
        Get the starting position of a lane.
        
        Args:
            lane_id: ID of the lane
            
        Returns:
            Tuple of (x, y) coordinates
        """
        return self.road_network.get_lane_position(lane_id)
    
    def get_lane_direction(self, lane_id: int, distance: Optional[float] = None) -> tuple:
        """
        Get the direction vector of a lane.
        If distance is provided and lane has centerline, returns direction at that distance.
        
        Args:
            lane_id: ID of the lane
            distance: Optional distance along lane to get direction at
            
        Returns:
            Tuple of (dx, dy) direction vector
        """
        return self.road_network.get_lane_direction(lane_id, distance)
    
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
        if isinstance(self.vehicle_spawn_rate, dict):
            # Per-lane spawning
            for lane_id, rate in self.vehicle_spawn_rate.items():
                if lane_id in self.spawn_timers:
                    self.spawn_timers[lane_id] += self.time_step
                    
                    if (self.spawn_timers[lane_id] >= 1.0 / rate and 
                        len(self.vehicles) < self.max_vehicles):
                        print(f"Attempting to spawn on lane {lane_id} (timer: {self.spawn_timers[lane_id]:.2f})")
                        self._spawn_vehicle(lane_id)
                        self.spawn_timers[lane_id] = 0.0
        else:
            # Global spawning
            if self.global_spawn_timer is not None:
                self.global_spawn_timer += self.time_step
                
                # Spawn new vehicles if needed (very slowly)
                if (self.global_spawn_timer >= 1.0 / self.vehicle_spawn_rate and 
                    len(self.vehicles) < self.max_vehicles):
                    self._spawn_vehicle()
                    self.global_spawn_timer = 0.0

        # Update traffic lights
        self.traffic_lights_node.step()

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
        if isinstance(self.vehicle_spawn_rate, dict):
            self.spawn_timers = {lane_id: 0.0 for lane_id in self.vehicle_spawn_rate.keys()}
            self.global_spawn_timer = None
        else:
            self.spawn_timers = None
            self.global_spawn_timer = 0.0
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
