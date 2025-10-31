"""
PyGame Visualization for Traffic Simulation

This module provides real-time visualization of the traffic simulation using PyGame.
It displays vehicles, roads, lanes, and simulation statistics.
"""

import pygame
import numpy as np
from typing import Tuple, List, Dict
import math
import os
import requests
from io import BytesIO
from PIL import Image

from ..models.traffic_model import TrafficSimulationModel
from ..agents.vehicle import Vehicle
from ..models.road_network import Lane, Point


class TrafficVisualization:
    """
    PyGame-based visualization for the traffic simulation.
    
    Features:
    - Real-time vehicle rendering
    - Road and lane visualization
    - Statistics display
    - Interactive controls
    - Zoom and pan capabilities
    """
    
    def __init__(self, 
                 model: TrafficSimulationModel,
                 width: int = 1200,
                 height: int = 800,
                 fps: int = 60):
        """
        Initialize the visualization.
        
        Args:
            model: Traffic simulation model
            width: Window width (pixels)
            height: Window height (pixels)
            fps: Target frames per second
        """
        self.model = model
        self.width = width
        self.height = height
        self.fps = fps
        
        # PyGame initialization
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Traffic Simulation")
        self.clock = pygame.time.Clock()
        
        # Colors
        self.colors = {
            'background': (240, 240, 240),  # Light grey background
            'road': (60, 60, 60),          # Dark grey/black roads
            'shoulder': (200, 200, 200),   # Light grey shoulders
            'lane_marker': (255, 255, 255), # White lane markers
            'center_line': (255, 165, 0),   # Orange center line (double)
            'crosswalk': (180, 100, 100),   # Dark red crosswalk
            'crosswalk_white': (255, 255, 255), # White crosswalk stripes
            'vehicle': (255, 220, 0),       # Yellow vehicles
            'vehicle_detail': (100, 150, 255), # Blue detail (like sensor)
            'text': (0, 0, 0),
            'stats_bg': (255, 255, 255, 230)
        }
        
        # View settings
        self.zoom = 1.0  # Start with better zoom to see lanes and vehicles clearly
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Fonts
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        
        # Statistics
        self.stats_history = {
            'speed': [],
            'vehicles': [],
            'time': []
        }
        
        # Running state
        self.running = True
        self.paused = False

        # mouse
        self.dragging = False
        
        # Car sprites
        self.car_sprites = {}
        # self._load_car_sprites()
    
    def _load_car_sprites(self):
        """Load car sprites from internet or create simple ones."""
        # TODO downloading sprites does not make sense
        try:
            # Try to load car images from internet
            self._load_car_from_internet()
        except Exception as e:
            print(f"Could not load car images from internet: {e}")
            print("Using simple car shapes instead")
            self._create_simple_car_sprites()
        # self._create_simple_car_sprites()
    
    # TODO
    # does not exist actually xd
    def _load_car_from_internet(self):
        """Load car images from internet."""
        # Simple car image URLs (you can replace these with better ones)
        car_urls = {
            'red': 'https://via.placeholder.com/64x32/FF0000/FFFFFF?text=🚗',
            'blue': 'https://via.placeholder.com/64x32/0000FF/FFFFFF?text=🚗',
            'green': 'https://via.placeholder.com/64x32/00FF00/FFFFFF?text=🚗',
            'yellow': 'https://via.placeholder.com/64x32/FFFF00/000000?text=🚗',
            'purple': 'https://via.placeholder.com/64x32/800080/FFFFFF?text=🚗',
        }
        
        for color_name, url in car_urls.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    # Convert to pygame surface
                    image = Image.open(BytesIO(response.content))
                    image = image.convert('RGBA')
                    
                    # Convert PIL image to pygame surface
                    mode = image.mode
                    size = image.size
                    data = image.tobytes()
                    
                    car_surface = pygame.image.fromstring(data, size, mode)
                    self.car_sprites[color_name] = car_surface
                    print(f"Loaded car sprite for {color_name}")
            except Exception as e:
                print(f"Failed to load car sprite for {color_name}: {e}")
    
    def _create_simple_car_sprites(self):
        """Create simple car sprites programmatically."""
        colors = {
            'red': (255, 0, 0),
            'blue': (0, 0, 255),
            'green': (0, 255, 0),
            'yellow': (255, 255, 0),
            'purple': (128, 0, 128),
        }
        
        for color_name, color in colors.items():
            # Create a simple car sprite
            car_surface = pygame.Surface((64, 32), pygame.SRCALPHA)
            
            # Draw car body
            pygame.draw.rect(car_surface, color, (8, 8, 48, 16))
            
            # Draw windows
            pygame.draw.rect(car_surface, (50, 50, 50), (12, 10, 12, 4))
            pygame.draw.rect(car_surface, (50, 50, 50), (40, 10, 12, 4))
            
            # Draw wheels
            pygame.draw.circle(car_surface, (30, 30, 30), (16, 12), 3)
            pygame.draw.circle(car_surface, (30, 30, 30), (16, 20), 3)
            pygame.draw.circle(car_surface, (30, 30, 30), (48, 12), 3)
            pygame.draw.circle(car_surface, (30, 30, 30), (48, 20), 3)
            
            self.car_sprites[color_name] = car_surface
            print(f"Created simple car sprite for {color_name}")
    
    def _get_car_color_name(self, color: Tuple[int, int, int]) -> str:
        """Convert RGB color to color name."""
        r, g, b = color
        
        # Simple color matching
        if r > 200 and g < 100 and b < 100:
            return 'red'
        elif r < 100 and g < 100 and b > 200:
            return 'blue'
        elif r < 100 and g > 200 and b < 100:
            return 'green'
        elif r > 200 and g > 200 and b < 100:
            return 'yellow'
        else:
            return 'purple'  # Default
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            world_x: World x coordinate
            world_y: World y coordinate
            
        Returns:
            Tuple of (screen_x, screen_y)
        """
        screen_x = int((world_x + self.pan_x) * self.zoom + self.width // 2)
        screen_y = int((world_y + self.pan_y) * self.zoom + self.height // 2)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.
        
        Args:
            screen_x: Screen x coordinate
            screen_y: Screen y coordinate
            
        Returns:
            Tuple of (world_x, world_y)
        """
        world_x = (screen_x - self.width // 2) / self.zoom - self.pan_x
        world_y = (screen_y - self.height // 2) / self.zoom - self.pan_y
        return world_x, world_y
    
    def draw_road(self, lane: Lane):
        """
        Draw a single lane/road segment with proper styling.
        
        Args:
            lane: Lane to draw
        """
        # Convert world coordinates to screen coordinates
        start_x, start_y = self.world_to_screen(lane.start_point.x, lane.start_point.y)
        end_x, end_y = self.world_to_screen(lane.end_point.x, lane.end_point.y)
        
        # Calculate road width - MUCH WIDER for visibility
        # Base width is much larger, scaled by zoom
        base_width = 15.0  # Base width in pixels (much larger)
        road_width = max(12, int(base_width * self.zoom))  # Ensure minimum visibility
        
        # Calculate perpendicular direction for drawing shoulders and center lines
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
            
            # Draw shoulder (light grey strip) - wider than road
            shoulder_width = road_width + max(6, int(base_width * self.zoom * 0.4))
            shoulder_start = (int(start_x + perp_x * shoulder_width/2),
                            int(start_y + perp_y * shoulder_width/2))
            shoulder_end = (int(start_x - perp_x * shoulder_width/2),
                          int(start_y - perp_y * shoulder_width/2))
            pygame.draw.line(self.screen, self.colors['shoulder'], 
                           shoulder_start, shoulder_end, shoulder_width)
            
            # Draw main road (dark grey) - MUCH WIDER
            road_start = (int(start_x + perp_x * road_width/2),
                         int(start_y + perp_y * road_width/2))
            road_end = (int(start_x - perp_x * road_width/2),
                       int(start_y - perp_y * road_width/2))
            pygame.draw.line(self.screen, self.colors['road'], 
                           road_start, road_end, road_width)
            
            # Draw center line (white lane marker) - thicker for visibility
            if road_width > 6:
                marker_width = max(2, int(road_width * 0.15))  # Thicker markers
                pygame.draw.line(self.screen, self.colors['lane_marker'], 
                               (start_x, start_y), (end_x, end_y), marker_width)
            
            # Draw directional arrows periodically
            self._draw_directional_arrows(start_x, start_y, end_x, end_y, dx, dy, length, road_width)
    
    def draw_vehicle(self, vehicle: Vehicle):
        """
        Draw a vehicle as a yellow rectangle with blue detail (like in the image).
        
        Args:
            vehicle: Vehicle to draw
        """
        # Get vehicle position
        visual_x, visual_y = vehicle.get_visual_position()
        screen_x, screen_y = self.world_to_screen(visual_x, visual_y)
        
        # Get vehicle angle
        angle = vehicle.get_visual_angle()
        
        # Vehicle dimensions - MUCH LARGER for visibility
        # Scale vehicles to be clearly visible
        base_length = 25.0  # Base vehicle length in pixels
        base_width = 15.0   # Base vehicle width in pixels
        length = max(20, int(base_length * self.zoom))  # Make cars much bigger
        width = max(12, int(base_width * self.zoom))   # Make cars much bigger
        
        # Draw vehicle as yellow rectangle (like in the image)
        self._draw_vehicle_shape(screen_x, screen_y, length, width, angle)
    
    def _draw_vehicle_shape(self, x: int, y: int, length: int, width: int, angle: float):
        """
        Draw a vehicle as a yellow rectangle with blue detail.
        
        Args:
            x, y: Center position
            length: Car length
            width: Car width
            angle: Rotation angle in radians
        """
        # Create vehicle surface
        car_surface = pygame.Surface((length + 4, width + 4), pygame.SRCALPHA)
        
        # Draw main yellow body (rectangle)
        body_rect = pygame.Rect(2, 2, length, width)
        pygame.draw.rect(car_surface, self.colors['vehicle'], body_rect)
        
        # Draw blue detail rectangle on front (like sensor/identifier)
        detail_width = max(3, int(width * 0.4))
        detail_length = max(4, int(length * 0.15))
        detail_rect = pygame.Rect(length - detail_length - 1, 
                                 (width - detail_width) // 2 + 2, 
                                 detail_length, detail_width)
        pygame.draw.rect(car_surface, self.colors['vehicle_detail'], detail_rect)
        
        # Draw border for better visibility
        pygame.draw.rect(car_surface, (200, 180, 0), body_rect, 1)
        
        # Rotate the vehicle surface
        rotated_surface = pygame.transform.rotate(car_surface, math.degrees(angle))
        
        # Get the rotated rect and center it
        rotated_rect = rotated_surface.get_rect()
        rotated_rect.center = (x, y)
        
        # Draw the rotated vehicle
        self.screen.blit(rotated_surface, rotated_rect)
    
    def _draw_car_shape(self, x: int, y: int, length: int, width: int, angle: float, color: Tuple[int, int, int]):
        """
        Draw a car-like shape.
        
        Args:
            x, y: Center position
            length: Car length
            width: Car width
            angle: Rotation angle in radians
            color: Car color
        """
        # Create car surface
        car_surface = pygame.Surface((length + 4, width + 4), pygame.SRCALPHA)
        
        # Draw car body (main rectangle)
        body_rect = pygame.Rect(2, 2, length, width)
        pygame.draw.rect(car_surface, color, body_rect)
        
        # Draw car front (slightly rounded)
        front_rect = pygame.Rect(length - length//4, 2, length//4, width)
        pygame.draw.rect(car_surface, (color[0]//2, color[1]//2, color[2]//2), front_rect)
        
        # Draw windows (darker rectangles)
        window_color = (color[0]//3, color[1]//3, color[2]//3)
        window_width = max(1, width//3)
        window_length = max(2, length//3)
        
        # Front window
        front_window = pygame.Rect(length//4, 2, window_length, window_width)
        pygame.draw.rect(car_surface, window_color, front_window)
        
        # Rear window
        rear_window = pygame.Rect(length - length//4 - window_length, 2, window_length, window_width)
        pygame.draw.rect(car_surface, window_color, rear_window)
        
        # Draw wheels (small circles)
        wheel_color = (50, 50, 50)
        wheel_radius = max(1, width//6)
        
        # Front wheels
        pygame.draw.circle(car_surface, wheel_color, (length//4, width//4), wheel_radius)
        pygame.draw.circle(car_surface, wheel_color, (length//4, 3*width//4), wheel_radius)
        
        # Rear wheels
        pygame.draw.circle(car_surface, wheel_color, (3*length//4, width//4), wheel_radius)
        pygame.draw.circle(car_surface, wheel_color, (3*length//4, 3*width//4), wheel_radius)
        
        # Rotate the car surface
        rotated_surface = pygame.transform.rotate(car_surface, math.degrees(angle))
        
        # Get the rotated rect and center it
        rotated_rect = rotated_surface.get_rect()
        rotated_rect.center = (x, y)
        
        # Draw the rotated car
        self.screen.blit(rotated_surface, rotated_rect)
    
    def _draw_car_sprite(self, x: int, y: int, length: int, width: int, angle: float, color_name: str):
        """
        Draw a car using a sprite/image.
        
        Args:
            x, y: Center position
            length: Car length
            width: Car width
            angle: Rotation angle in radians
            color_name: Color name for sprite lookup
        """
        if color_name not in self.car_sprites:
            return
        
        # Get the original sprite
        original_sprite = self.car_sprites[color_name]
        
        # Scale the sprite to match the desired size
        scaled_sprite = pygame.transform.scale(original_sprite, (length, width))
        
        # Rotate the sprite
        rotated_sprite = pygame.transform.rotate(scaled_sprite, math.degrees(angle))
        
        # Get the rotated rect and center it
        rotated_rect = rotated_sprite.get_rect()
        rotated_rect.center = (x, y)
        
        # Draw the rotated sprite
        self.screen.blit(rotated_sprite, rotated_rect)
    
    def draw_statistics(self):
        """Draw simulation statistics on screen."""
        # Get current model info
        model_info = self.model.get_model_info()
        stats = model_info['stats']
        
        # Create statistics text
        stats_text = [
            f"Time: {model_info['current_time']:.1f}s",
            f"Vehicles: {model_info['num_vehicles']}",
            f"Average Speed: {stats['average_speed']:.1f} m/s",
            f"Total Spawned: {stats['total_vehicles_spawned']}",
            f"Total Removed: {stats['total_vehicles_removed']}",
            f"Zoom: {self.zoom:.2f}x"
        ]
        
        # Draw statistics background (white with border)
        stats_height = len(stats_text) * 25 + 10
        stats_rect = pygame.Rect(10, 10, 250, stats_height)
        pygame.draw.rect(self.screen, (255, 255, 255), stats_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), stats_rect, 2)
        
        # Draw statistics text (black text on white background)
        for i, text in enumerate(stats_text):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (20, 20 + i * 25))
        
        # Draw controls
        controls_text = [
            "Controls:",
            "SPACE - Pause/Resume",
            "R - Reset simulation",
            "+/- - Zoom in/out",
            "Mouse Wheel - Zoom",
            "Mouse Drag - Pan"
        ]
        
        controls_height = len(controls_text) * 20 + 10
        controls_rect = pygame.Rect(10, self.height - controls_height - 10, 200, controls_height)
        pygame.draw.rect(self.screen, (255, 255, 255), controls_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), controls_rect, 2)
        
        for i, text in enumerate(controls_text):
            text_surface = self.small_font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (20, self.height - controls_height + i * 20))
    
    def draw_road_network(self):
        """Draw the entire road network."""
        # Draw all lanes (roads first)
        for lane_id, lane in self.model.road_network.all_lanes.items():
            self.draw_road(lane)
        
        # Draw crosswalks at intersections
        self.draw_crosswalks()
        
        # Draw traffic lights
        self.draw_traffic_lights()
    
    def draw_crosswalks(self):
        """Draw crosswalk patterns at intersections."""
        for intersection in self.model.road_network.intersections:
            center_x, center_y = self.world_to_screen(
                intersection.center_point.x, 
                intersection.center_point.y
            )
            
            # Draw crosswalk pattern (alternating stripes)
            crosswalk_size = max(30, int(50 * self.zoom))
            stripe_width = max(3, int(5 * self.zoom))
            num_stripes = 8
            
            # Draw crosswalk for each approach
            for lane_id in intersection.connected_lanes:
                lane = self.model.road_network.get_lane(lane_id)
                if not lane:
                    continue
                
                # Draw crosswalk perpendicular to lane direction
                dx = lane.end_point.x - lane.start_point.x
                dy = lane.end_point.y - lane.start_point.y
                length = math.sqrt(dx*dx + dy*dy)
                
                if length == 0:
                    continue
                
                # Perpendicular direction
                perp_x = -dy / length
                perp_y = dx / length
                
                # Draw alternating stripes
                for i in range(num_stripes):
                    stripe_offset = (i - num_stripes/2) * stripe_width * 2
                    
                    if i % 2 == 0:
                        # White stripe
                        stripe_color = self.colors['crosswalk_white']
                    else:
                        # Dark red stripe
                        stripe_color = self.colors['crosswalk']
                    
                    stripe_start = (int(center_x + perp_x * (crosswalk_size/2 + stripe_offset)),
                                   int(center_y + perp_y * (crosswalk_size/2 + stripe_offset)))
                    stripe_end = (int(center_x + perp_x * (crosswalk_size/2 + stripe_offset + stripe_width)),
                                 int(center_y + perp_y * (crosswalk_size/2 + stripe_offset + stripe_width)))
                    
                    pygame.draw.line(self.screen, stripe_color, stripe_start, stripe_end, stripe_width)
    
    def _draw_directional_arrows(self, start_x, start_y, end_x, end_y, dx, dy, length, road_width):
        """
        Draw white directional arrows on the road.
        
        Args:
            start_x, start_y: Start position
            end_x, end_y: End position
            dx, dy: Direction vector
            length: Length of road segment
            road_width: Width of road
        """
        if length == 0:
            return
        
        # Only draw arrows if road is wide enough
        if road_width < 10:
            return
        
        # Draw arrows at regular intervals
        arrow_spacing = 100 * self.zoom  # Spacing between arrows
        num_arrows = int(length / arrow_spacing)
        
        if num_arrows < 1:
            return
        
        # Direction vector (normalized)
        dir_x = dx / length
        dir_y = dy / length
        
        # Perpendicular vector for arrow position
        perp_x = -dir_y
        perp_y = dir_x
        
        arrow_size = max(4, int(road_width * 0.3))
        
        for i in range(1, num_arrows + 1):
            t = i / (num_arrows + 1)
            arrow_x = start_x + t * dx
            arrow_y = start_y + t * dy
            
            # Draw simple arrow (triangle pointing forward)
            arrow_points = [
                (int(arrow_x + dir_x * arrow_size), int(arrow_y + dir_y * arrow_size)),  # Tip
                (int(arrow_x - dir_x * arrow_size/2 + perp_x * arrow_size/2), 
                 int(arrow_y - dir_y * arrow_size/2 + perp_y * arrow_size/2)),  # Left
                (int(arrow_x - dir_x * arrow_size/2 - perp_x * arrow_size/2), 
                 int(arrow_y - dir_y * arrow_size/2 - perp_y * arrow_size/2)),  # Right
            ]
            
            pygame.draw.polygon(self.screen, self.colors['lane_marker'], arrow_points)
    
    def draw_traffic_lights(self):
        """Draw traffic lights at intersections."""
        for intersection in self.model.road_network.intersections:
            for lane_id, light_state in intersection.traffic_lights.items():
                lane = self.model.road_network.get_lane(lane_id)
                if not lane:
                    continue
                
                # Draw traffic light at the end of incoming lane (near intersection)
                light_position = lane.end_point
                screen_x, screen_y = self.world_to_screen(light_position.x, light_position.y)
                
                # Draw traffic light circle (smaller and more proportional)
                light_size = max(4, int(6 * self.zoom))  # Much smaller
                light_size = min(light_size, 12)  # Cap maximum size
                
                # Color based on state
                if light_state == 'green':
                    color = (0, 255, 0)
                elif light_state == 'yellow':
                    color = (255, 255, 0)
                else:  # red
                    color = (255, 0, 0)
                
                # Draw stop line (red line across road) when red
                if light_state == 'red':
                    # Draw perpendicular line across road
                    dx = lane.end_point.x - lane.start_point.x
                    dy = lane.end_point.y - lane.start_point.y
                    length = math.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        perp_x = -dy / length
                        perp_y = dx / length
                        
                        # Use consistent road width calculation
                        base_width = 15.0
                        road_width = max(12, int(base_width * self.zoom))
                        stop_line_length = road_width + max(4, int(base_width * self.zoom * 0.4))
                        
                        stop_start = (int(screen_x + perp_x * stop_line_length/2),
                                     int(screen_y + perp_y * stop_line_length/2))
                        stop_end = (int(screen_x - perp_x * stop_line_length/2),
                                   int(screen_y - perp_y * stop_line_length/2))
                        pygame.draw.line(self.screen, color, stop_start, stop_end, 2)  # Thinner line
                
                # Draw traffic light circle
                pygame.draw.circle(self.screen, color, (screen_x, screen_y), light_size)
                pygame.draw.circle(self.screen, (255, 255, 255), (screen_x, screen_y), light_size, 2)
    
    def handle_events(self):
        """Handle PyGame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pygame.K_r:
                    self.model.reset()
                elif event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Zoom in
                    self.zoom *= 1.2
                    self.zoom = min(5.0, self.zoom)
                elif event.key == pygame.K_MINUS:
                    # Zoom out
                    self.zoom *= 0.8
                    self.zoom = max(0.1, self.zoom)
            
            elif event.type == pygame.MOUSEWHEEL:
                # Zoom in/out
                zoom_factor = 1.1 if event.y > 0 else 0.9
                self.zoom *= zoom_factor
                self.zoom = max(0.1, min(8.0, self.zoom))  # Clamp zoom
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.dragging = True
                    self.last_mouse_pos = pygame.mouse.get_pos()
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    self.dragging = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    current_mouse_pos = pygame.mouse.get_pos()
                    dx = current_mouse_pos[0] - self.last_mouse_pos[0]
                    dy = current_mouse_pos[1] - self.last_mouse_pos[1]
                    
                    # Convert screen movement to world movement
                    self.pan_x += dx / self.zoom
                    self.pan_y += dy / self.zoom
                    
                    self.last_mouse_pos = current_mouse_pos
    
    def update(self):
        """Update the visualization."""
        # Handle events
        self.handle_events()
        
        # Update simulation if not paused
        if not self.paused:
            self.model.step()
    
    def draw(self):
        """Draw the current frame."""
        # Clear screen
        self.screen.fill(self.colors['background'])
        
        # Draw road network
        self.draw_road_network()
        
        # Draw vehicles
        for vehicle in self.model.vehicles:
            self.draw_vehicle(vehicle)
        
        # Draw statistics
        self.draw_statistics()
        
        # Draw pause indicator
        if self.paused:
            pause_text = self.font.render("PAUSED", True, (255, 0, 0))
            pause_rect = pause_text.get_rect(center=(self.width // 2, 50))
            self.screen.blit(pause_text, pause_rect)
        
        # Update display
        pygame.display.flip()
    
    def run(self):
        """Main visualization loop."""
        print("Starting traffic simulation visualization...")
        print("Controls:")
        print("  SPACE - Pause/Resume")
        print("  R - Reset simulation")
        print("  +/- - Zoom in/out")
        print("  Mouse Wheel - Zoom")
        print("  Mouse Drag - Pan")
        print("  ESC - Exit")
        
        while self.running:
            # Update
            self.update()
            
            # Draw
            self.draw()
            
            # Control frame rate
            self.clock.tick(self.fps)
        
        # Cleanup
        pygame.quit()
        print("Visualization ended")
