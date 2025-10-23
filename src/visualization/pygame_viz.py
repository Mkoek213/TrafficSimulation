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
            'background': (30, 30, 30),  # Darker background
            'road': (120, 120, 120),      # Much lighter gray for roads
            'lane_marker': (255, 255, 255),
            'vehicle': (255, 0, 0),
            'text': (255, 255, 255),
            'stats_bg': (0, 0, 0, 128)
        }
        
        # View settings
        self.zoom = 0.5  # Start with a better zoom level
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
        Draw a single lane/road segment.
        
        Args:
            lane: Lane to draw
        """
        # Convert world coordinates to screen coordinates
        start_x, start_y = self.world_to_screen(lane.start_point.x, lane.start_point.y)
        end_x, end_y = self.world_to_screen(lane.end_point.x, lane.end_point.y)
        
        # Draw road (MUCH thicker line for laptop visibility)
        # road_width = int(lane.lane_width * self.zoom * 20)  # Make roads MUCH wider
        # if road_width < 20:
        #     road_width = 20
        road_width = int(lane.lane_width * self.zoom)
        

        pygame.draw.line(self.screen, self.colors['road'], 
                        (start_x, start_y), (end_x, end_y), road_width)
        
        # Draw lane markers (center line)
        if road_width > 40:
            pygame.draw.line(self.screen, self.colors['lane_marker'], 
                           (start_x, start_y), (end_x, end_y), 8)
    
    def draw_vehicle(self, vehicle: Vehicle):
        """
        Draw a vehicle with a car-like shape.
        
        Args:
            vehicle: Vehicle to draw
        """
        # Get vehicle position
        visual_x, visual_y = vehicle.get_visual_position()
        screen_x, screen_y = self.world_to_screen(visual_x, visual_y)
        
        # Get vehicle angle
        angle = vehicle.get_visual_angle()
        
        # Vehicle dimensions (scaled by zoom)
        # length = int(vehicle.length * self.zoom * 8)  # Make cars MUCH bigger
        # width = int(vehicle.width * self.zoom * 8)    # Make cars MUCH bigger
        length = int(vehicle.length * self.zoom)  # Make cars MUCH bigger
        width = int(vehicle.width * self.zoom)    # Make cars MUCH bigger
        
        # Ensure minimum size
        # length = max(32, length)
        # width = max(16, width)
        
        # Try to use car sprite, fallback to drawn shape
        color_name = self._get_car_color_name(vehicle.color)
        if color_name in self.car_sprites:
            self._draw_car_sprite(screen_x, screen_y, length, width, angle, color_name)
        else:
            self._draw_car_shape(screen_x, screen_y, length, width, angle, vehicle.color)
        
        # Draw speed indicator (small line showing speed)
        if self.zoom > 1.0:  # Only show when zoomed in
            speed_line_length = int(vehicle.speed * self.zoom * 0.5)
            if speed_line_length > 0:
                end_x = screen_x + int(speed_line_length * math.cos(angle))
                end_y = screen_y + int(speed_line_length * math.sin(angle))
                pygame.draw.line(self.screen, (255, 255, 0), 
                               (screen_x, screen_y), (end_x, end_y), 2)
    
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
        
        # Draw statistics background
        stats_height = len(stats_text) * 25 + 10
        stats_rect = pygame.Rect(10, 10, 250, stats_height)
        pygame.draw.rect(self.screen, self.colors['stats_bg'], stats_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), stats_rect, 2)
        
        # Draw statistics text
        for i, text in enumerate(stats_text):
            text_surface = self.font.render(text, True, self.colors['text'])
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
        pygame.draw.rect(self.screen, self.colors['stats_bg'], controls_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), controls_rect, 2)
        
        for i, text in enumerate(controls_text):
            text_surface = self.small_font.render(text, True, self.colors['text'])
            self.screen.blit(text_surface, (20, self.height - controls_height + i * 20))
    
    def draw_road_network(self):
        """Draw the entire road network."""
        # Draw all lanes
        for lane_id, lane in self.model.road_network.all_lanes.items():
            self.draw_road(lane)
    
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
