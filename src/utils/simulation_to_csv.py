"""
Simulation to CSV Converter

Converts MESA traffic simulation output to CSV format compatible with
the visualization system in the EDA directory.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from ..models.traffic_model import TrafficSimulationModel
from ..agents.vehicle import Vehicle
from .bbox_utils import (
    calculate_bounding_box_corners,
    world_to_image_coordinates,
    get_world_bounds_from_road_network
)


class SimulationToCSVConverter:
    """
    Converts MESA simulation frames to CSV format compatible with visualization.
    
    The CSV format matches the Korea crossing dataset format:
    - frame: Frame number
    - track_id: Vehicle ID
    - timestamp: Time stamp
    - center_x, center_y: Center of bounding box
    - x1, y1, x2, y2, x3, y3, x4, y4: Bounding box corners
    - class_id: Vehicle class (optional)
    """
    
    def __init__(self,
                 model: TrafficSimulationModel,
                 image_width: int = 1920,
                 image_height: int = 1080,
                 world_bounds: Optional[Tuple[float, float, float, float]] = None):
        """
        Initialize the converter.
        
        Args:
            model: MESA traffic simulation model
            image_width: Width of target image in pixels
            image_height: Height of target image in pixels
            world_bounds: World coordinate bounds (min_x, min_y, max_x, max_y)
            offset_x: X offset for coordinate transformation
            offset_y: Y offset for coordinate transformation
            scale: Scale factor for coordinate transformation
        """
        self.model = model
        
        # Image dimensions
        self.image_width = image_width
        self.image_height = image_height
        
        # Coordinate transformation parameters
        if world_bounds is None:
            self.world_bounds = get_world_bounds_from_road_network(model.road_network)
        else:
            self.world_bounds = world_bounds
        
        self.offset_x = model.offset_x
        self.offset_y = model.offset_y
        self.scale = model.scale
        
        # Vehicle dimensions (in meters) - typical car dimensions
        self.default_length = 4.5  # meters
        self.default_width = 2.0   # meters
        
        # Pixels per meter - use separate scales for positioning and bounding boxes
        # Position scale: small (0.3507 px/m) to fit roads within image bounds
        # Bounding box scale: larger (~14 px/m) to match real data sizes and make cars visible
        # Real data shows cars ~60-70 pixels for 4.5m length = ~14 px/m
        self.position_scale = model.scale  # For coordinate transformation (keeps vehicles in bounds)
        self.bbox_pixels_per_meter = model.scale  # For bounding box size (makes cars visible)
        
        # Frame counter
        self.current_frame = 0
    
    def get_frame_data(self, frame_number: Optional[int] = None) -> pd.DataFrame:
        """
        Get bounding box data for the current simulation state.
        
        Args:
            frame_number: Frame number (if None, uses current_frame)
        
        Returns:
            DataFrame with columns: frame, track_id, timestamp, center_x, center_y,
            x1, y1, x2, y2, x3, y3, x4, y4, class_id
        """
        if frame_number is None:
            frame_number = self.current_frame
        
        rows = []
        
        for vehicle in self.model.vehicles:
            # Get vehicle position and angle
            world_x, world_y = vehicle.get_visual_position()
            angle = vehicle.get_visual_angle()
            
            # Get vehicle dimensions in meters
            length_meters = vehicle.length
            width_meters = vehicle.width
            
            # Transform center to image coordinates
            center_x, center_y = world_to_image_coordinates(
                world_x, world_y,
                self.image_width, self.image_height,
                self.world_bounds,
                self.offset_x, self.offset_y,
                self.position_scale
            )
            
            # Convert vehicle dimensions to pixels
            # Use separate scale for bounding boxes to ensure correct size
            length_pixels = length_meters * self.bbox_pixels_per_meter
            width_pixels = width_meters * self.bbox_pixels_per_meter
            
            # Calculate bounding box corners directly in image coordinates
            # Note: angle is already in world coordinates, which is fine
            bbox_corners_image = calculate_bounding_box_corners(
                center_x, center_y, length_pixels, width_pixels, angle
            )
            
            # Extract corners and clip to image bounds
            corners_image = [
                max(0, min(self.image_width, bbox_corners_image[0])),  # x1, clipped
                max(0, min(self.image_height, bbox_corners_image[1])),  # y1, clipped
                max(0, min(self.image_width, bbox_corners_image[2])),  # x2, clipped
                max(0, min(self.image_height, bbox_corners_image[3])),  # y2, clipped
                max(0, min(self.image_width, bbox_corners_image[4])),  # x3, clipped
                max(0, min(self.image_height, bbox_corners_image[5])),  # y3, clipped
                max(0, min(self.image_width, bbox_corners_image[6])),  # x4, clipped
                max(0, min(self.image_height, bbox_corners_image[7]))   # y4, clipped
            ]
            
            # Also clip center to bounds
            center_x = max(0, min(self.image_width, center_x))
            center_y = max(0, min(self.image_height, center_y))
            
            # Create row
            row = {
                'frame': frame_number,
                'track_id': vehicle.unique_id,
                'timestamp': self.model.current_time,
                'center_x': center_x,
                'center_y': center_y,
                'x1': corners_image[0],
                'y1': corners_image[1],
                'x2': corners_image[2],
                'y2': corners_image[3],
                'x3': corners_image[4],
                'y3': corners_image[5],
                'x4': corners_image[6],
                'y4': corners_image[7],
                'class_id': 1  # Default to car class
            }
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        return df
    
    def run_simulation_and_save(self,
                                output_path: str,
                                num_frames: int = 300,
                                frames_per_second: float = 8.0,
                                dt_per_frame: Optional[float] = None):
        """
        Run simulation for specified number of frames and save to CSV.
        
        Args:
            output_path: Path to output CSV file
            num_frames: Number of frames to simulate
            frames_per_second: Target frames per second (default: 8.0 for 8 detections/second)
            dt_per_frame: Time step per frame (if None, uses model.time_step)
        """
        if dt_per_frame is None:
            dt_per_frame = self.model.time_step
        
        # Now dt_per_frame is guaranteed to be float
        dt_per_frame_float: float = dt_per_frame
        
        # Calculate steps per frame to achieve target FPS
        # If fps = 8, time per frame = 1/8 = 0.125 seconds
        # If time_step = 0.1, steps_per_frame = 0.125 / 0.1 = 1.25
        # Round to nearest integer
        time_per_frame = 1.0 / frames_per_second
        steps_per_frame = max(1, round(time_per_frame / dt_per_frame_float))
        
        print(f"  Target FPS: {frames_per_second} (time per frame: {time_per_frame:.3f}s)")
        print(f"  Simulation time_step: {dt_per_frame_float}s")
        print(f"  Steps per frame: {steps_per_frame} (actual FPS: {1.0/(steps_per_frame * dt_per_frame_float):.2f})")
        
        all_frames_data = []
        
        print(f"Running simulation for {num_frames} frames...")
        print(f"Simulation steps per frame: {steps_per_frame}")
        
        for frame_idx in range(num_frames):
            # Run simulation steps for this frame
            for _ in range(steps_per_frame):
                self.model.step()
            
            # Get frame data (always create frame, even if empty)
            frame_data = self.get_frame_data(frame_idx)
            all_frames_data.append(frame_data)
            
            if (frame_idx + 1) % 10 == 0:
                print(f"Processed frame {frame_idx + 1}/{num_frames} ({len(frame_data)} vehicles)")
        
        # Concatenate all frames
        combined_df = pd.concat(all_frames_data, ignore_index=True)
        
        # Save to CSV
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Saved simulation data to {output_path}")
        print(f"   Total frames: {num_frames}")
        print(f"   Total detections: {len(combined_df)}")
        print(f"   Unique vehicles: {combined_df['track_id'].nunique()}")
        
        return combined_df

