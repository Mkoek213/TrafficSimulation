#!/usr/bin/env python3
"""
Interactive Centerline Marking Tool

This tool allows you to:
1. Mark centerlines (single lines) where cars can go
2. Set spawn points (on first point of line)
3. Mark lane-changing connections (create transition paths between lanes)

Usage:
    python mark_centerlines_ui.py --image eda/data/media/SiteA.jpg --output eda/data/lanes.json
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import math


class CenterlineMarkerUI:
    """Interactive tool for marking centerlines, spawn points, and lane-changing connections."""
    
    def __init__(self, image_path: str, output_path: str):
        """
        Initialize the centerline marker.
        
        Args:
            image_path: Path to the background image
            output_path: Path to save lane data
        """
        self.image_path = image_path
        self.output_path = output_path
        
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Load existing lanes if file exists
        self.lanes_data = self._load_existing_lanes()
        
        # Current state
        self.mode = 'draw'  # 'draw', 'spawn', 'connect'
        self.current_lane_id = len(self.lanes_data.get('lanes', []))
        self.current_lane: List[Tuple[int, int]] = []
        self.selected_lane_id: Optional[int] = None
        self.connection_source_lane_id: Optional[int] = None
        self.connection_source_point: Optional[Tuple[int, int]] = None
        self.connection_target_lane_id: Optional[int] = None
        self.connection_points: List[Tuple[int, int]] = []
        
        # Zoom and pan state
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Initialize window size
        self.win_width = 1920
        self.win_height = 1080
        
        # Setup window
        self.window_name = "Centerline Marker - Draw lanes, set spawn points, create connections"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Instructions
        self._print_instructions()
        self._redraw()
    
    def _load_existing_lanes(self) -> Dict:
        """Load existing lanes from JSON file if it exists, but start fresh."""
        # Always start with a fresh structure - ignore existing lanes
        # We only preserve metadata like image_path, image_width, image_height
        if Path(self.output_path).exists():
            with open(self.output_path, 'r') as f:
                data = json.load(f)
                # Keep only metadata, clear lanes and connections
                return {
                    'image_path': data.get('image_path', self.image_path),
                    'image_width': data.get('image_width', self.width),
                    'image_height': data.get('image_height', self.height),
                    'lanes': [],  # Start fresh - no existing lanes
                    'lane_connections': []  # Start fresh - no existing connections
                }
        else:
            # Create new structure
            return {
                'image_path': self.image_path,
                'image_width': self.width,
                'image_height': self.height,
                'lanes': [],
                'lane_connections': []  # For lane-changing paths
            }
    
    def _print_instructions(self):
        """Print instructions for the user."""
        print("="*70)
        print("CENTERLINE MARKING TOOL")
        print("="*70)
        print("\nModes:")
        print("  DRAW MODE (default):")
        print("    - LEFT CLICK: Add points to current centerline")
        print("    - 'n': Start NEW centerline")
        print("    - 's': SAVE current centerline")
        print("    - 'd': Delete current centerline")
        print("    - 'u': UNDO last point")
        print("    - '1-9, 0': Select lane by number")
        print("")
        print("  SPAWN MODE ('t' key):")
        print("    - '1-9, 0': Select lane by number")
        print("    - 't': Toggle spawn point for selected lane (on/off)")
        print("    - Spawn point is always at FIRST point of centerline")
        print("")
        print("  CONNECTION MODE ('c' key):")
        print("    - '1-9, 0': Select source lane")
        print("    - LEFT CLICK on source lane: Set connection start point")
        print("    - LEFT CLICK on target lane: Set connection end point")
        print("    - Continue clicking to create connection path")
        print("    - 'u': UNDO last connection point")
        print("    - 'ENTER': Finish connection")
        print("    - 'ESC': Cancel connection")
        print("")
        print("General:")
        print("  - 'm': Toggle mode (draw/spawn/connect)")
        print("  - 'w': SAVE all data to JSON")
        print("  - 'q': QUIT")
        print("  - RIGHT MOUSE + DRAG: Pan")
        print("  - +/-: Zoom in/out")
        print("  - 'r': Reset zoom/pan")
        print("="*70)
    
    def _screen_to_image(self, x: int, y: int) -> Tuple[int, int]:
        """Convert screen coordinates to image coordinates."""
        # Get actual window size
        try:
            win_rect = cv2.getWindowImageRect(self.window_name)
            if win_rect and win_rect[2] > 0 and win_rect[3] > 0:
                self.win_width = win_rect[2]
                self.win_height = win_rect[3]
        except:
            pass  # Use default if can't get window size
        
        # Calculate scale to fit image in window (same as mark_lanes.py)
        scale_to_fit_x = self.win_width / self.width
        scale_to_fit_y = self.win_height / self.height
        scale_to_fit = min(scale_to_fit_x, scale_to_fit_y) * 0.95  # 95% to leave some margin
        effective_zoom = scale_to_fit * self.zoom_factor
        
        # If image fits in window, account for padding (centering)
        if effective_zoom * self.width <= self.win_width and effective_zoom * self.height <= self.win_height:
            # Image is centered - adjust for padding
            scaled_width = int(self.width * effective_zoom)
            scaled_height = int(self.height * effective_zoom)
            pad_left = (self.win_width - scaled_width) // 2
            pad_top = (self.win_height - scaled_height) // 2
            img_x = int((x - pad_left) / effective_zoom)
            img_y = int((y - pad_top) / effective_zoom)
        else:
            # Image is larger - account for pan
            img_x = int((x + self.pan_x) / effective_zoom)
            img_y = int((y + self.pan_y) / effective_zoom)
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, self.width - 1))
        img_y = max(0, min(img_y, self.height - 1))
        return img_x, img_y
    
    def _find_lane_at_point(self, point: Tuple[int, int]) -> Optional[int]:
        """Find which lane (if any) is closest to the given point."""
        min_dist = float('inf')
        closest_lane_id = None
        
        for lane_info in self.lanes_data.get('lanes', []):
            lane_id = lane_info['lane_id']
            # Only use centerline_points, ignore polygon points
            points = lane_info.get('centerline_points', [])
            if not points or len(points) < 2:
                continue
            
            # Find closest point on this lane's centerline
            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]
                
                # Calculate distance from point to line segment
                line_vec = (p2[0] - p1[0], p2[1] - p1[1])
                point_vec = (point[0] - p1[0], point[1] - p1[1])
                
                line_len_sq = line_vec[0]**2 + line_vec[1]**2
                if line_len_sq == 0:
                    dist = math.sqrt((point[0] - p1[0])**2 + (point[1] - p1[1])**2)
                else:
                    t = max(0, min(1, (point_vec[0] * line_vec[0] + point_vec[1] * line_vec[1]) / line_len_sq))
                    proj = (p1[0] + t * line_vec[0], p1[1] + t * line_vec[1])
                    dist = math.sqrt((point[0] - proj[0])**2 + (point[1] - proj[1])**2)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_lane_id = lane_id
        
        return closest_lane_id if min_dist < 50 else None
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        # Handle right mouse button for panning
        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y
            return  # Don't process anything else when starting pan
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False
            return  # Don't process anything else when ending pan
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning and (flags & cv2.EVENT_FLAG_RBUTTON):
                # Update pan position (same as mark_lanes.py)
                self.pan_x += x - self.pan_start_x
                self.pan_y += y - self.pan_start_y
                self.pan_start_x = x
                self.pan_start_y = y
                self._redraw()
            return
        
        # Handle left mouse button for drawing (only if not panning)
        if event == cv2.EVENT_LBUTTONDOWN:
            # Ensure we're not panning
            if self.is_panning:
                self.is_panning = False  # Clear panning flag
                return
            
            # Convert screen coordinates to image coordinates
            img_x, img_y = self._screen_to_image(x, y)
            
            if self.mode == 'draw':
                # Add point to current centerline
                self.current_lane.append((img_x, img_y))
                print(f"✓ Added point: ({img_x}, {img_y}) - Total points: {len(self.current_lane)}")
                self._redraw()
            elif self.mode == 'connect':
                # Handle connection mode
                clicked_lane_id = self._find_lane_at_point((img_x, img_y))
                
                if clicked_lane_id is None:
                    # Clicked in empty space - add point to connection path
                    if self.connection_source_point:
                        self.connection_points.append((img_x, img_y))
                        self._redraw()
                    return
                
                if self.connection_source_lane_id is None:
                    # First click - set source lane
                    self.connection_source_lane_id = clicked_lane_id
                    self.connection_source_point = (img_x, img_y)
                    self.connection_points = [(img_x, img_y)]
                    print(f"✓ Connection source: Lane {clicked_lane_id}")
                    self._redraw()
                elif clicked_lane_id != self.connection_source_lane_id:
                    # Clicked on different lane - set as target
                    if self.connection_target_lane_id is None:
                        self.connection_target_lane_id = clicked_lane_id
                        self.connection_points.append((img_x, img_y))
                        print(f"✓ Connection target: Lane {clicked_lane_id}")
                        self._redraw()
                    else:
                        # Continue adding points to connection path
                        self.connection_points.append((img_x, img_y))
                        self._redraw()
                else:
                    # Clicked on same lane - add point to path
                    self.connection_points.append((img_x, img_y))
                    self._redraw()
    
    def _get_lane_by_id(self, lane_id: int) -> Optional[Dict]:
        """Get lane data by ID."""
        for lane in self.lanes_data.get('lanes', []):
            if lane['lane_id'] == lane_id:
                return lane
        return None
    
    def _redraw(self):
        """Redraw the display image."""
        if self.image is None:
            return
        working_image = self.image.copy()
        
        # Draw all lanes
        for lane_info in self.lanes_data.get('lanes', []):
            lane_id = lane_info['lane_id']
            # Only use centerline_points, ignore polygon points
            points = lane_info.get('centerline_points', [])
            
            if not points or len(points) < 2:
                continue
            
            # Determine color based on selection and spawn status
            spawn_enabled = lane_info.get('spawn_enabled', False)
            if lane_id == self.selected_lane_id:
                color = (0, 255, 0)  # Green for selected
                thickness = 3
            elif spawn_enabled:
                color = (255, 255, 0)  # Yellow for spawn-enabled
                thickness = 2
            else:
                color = (255, 255, 255)  # White for normal
                thickness = 1
            
            # Draw centerline
            for i in range(len(points) - 1):
                pt1 = tuple(points[i])
                pt2 = tuple(points[i + 1])
                cv2.line(working_image, pt1, pt2, color, thickness)
            
            # Draw spawn point indicator
            if spawn_enabled and points:
                spawn_point = points[0]
                cv2.circle(working_image, tuple(spawn_point), 8, (0, 255, 255), -1)
                cv2.putText(working_image, "SPAWN", 
                           (spawn_point[0] + 10, spawn_point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw lane ID
            if points:
                first_point = points[0]
                cv2.putText(working_image, f"L{lane_id}", 
                           (first_point[0] + 10, first_point[1] + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw current centerline being drawn
        if self.mode == 'draw' and self.current_lane:
            # Draw lines between points
            for i in range(len(self.current_lane) - 1):
                pt1 = self.current_lane[i]
                pt2 = self.current_lane[i + 1]
                cv2.line(working_image, pt1, pt2, (0, 255, 255), 3)  # Cyan, thicker line
            # Draw points as larger circles
            for pt in self.current_lane:
                cv2.circle(working_image, pt, 8, (0, 255, 255), -1)  # Cyan filled circle, larger
                cv2.circle(working_image, pt, 8, (0, 0, 0), 2)  # Black border for visibility
        
        # Draw connection being created
        if self.mode == 'connect' and self.connection_points:
            # Draw connection path
            for i in range(len(self.connection_points) - 1):
                pt1 = self.connection_points[i]
                pt2 = self.connection_points[i + 1]
                cv2.line(working_image, pt1, pt2, (255, 0, 255), 3)
            # Draw points
            for pt in self.connection_points:
                cv2.circle(working_image, pt, 6, (255, 0, 255), -1)
            
            # Draw source/target indicators
            if self.connection_source_point:
                cv2.circle(working_image, self.connection_source_point, 10, (0, 255, 0), 2)
                cv2.putText(working_image, "START", 
                           (self.connection_source_point[0] + 15, self.connection_source_point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if len(self.connection_points) > 1:
                end_point = self.connection_points[-1]
                cv2.circle(working_image, end_point, 10, (0, 0, 255), 2)
                cv2.putText(working_image, "END", 
                           (end_point[0] + 15, end_point[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw existing lane connections
        for conn in self.lanes_data.get('lane_connections', []):
            points = conn.get('points', [])
            if len(points) >= 2:
                for i in range(len(points) - 1):
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i + 1])
                    cv2.line(working_image, pt1, pt2, (128, 0, 128), 2)
        
        # Get actual window size
        try:
            win_rect = cv2.getWindowImageRect(self.window_name)
            if win_rect and win_rect[2] > 0 and win_rect[3] > 0:
                self.win_width = win_rect[2]
                self.win_height = win_rect[3]
        except:
            pass  # Use default if can't get window size
        
        # Calculate scale to fit image in window (same as mark_lanes.py)
        scale_to_fit_x = self.win_width / self.width
        scale_to_fit_y = self.win_height / self.height
        scale_to_fit = min(scale_to_fit_x, scale_to_fit_y) * 0.95  # 95% to leave some margin
        
        # Apply zoom (relative to fit scale)
        effective_zoom = scale_to_fit * self.zoom_factor
        
        # Resize image based on effective zoom
        if effective_zoom != 1.0:
            new_width = int(self.width * effective_zoom)
            new_height = int(self.height * effective_zoom)
            zoomed_image = cv2.resize(working_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        else:
            zoomed_image = working_image
        
        # If zoomed image fits in window, center it and show whole image
        if zoomed_image.shape[0] <= self.win_height and zoomed_image.shape[1] <= self.win_width:
            # Image fits - center it
            pad_top = (self.win_height - zoomed_image.shape[0]) // 2
            pad_bottom = self.win_height - zoomed_image.shape[0] - pad_top
            pad_left = (self.win_width - zoomed_image.shape[1]) // 2
            pad_right = self.win_width - zoomed_image.shape[1] - pad_left
            self.display_image = cv2.copyMakeBorder(zoomed_image, pad_top, pad_bottom, pad_left, pad_right,
                                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
            # Reset pan when showing full image
            self.pan_x = 0
            self.pan_y = 0
        else:
            # Image is larger than window - apply pan (crop to window size)
            # Constrain pan to valid range: 0 to (zoomed_image_size - window_size)
            max_pan_x = max(0, zoomed_image.shape[1] - self.win_width)
            max_pan_y = max(0, zoomed_image.shape[0] - self.win_height)
            
            # Clamp pan coordinates to valid range
            self.pan_x = max(0, min(self.pan_x, max_pan_x))
            self.pan_y = max(0, min(self.pan_y, max_pan_y))
            
            # Calculate crop region
            start_x = int(self.pan_x)
            start_y = int(self.pan_y)
            end_x = min(zoomed_image.shape[1], start_x + self.win_width)
            end_y = min(zoomed_image.shape[0], start_y + self.win_height)
            
            # Crop image
            if start_x < end_x and start_y < end_y:
                cropped = zoomed_image[start_y:end_y, start_x:end_x]
                # Pad if needed to fill window
                if cropped.shape[0] < self.win_height or cropped.shape[1] < self.win_width:
                    pad_bottom = max(0, self.win_height - cropped.shape[0])
                    pad_right = max(0, self.win_width - cropped.shape[1])
                    cropped = cv2.copyMakeBorder(cropped, 0, pad_bottom, 0, pad_right, 
                                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
                self.display_image = cropped
            else:
                self.display_image = zoomed_image
        
        # Draw status text on display image
        status_y = 30
        mode_colors = {
            'draw': (0, 255, 255),
            'spawn': (255, 255, 0),
            'connect': (255, 0, 255)
        }
        cv2.putText(self.display_image, f"Mode: {self.mode.upper()}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_colors.get(self.mode, (255, 255, 255)), 2)
        status_y += 35
        
        if self.selected_lane_id is not None:
            cv2.putText(self.display_image, f"Selected Lane: {self.selected_lane_id}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            status_y += 30
        
        if self.mode == 'draw' and self.current_lane:
            cv2.putText(self.display_image, f"Current lane: {len(self.current_lane)} points", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            status_y += 30
            cv2.putText(self.display_image, "Press 's' to save, 'n' for new lane", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if self.mode == 'connect':
            if self.connection_source_lane_id is not None:
                cv2.putText(self.display_image, f"Connection: Lane {self.connection_source_lane_id} -> ...", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                status_y += 30
                if self.connection_target_lane_id is not None:
                    cv2.putText(self.display_image, f"Target: Lane {self.connection_target_lane_id}", 
                               (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    status_y += 30
                cv2.putText(self.display_image, "Press ENTER to finish, ESC to cancel", 
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow(self.window_name, self.display_image)
        cv2.waitKey(1)  # Force window update
    
    def _apply_zoom_and_pan(self, img: np.ndarray) -> np.ndarray:
        """Apply zoom and pan transformations to the image for display."""
        if self.zoom_factor == 1.0 and self.pan_x == 0 and self.pan_y == 0:
            return img
        
        # Scale the image
        if self.zoom_factor != 1.0:
            new_width = int(self.width * self.zoom_factor)
            new_height = int(self.height * self.zoom_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Apply pan by creating a larger canvas
        if self.pan_x != 0 or self.pan_y != 0:
            # Get window size to determine canvas size
            window_rect = cv2.getWindowImageRect(self.window_name)
            if window_rect:
                win_w, win_h = window_rect[2], window_rect[3]
            else:
                win_w, win_h = int(self.width * self.zoom_factor), int(self.height * self.zoom_factor)
            
            # Create canvas large enough for panning
            canvas_width = max(win_w, int(self.width * self.zoom_factor) + abs(self.pan_x))
            canvas_height = max(win_h, int(self.height * self.zoom_factor) + abs(self.pan_y))
            
            # Create black canvas
            canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Calculate where to place the image
            # Center the image, then apply pan offset
            img_h, img_w = img.shape[:2]
            center_x = (canvas_width - img_w) // 2
            center_y = (canvas_height - img_h) // 2
            
            x_offset = center_x + self.pan_x
            y_offset = center_y + self.pan_y
            
            # Clamp to canvas bounds
            x_start = max(0, x_offset)
            y_start = max(0, y_offset)
            x_end = min(canvas_width, x_offset + img_w)
            y_end = min(canvas_height, y_offset + img_h)
            
            # Calculate source region
            src_x_start = max(0, -x_offset)
            src_y_start = max(0, -y_offset)
            src_x_end = src_x_start + (x_end - x_start)
            src_y_end = src_y_start + (y_end - y_start)
            
            # Place the image on canvas
            if y_end > y_start and x_end > x_start:
                canvas[y_start:y_end, x_start:x_end] = img[src_y_start:src_y_end, src_x_start:src_x_end]
            
            img = canvas
        
        return img
    
    def _undo_last_point(self):
        """Undo the last drawn point."""
        if self.mode == 'draw':
            if self.current_lane:
                removed_point = self.current_lane.pop()
                print(f"✓ Removed last point: ({removed_point[0]}, {removed_point[1]})")
                self._redraw()
            else:
                print("⚠ No points to undo")
        elif self.mode == 'connect':
            if len(self.connection_points) > 1:
                # Don't remove the first point (source point)
                removed_point = self.connection_points.pop()
                # If we removed the target point, clear target_lane_id
                if self.connection_target_lane_id and len(self.connection_points) == 1:
                    self.connection_target_lane_id = None
                    print(f"✓ Removed last point and cleared target")
                else:
                    print(f"✓ Removed last point: ({removed_point[0]}, {removed_point[1]})")
                self._redraw()
            else:
                print("⚠ Cannot undo - need at least the source point")
        else:
            print("⚠ Undo only works in DRAW or CONNECT mode")
    
    def _save_current_lane(self):
        """Save the current centerline."""
        if len(self.current_lane) < 2:
            print("⚠ Need at least 2 points to save a lane")
            return
        
        lane_id = self.current_lane_id
        lane_data = {
            'lane_id': lane_id,
            'lane_number': lane_id,
            'centerline_points': self.current_lane.copy(),  # Only centerline_points, no polygon points
            'spawn_enabled': False,
            'num_points': len(self.current_lane)
        }
        
        self.lanes_data['lanes'].append(lane_data)
        print(f"✓ Saved lane {lane_id} with {len(self.current_lane)} centerline points")
        
        self.current_lane_id += 1
        self.current_lane = []
        self._redraw()
    
    def _finish_connection(self):
        """Finish the current connection."""
        if len(self.connection_points) < 2:
            print("⚠ Need at least 2 points for connection")
            return
        
        if self.connection_source_lane_id is None or self.connection_target_lane_id is None:
            print("⚠ Need both source and target lanes for connection")
            return
        
        connection_data = {
            'source_lane_id': self.connection_source_lane_id,
            'target_lane_id': self.connection_target_lane_id,
            'points': self.connection_points.copy()
        }
        
        if 'lane_connections' not in self.lanes_data:
            self.lanes_data['lane_connections'] = []
        
        self.lanes_data['lane_connections'].append(connection_data)
        print(f"✓ Saved connection: Lane {self.connection_source_lane_id} -> Lane {self.connection_target_lane_id}")
        
        # Reset connection state
        self.connection_source_lane_id = None
        self.connection_target_lane_id = None
        self.connection_source_point = None
        self.connection_points = []
        self._redraw()
    
    def _cancel_connection(self):
        """Cancel the current connection."""
        self.connection_source_lane_id = None
        self.connection_target_lane_id = None
        self.connection_source_point = None
        self.connection_points = []
        print("✓ Cancelled connection")
        self._redraw()
    
    def _toggle_spawn(self):
        """Toggle spawn point for selected lane."""
        if self.selected_lane_id is None:
            print("⚠ Please select a lane first (press number key)")
            return
        
        lane = self._get_lane_by_id(self.selected_lane_id)
        if not lane:
            print(f"⚠ Lane {self.selected_lane_id} not found")
            return
        
        current_state = lane.get('spawn_enabled', False)
        lane['spawn_enabled'] = not current_state
        
        if lane['spawn_enabled']:
            print(f"✓ Enabled spawn point for lane {self.selected_lane_id} (at first point)")
        else:
            print(f"✓ Disabled spawn point for lane {self.selected_lane_id}")
        
        self._redraw()
    
    def _select_lane(self, lane_idx: int):
        """Select a lane by index."""
        lanes = self.lanes_data.get('lanes', [])
        if 0 <= lane_idx < len(lanes):
            self.selected_lane_id = lanes[lane_idx]['lane_id']
            print(f"✓ Selected lane {self.selected_lane_id}")
            self._redraw()
        else:
            print(f"⚠ Lane index {lane_idx} out of range (available: {len(lanes)} lanes)")
    
    def _save_to_file(self):
        """Save all data to JSON file."""
        # Ensure image_path is relative
        if Path(self.image_path).is_absolute():
            # Try to make it relative to output file
            output_dir = Path(self.output_path).parent
            try:
                rel_path = Path(self.image_path).relative_to(Path.cwd())
                self.lanes_data['image_path'] = str(rel_path)
            except:
                self.lanes_data['image_path'] = self.image_path
        else:
            self.lanes_data['image_path'] = self.image_path
        
        self.lanes_data['image_width'] = self.width
        self.lanes_data['image_height'] = self.height
        
        with open(self.output_path, 'w') as f:
            json.dump(self.lanes_data, f, indent=2)
        
        num_lanes = len(self.lanes_data.get('lanes', []))
        num_connections = len(self.lanes_data.get('lane_connections', []))
        print(f"\n✓ Saved to {self.output_path}")
        print(f"  - {num_lanes} lane(s)")
        print(f"  - {num_connections} connection(s)")
    
    def run(self):
        """Run the interactive marking tool."""
        self._redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('w'):
                self._save_to_file()
            elif key == ord('m'):
                # Toggle mode
                modes = ['draw', 'spawn', 'connect']
                current_idx = modes.index(self.mode)
                self.mode = modes[(current_idx + 1) % len(modes)]
                print(f"✓ Switched to {self.mode.upper()} mode")
                # Reset connection state when switching modes
                if self.mode != 'connect':
                    self._cancel_connection()
                self._redraw()
            elif key == ord('n') and self.mode == 'draw':
                # Start new lane
                if self.current_lane:
                    self._save_current_lane()
                self.current_lane = []
                print("✓ Started new lane")
                self._redraw()
            elif key == ord('s') and self.mode == 'draw':
                # Save current lane
                self._save_current_lane()
            elif key == ord('d') and self.mode == 'draw':
                # Delete current lane
                self.current_lane = []
                print("✓ Cleared current lane")
                self._redraw()
            elif key == ord('u'):
                # Undo last point
                self._undo_last_point()
            elif key == ord('t') and self.mode == 'spawn':
                # Toggle spawn point
                self._toggle_spawn()
            elif key == 13 and self.mode == 'connect':  # ENTER key
                # Finish connection
                self._finish_connection()
            elif key == 27 and self.mode == 'connect':  # ESC key
                # Cancel connection
                self._cancel_connection()
            elif key >= ord('0') and key <= ord('9'):
                # Select lane by number
                lanes = self.lanes_data.get('lanes', [])
                if not lanes:
                    print("⚠ No lanes available")
                    continue
                
                if key == ord('0'):
                    lane_idx = 9
                else:
                    lane_idx = key - ord('1')
                
                self._select_lane(lane_idx)
            elif key == ord('+') or key == ord('='):
                self.zoom_factor = min(5.0, self.zoom_factor + 0.1)
                self._redraw()
            elif key == ord('-') or key == ord('_'):
                self.zoom_factor = max(0.2, self.zoom_factor - 0.1)
                self._redraw()
            elif key == ord('r'):
                self.zoom_factor = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self._redraw()
            elif key == ord('0'):
                # Reset zoom to 1.0 (fit to window)
                self.zoom_factor = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self._redraw()
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Mark centerlines, spawn points, and lane-changing connections')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to background image')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to save lanes.json file')
    
    args = parser.parse_args()
    
    marker = CenterlineMarkerUI(args.image, args.output)
    marker.run()


if __name__ == '__main__':
    main()

