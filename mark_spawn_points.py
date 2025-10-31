#!/usr/bin/env python3
"""
Interactive Spawn Point Marking Tool

Loads existing lanes and allows you to mark spawn points (entry points) for each lane.
Vehicles will spawn at these marked points.

Usage:
    python mark_spawn_points.py --lanes eda/data/lanes.json
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict


class SpawnPointMarker:
    """Interactive tool for marking spawn points on lanes."""
    
    def __init__(self, lanes_json_path: str):
        """
        Initialize the spawn point marker.
        
        Args:
            lanes_json_path: Path to lanes JSON file
        """
        self.lanes_json_path = lanes_json_path
        
        # Load lanes data
        with open(lanes_json_path) as f:
            self.lanes_data = json.load(f)
        
        # Load image
        image_path = self.lanes_data.get('image_path', 'eda/data/media/SiteA.jpg')
        if not Path(image_path).is_absolute():
            # Relative path - try project root first, then lanes directory
            project_root = Path.cwd()
            project_path = project_root / image_path
            
            if project_path.exists():
                image_path = str(project_path)
            else:
                # Try relative to lanes directory
                lanes_dir = Path(lanes_json_path).parent
                lanes_path = lanes_dir / image_path
                if lanes_path.exists():
                    image_path = str(lanes_path)
                else:
                    # Last resort: try project root with the path as-is
                    image_path = str(project_path)
        
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Initialize spawn points storage
        # Map: lane_id -> (x, y) spawn point
        self.spawn_points: Dict[int, Tuple[int, int]] = {}
        
        # Load existing spawn points if they exist
        if 'spawn_points' in self.lanes_data:
            for lane_id, point in self.lanes_data['spawn_points'].items():
                self.spawn_points[int(lane_id)] = tuple(point)
        
        # Current lane being edited
        self.current_lane_id = None
        self.lane_ids = sorted([lane['lane_id'] for lane in self.lanes_data['lanes']])
        if self.lane_ids:
            self.current_lane_id = self.lane_ids[0]
        
        # Zoom and pan state (same as mark_lanes.py)
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Setup window (same as mark_lanes.py)
        self.window_name = "Spawn Point Marker - Click to mark spawn point, 'n'/'p' to switch lanes, 's' to save, 'q' to quit, 'f' for fullscreen"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Initialize window size (same as mark_lanes.py)
        self.win_width = 1920
        self.win_height = 1080
        
        # Fullscreen state
        self.is_fullscreen = False
        
        # Instructions
        self._print_instructions()
    
    def _print_instructions(self):
        """Print instructions for the user."""
        print("="*60)
        print("SPAWN POINT MARKING TOOL")
        print("="*60)
        print("\nInstructions:")
        print("  1. Click LEFT MOUSE to mark spawn point for current lane")
        print("  2. RIGHT MOUSE + DRAG to pan the image")
        print("  3. MOUSE WHEEL or +/- keys to zoom in/out")
        print("  4. Press 'n' or ARROW RIGHT to go to NEXT lane")
        print("  5. Press 'p' or ARROW LEFT to go to PREVIOUS lane")
        print("  6. Press 's' to SAVE spawn points to file")
        print("  7. Press 'q' to QUIT")
        print("  8. Press 'c' to CLEAR spawn point for current lane")
        print("  9. Press '+' or '-' to ZOOM in/out")
        print(" 10. Press '0' to RESET zoom to fit window")
        print(" 11. Press 'r' to RESET zoom/pan")
        print(" 12. Press 'f' to TOGGLE fullscreen")
        print("\nSpawn Points:")
        print("  Mark the entry point where vehicles should spawn for each lane.")
        print("  This should be at the missing edge of the lane polygon.")
        print("="*60)
        print(f"\nImage size: {self.width}x{self.height}")
        print(f"Total lanes: {len(self.lane_ids)}")
        print(f"Output will be saved to: {self.lanes_json_path}")
        print("\nStarting...\n")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events (same approach as mark_lanes.py)."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert screen coordinates to image coordinates (same as mark_lanes.py)
            scale_to_fit_x = self.win_width / self.width
            scale_to_fit_y = self.win_height / self.height
            scale_to_fit = min(scale_to_fit_x, scale_to_fit_y) * 0.95
            effective_zoom = scale_to_fit * self.zoom_factor
            
            # If image fits in window, account for padding
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
            
            if self.current_lane_id is not None:
                # Mark spawn point for current lane
                self.spawn_points[self.current_lane_id] = (int(img_x), int(img_y))
                print(f"Marked spawn point for lane {self.current_lane_id}: ({int(img_x)}, {int(img_y)})")
                self._redraw()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Start panning
            self.is_panning = True
            self.pan_start_x = x
            self.pan_start_y = y
        
        elif event == cv2.EVENT_RBUTTONUP:
            # Stop panning
            self.is_panning = False
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                # Pan the image (same as mark_lanes.py)
                dx = x - self.pan_start_x
                dy = y - self.pan_start_y
                scale_to_fit_x = self.win_width / self.width
                scale_to_fit_y = self.win_height / self.height
                scale_to_fit = min(scale_to_fit_x, scale_to_fit_y) * 0.95
                effective_zoom = scale_to_fit * self.zoom_factor
                self.pan_x += dx / effective_zoom
                self.pan_y += dy / effective_zoom
                self.pan_start_x = x
                self.pan_start_y = y
                self._redraw()
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom (same as mark_lanes.py)
            zoom_delta = 0.1
            if flags > 0:  # Scroll up - zoom in
                self.zoom_factor = min(5.0, self.zoom_factor + zoom_delta)
            else:  # Scroll down - zoom out
                self.zoom_factor = max(0.2, self.zoom_factor - zoom_delta)
            self._redraw()
    
    def _redraw(self):
        """Redraw the display image with lanes and spawn points (same approach as mark_lanes.py)."""
        if self.image is None:
            return
        # Create working copy
        working_image = self.image.copy()
        
        # Draw all lanes
        for lane_info in self.lanes_data['lanes']:
            lane_id = lane_info['lane_id']
            points = lane_info['points']
            
            # Draw lane polygon
            if len(points) >= 3:
                pts = np.array(points, dtype=np.int32)
                # Highlight current lane in green, others in blue
                color = (0, 255, 0) if lane_id == self.current_lane_id else (100, 100, 255)
                thickness = 3 if lane_id == self.current_lane_id else 2
                cv2.polylines(working_image, [pts], True, color, thickness)
            
            # Draw spawn point if exists
            if lane_id in self.spawn_points:
                spawn_x, spawn_y = self.spawn_points[lane_id]
                # Draw spawn point as a large circle
                cv2.circle(working_image, (spawn_x, spawn_y), 15, (0, 255, 0), -1)
                cv2.circle(working_image, (spawn_x, spawn_y), 20, (0, 255, 0), 3)
                # Draw label
                cv2.putText(working_image, f"Spawn {lane_id}", 
                          (spawn_x + 25, spawn_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Get actual window size (same as mark_lanes.py)
        try:
            win_rect = cv2.getWindowImageRect(self.window_name)
            if win_rect[2] > 0 and win_rect[3] > 0:
                self.win_width = win_rect[2]
                self.win_height = win_rect[3]
        except:
            pass  # Use default if can't get window size
        
        # Calculate scale to fit image in window initially (same as mark_lanes.py)
        scale_to_fit_x = self.win_width / self.width
        scale_to_fit_y = self.win_height / self.height
        scale_to_fit = min(scale_to_fit_x, scale_to_fit_y) * 0.95  # 95% to leave some margin
        
        # Apply zoom (relative to fit scale)
        effective_zoom = scale_to_fit * self.zoom_factor
        
        # Resize image based on effective zoom (same as mark_lanes.py)
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
        
        # Draw legend on display image
        legend_y = 30
        cv2.putText(self.display_image, f"Current Lane: {self.current_lane_id}", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(self.display_image, f"Zoom: {self.zoom_factor:.2f}x | Pan: ({int(self.pan_x)}, {int(self.pan_y)})", 
                   (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.display_image, f"Marked: {len(self.spawn_points)}/{len(self.lane_ids)} lanes", 
                   (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.display_image, "Green = Current lane, Blue = Other lanes", 
                   (10, legend_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _next_lane(self):
        """Move to next lane."""
        if not self.lane_ids:
            return
        
        if self.current_lane_id is None:
            self.current_lane_id = self.lane_ids[0]
        else:
            try:
                idx = self.lane_ids.index(self.current_lane_id)
                self.current_lane_id = self.lane_ids[(idx + 1) % len(self.lane_ids)]
            except ValueError:
                self.current_lane_id = self.lane_ids[0]
        
        print(f"Switched to lane {self.current_lane_id}")
        self._redraw()
    
    def _prev_lane(self):
        """Move to previous lane."""
        if not self.lane_ids:
            return
        
        if self.current_lane_id is None:
            self.current_lane_id = self.lane_ids[-1]
        else:
            try:
                idx = self.lane_ids.index(self.current_lane_id)
                self.current_lane_id = self.lane_ids[(idx - 1) % len(self.lane_ids)]
            except ValueError:
                self.current_lane_id = self.lane_ids[-1]
        
        print(f"Switched to lane {self.current_lane_id}")
        self._redraw()
    
    def _clear_current_spawn(self):
        """Clear spawn point for current lane."""
        if self.current_lane_id is not None and self.current_lane_id in self.spawn_points:
            del self.spawn_points[self.current_lane_id]
            print(f"Cleared spawn point for lane {self.current_lane_id}")
            self._redraw()
    
    def _save(self):
        """Save spawn points to JSON file."""
        # Update lanes_data with spawn points
        self.lanes_data['spawn_points'] = {
            str(lane_id): list(point) for lane_id, point in self.spawn_points.items()
        }
        
        # Save to file
        with open(self.lanes_json_path, 'w') as f:
            json.dump(self.lanes_data, f, indent=2)
        
        print(f"\n✅ Saved {len(self.spawn_points)} spawn points to {self.lanes_json_path}")
        print(f"   Lanes with spawn points: {sorted(self.spawn_points.keys())}")
    
    def run(self):
        """Run the interactive marking tool (same approach as mark_lanes.py)."""
        # Draw initial state (same as mark_lanes.py)
        self._redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._save()
            elif key == ord('n') or key == 83:  # 'n' or right arrow
                self._next_lane()
            elif key == ord('p') or key == 81:  # 'p' or left arrow
                self._prev_lane()
            elif key == ord('c'):
                self._clear_current_spawn()
            elif key == ord('+') or key == ord('='):
                self.zoom_factor = min(5.0, self.zoom_factor + 0.1)
                self._redraw()
            elif key == ord('-') or key == ord('_'):
                self.zoom_factor = max(0.2, self.zoom_factor - 0.1)
                self._redraw()
            elif key == ord('0'):
                self.zoom_factor = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self._redraw()
            elif key == ord('r'):
                self.zoom_factor = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self._redraw()
            elif key == ord('f'):
                # Toggle fullscreen
                self.is_fullscreen = not self.is_fullscreen
                if self.is_fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    # Get actual screen size for fullscreen
                    try:
                        import tkinter as tk
                        root = tk.Tk()
                        self.win_width = root.winfo_screenwidth()
                        self.win_height = root.winfo_screenheight()
                        root.destroy()
                    except:
                        pass
                    print("Entered fullscreen mode")
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    self.win_width = 1920
                    self.win_height = 1080
                    cv2.resizeWindow(self.window_name, self.win_width, self.win_height)
                    print("Exited fullscreen mode")
                # Wait a bit for window to resize, then redraw
                cv2.waitKey(100)
                self._redraw()
        
        cv2.destroyAllWindows()
        
        # Ask if user wants to save before quitting
        if self.spawn_points:
            print("\nYou have unsaved spawn points. Save before quitting? (y/n): ", end='')
            response = input().strip().lower()
            if response == 'y':
                self._save()


def main():
    parser = argparse.ArgumentParser(
        description="Mark spawn points for lanes in traffic simulation"
    )
    
    parser.add_argument(
        '--lanes',
        required=True,
        help='Path to lanes JSON file'
    )
    
    args = parser.parse_args()
    
    if not Path(args.lanes).exists():
        print(f"Error: Lanes file not found: {args.lanes}")
        return
    
    marker = SpawnPointMarker(args.lanes)
    marker.run()


if __name__ == "__main__":
    main()
