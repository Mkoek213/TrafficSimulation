#!/usr/bin/env python3
"""
Interactive Centerline Marking Tool

Click on the image to mark lane centerlines. Vehicles will follow these centerlines exactly.
You can also specify allowed directions (straight, left, right) for each lane.

Usage:
    python mark_centerlines.py --lanes eda/data/lanes.json
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional


class CenterlineMarker:
    """Interactive tool for marking lane centerlines on an image."""
    
    def __init__(self, lanes_json_path: str):
        """
        Initialize the centerline marker.
        
        Args:
            lanes_json_path: Path to existing lanes.json file
        """
        self.lanes_json_path = Path(lanes_json_path)
        
        # Load existing lanes data
        if self.lanes_json_path.exists():
            with open(self.lanes_json_path) as f:
                self.lanes_data = json.load(f)
        else:
            raise ValueError(f"Lanes file not found: {lanes_json_path}")
        
        # Get image path from lanes data
        image_path = self.lanes_data.get('image_path', '')
        if not image_path:
            raise ValueError("No image_path found in lanes.json")
        
        # Resolve image path (check project root first, then relative to lanes.json)
        project_root = Path.cwd()
        lanes_dir = self.lanes_json_path.parent
        
        if (project_root / image_path).exists():
            image_path = str(project_root / image_path)
        elif (lanes_dir / image_path).exists():
            image_path = str(lanes_dir / image_path)
        else:
            raise ValueError(f"Could not find image: {image_path}")
        
        # Load image
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.display_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
        # Load existing lanes
        self.lanes = self.lanes_data.get('lanes', [])
        
        # Current editing state
        self.current_lane_id: Optional[int] = None
        self.current_centerline: List[Tuple[int, int]] = []
        
        # Zoom and pan state
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Setup window (same as mark_spawn_points.py)
        self.window_name = "Centerline Marker - Click to mark centerlines, 's' to save, 'q' to quit"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Initialize window size (same as mark_spawn_points.py)
        self.win_width = 1920
        self.win_height = 1080
        
        # Zoom starts at 1.0 (same as mark_spawn_points.py)
        self.zoom_factor = 1.0
        
        # Instructions
        self._print_instructions()
    
    def _print_instructions(self):
        """Print instructions for the user."""
        print("="*60)
        print("CENTERLINE MARKING TOOL")
        print("="*60)
        print("\nInstructions:")
        print("  1. Type lane ID (1-9) and press ENTER to select a lane")
        print("  2. Click LEFT MOUSE to add centerline points (draw the line)")
        print("  3. RIGHT MOUSE + DRAG to pan the image")
        print("  4. MOUSE WHEEL or +/- keys to zoom in/out")
        print("  5. Press 's' to FINISH current lane centerline")
        print("  6. Press 'x' to TOGGLE spawn for current lane (cars spawn at first point)")
        print("  7. Press 'S' (capital) to SAVE all lanes to file")
        print("  8. Press 'q' to QUIT")
        print("  9. Press 'c' to CLEAR current centerline")
        print(" 10. Press 'u' to UNDO last point")
        print(" 11. Press '0' or 'r' to RESET zoom")
        print("\nSpawn Points:")
        print("  Mark lanes with 'x' where cars should spawn.")
        print("  Cars will spawn at the FIRST point of spawn-enabled centerlines.")
        print("  Green/Blue lanes = existing centerlines, Red = currently editing")
        print("  Yellow star = spawn point (first point of spawn-enabled lane)")
        print("="*60)
        print(f"\nImage size: {self.width}x{self.height}")
        print(f"Output will be saved to: {self.lanes_json_path}")
        print("\nType a lane ID (1-9) to begin marking its centerline...\n")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events (same approach as mark_spawn_points.py)."""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_lane_id is None:
                print("⚠ Please select a lane ID first (type lane number)")
                return
            
            # Convert screen coordinates to image coordinates (same as mark_spawn_points.py)
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
            
            self.current_centerline.append((img_x, img_y))
            print(f"  Added point {len(self.current_centerline)}: ({img_x}, {img_y})")
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
                # Pan the image (same as mark_spawn_points.py)
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
            # Zoom (same as mark_spawn_points.py)
            zoom_delta = 0.1
            if flags > 0:  # Scroll up - zoom in
                self.zoom_factor = min(5.0, self.zoom_factor + zoom_delta)
            else:  # Scroll down - zoom out
                self.zoom_factor = max(0.2, self.zoom_factor - zoom_delta)
            self._redraw()
    
    def _redraw(self):
        """Redraw the display image with centerlines (same approach as mark_spawn_points.py)."""
        if self.image is None:
            return
        
        # Create working copy
        working_image = self.image.copy()
        
        # Draw existing centerlines
        for lane_info in self.lanes:
            lane_id = lane_info.get('lane_id')
            centerline = lane_info.get('centerline_points', [])
            spawn_enabled = lane_info.get('spawn_enabled', False)
            
            if centerline:
                pts = np.array(centerline, dtype=np.int32)
                # Highlight current lane being edited
                color = (0, 255, 0) if lane_id == self.current_lane_id else (100, 100, 255)
                thickness = 3 if lane_id == self.current_lane_id else 2
                cv2.polylines(working_image, [pts], False, color, thickness)
                
                # Draw first and last points
                if len(centerline) > 0:
                    cv2.circle(working_image, tuple(centerline[0]), 5, color, -1)
                    cv2.circle(working_image, tuple(centerline[-1]), 5, color, -1)
                    
                    # Draw spawn marker (yellow star) at first point if spawn enabled
                    if spawn_enabled:
                        first_pt = centerline[0]
                        # Draw a yellow star/circle at spawn point
                        cv2.circle(working_image, first_pt, 12, (0, 255, 255), 3)
                        cv2.circle(working_image, first_pt, 8, (0, 255, 255), -1)
                        cv2.putText(working_image, "SPAWN", 
                                  (first_pt[0] + 15, first_pt[1]), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Draw lane ID
                if len(centerline) > 0:
                    mid_idx = len(centerline) // 2
                    label = f"L{lane_id}"
                    if spawn_enabled:
                        label += " [SPAWN]"
                    cv2.putText(working_image, label, 
                              tuple(centerline[mid_idx]), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, color, 2)
        
        # Draw current centerline being edited
        if self.current_centerline:
            pts = np.array(self.current_centerline, dtype=np.int32)
            cv2.polylines(working_image, [pts], False, (255, 0, 0), 3)
            for pt in self.current_centerline:
                cv2.circle(working_image, pt, 5, (255, 0, 0), -1)
        
        # Get actual window size (same as mark_spawn_points.py)
        try:
            win_rect = cv2.getWindowImageRect(self.window_name)
            if win_rect[2] > 0 and win_rect[3] > 0:
                self.win_width = win_rect[2]
                self.win_height = win_rect[3]
        except:
            pass  # Use default if can't get window size
        
        # Calculate scale to fit image in window initially (same as mark_spawn_points.py)
        scale_to_fit_x = self.win_width / self.width
        scale_to_fit_y = self.win_height / self.height
        scale_to_fit = min(scale_to_fit_x, scale_to_fit_y) * 0.95  # 95% to leave some margin
        
        # Apply zoom (relative to fit scale)
        effective_zoom = scale_to_fit * self.zoom_factor
        
        # Resize image based on effective zoom (same as mark_spawn_points.py)
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
        if self.current_lane_id:
            # Check if current lane has spawn enabled
            spawn_status = ""
            for lane_info in self.lanes:
                if lane_info.get('lane_id') == self.current_lane_id:
                    if lane_info.get('spawn_enabled', False):
                        spawn_status = " [SPAWN ENABLED]"
                    break
            cv2.putText(self.display_image, f"Current Lane: {self.current_lane_id}{spawn_status}", 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(self.display_image, f"Points: {len(self.current_centerline)} | Zoom: {self.zoom_factor:.2f}x | Pan: ({int(self.pan_x)}, {int(self.pan_y)})", 
                   (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.display_image, "Green = Current lane, Blue = Other lanes, Red = Editing, Yellow = Spawn point", 
                   (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _finish_current_lane(self):
        """Finish marking current lane and save to lanes data."""
        if self.current_lane_id is None or not self.current_centerline:
            print("⚠ No active lane or centerline to finish")
            return
        
        # Find lane in lanes data
        lane_found = False
        for lane_info in self.lanes:
            if lane_info.get('lane_id') == self.current_lane_id:
                lane_info['centerline_points'] = self.current_centerline
                # Preserve spawn_enabled status if it exists
                if 'spawn_enabled' not in lane_info:
                    lane_info['spawn_enabled'] = False
                lane_found = True
                spawn_status = " [SPAWN ENABLED]" if lane_info.get('spawn_enabled', False) else ""
                print(f"✓ Saved centerline for lane {self.current_lane_id} ({len(self.current_centerline)} points){spawn_status}")
                if lane_info.get('spawn_enabled', False):
                    first_pt = self.current_centerline[0]
                    print(f"  Cars will spawn at first point: ({first_pt[0]}, {first_pt[1]})")
                break
        
        if not lane_found:
            print(f"⚠ Lane {self.current_lane_id} not found in lanes data")
        
        # Reset current editing
        self.current_lane_id = None
        self.current_centerline = []
        self._redraw()
    
    def _toggle_spawn(self):
        """Toggle spawn enabled/disabled for the current lane."""
        if self.current_lane_id is None:
            print("⚠ Please select a lane ID first")
            return
        
        # Find lane and toggle spawn
        for lane_info in self.lanes:
            if lane_info.get('lane_id') == self.current_lane_id:
                current_spawn = lane_info.get('spawn_enabled', False)
                lane_info['spawn_enabled'] = not current_spawn
                status = "ENABLED" if not current_spawn else "DISABLED"
                print(f"✓ Spawn {status} for lane {self.current_lane_id}")
                if not current_spawn and lane_info.get('centerline_points'):
                    first_pt = lane_info['centerline_points'][0]
                    print(f"  Cars will spawn at first point: ({first_pt[0]}, {first_pt[1]})")
                self._redraw()
                return
        
        print(f"⚠ Lane {self.current_lane_id} not found")
    
    def _save_all(self):
        """Save all lanes data to file."""
        # Save current centerline if any
        if self.current_lane_id is not None and self.current_centerline:
            self._finish_current_lane()
        
        # Backup original file
        backup_path = self.lanes_json_path.with_suffix('.json.bak')
        import shutil
        shutil.copy(self.lanes_json_path, backup_path)
        print(f"✓ Created backup: {backup_path}")
        
        # Save updated data
        with open(self.lanes_json_path, 'w') as f:
            json.dump(self.lanes_data, f, indent=2)
        
        print(f"✓ Saved all lanes to {self.lanes_json_path}")
    
    def run(self):
        """Run the interactive marking tool."""
        self._redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._finish_current_lane()
            elif key == ord('S'):
                self._save_all()
            elif key == ord('c'):
                self.current_centerline = []
                print("  Cleared current centerline")
                self._redraw()
            elif key == ord('u'):
                if self.current_centerline:
                    removed = self.current_centerline.pop()
                    print(f"  Removed last point: {removed}")
                    self._redraw()
            elif key == ord('x'):
                self._toggle_spawn()
            elif key == ord('+') or key == ord('='):
                # Zoom in (same as mark_spawn_points.py)
                zoom_delta = 0.1
                self.zoom_factor = min(5.0, self.zoom_factor + zoom_delta)
                self._redraw()
            elif key == ord('-') or key == ord('_'):
                # Zoom out (same as mark_spawn_points.py)
                zoom_delta = 0.1
                self.zoom_factor = max(0.2, self.zoom_factor - zoom_delta)
                self._redraw()
            elif key == ord('0') or key == ord('r'):
                # Reset zoom and pan (same as mark_spawn_points.py)
                self.zoom_factor = 1.0
                self.pan_x = 0
                self.pan_y = 0
                self._redraw()
            elif key == ord('\n') or key == ord('\r'):
                # Enter key - prompt for lane ID
                try:
                    lane_id_input = input("\nEnter lane ID (1-9) or 'q' to quit: ").strip()
                    if lane_id_input.lower() == 'q':
                        break
                    lane_id = int(lane_id_input)
                    if 1 <= lane_id <= 9:
                        self.current_lane_id = lane_id
                        # Load existing centerline if any
                        for lane_info in self.lanes:
                            if lane_info.get('lane_id') == lane_id:
                                self.current_centerline = lane_info.get('centerline_points', [])
                                spawn_status = " [SPAWN ENABLED]" if lane_info.get('spawn_enabled', False) else ""
                                if self.current_centerline:
                                    print(f"  Loaded existing centerline for lane {lane_id} ({len(self.current_centerline)} points){spawn_status}")
                                else:
                                    print(f"  Starting new centerline for lane {lane_id}{spawn_status}")
                                break
                        else:
                            self.current_centerline = []
                            print(f"  Starting new centerline for lane {lane_id}")
                        self._redraw()
                    else:
                        print("⚠ Lane ID must be between 1 and 9")
                except ValueError:
                    print("⚠ Invalid lane ID")
        
        cv2.destroyAllWindows()
        print("\nFinished marking centerlines.")


def main():
    parser = argparse.ArgumentParser(description='Mark lane centerlines')
    parser.add_argument('--lanes', type=str, required=True,
                       help='Path to lanes.json file')
    
    args = parser.parse_args()
    
    marker = CenterlineMarker(args.lanes)
    marker.run()


if __name__ == '__main__':
    main()

