#!/usr/bin/env python3
"""
Interactive Lane Marking Tool

Click on the image to mark lane centerlines. The lanes will be saved and used
for the traffic simulation.

Usage:
    python mark_lanes.py --image eda/data/media/SiteA.jpg --output eda/data/lanes.json
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple


class LaneMarker:
    """Interactive tool for marking lanes on an image."""
    
    def __init__(self, image_path: str, output_path: str):
        """
        Initialize the lane marker.
        
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
        
        # Lane storage: list of lanes, each lane is a list of points
        self.lanes: List[List[Tuple[int, int]]] = []
        self.current_lane: List[Tuple[int, int]] = []
        self.current_lane_id = 0
        
        # Zoom and pan state
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        # Setup window
        self.window_name = "Lane Marker - Click to mark lanes, 'n' for new lane, 's' to save, 'q' to quit"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        # Initialize window size
        self.win_width = 1920
        self.win_height = 1080
        
        # Instructions
        self._print_instructions()
    
    def _print_instructions(self):
        """Print instructions for the user."""
        print("="*60)
        print("LANE MARKING TOOL")
        print("="*60)
        print("\nInstructions:")
        print("  1. Click LEFT MOUSE to add points to current lane")
        print("  2. RIGHT MOUSE + DRAG to pan the image")
        print("  3. MOUSE WHEEL or +/- keys to zoom in/out")
        print("  4. Press 'n' to start a NEW lane")
        print("  5. Press 's' to SAVE lanes to file")
        print("  6. Press 'q' to QUIT")
        print("  7. Press 'c' to CLEAR current lane")
        print("  8. Press 'u' to UNDO last point")
        print("  9. Press '+' or '-' to ZOOM in/out")
        print(" 10. Press '0' to RESET zoom to fit window")
        print(" 11. Press 'r' to RESET zoom/pan")
        print("\nLane Numbering:")
        print("  Lane 1 = Most bottom line (southernmost)")
        print("  Lane 2 = Next line above Lane 1")
        print("  Lane 3 = Next line above Lane 2")
        print("  ... and so on")
        print("\nMark lanes from BOTTOM to TOP in order.")
        print("="*60)
        print(f"\nImage size: {self.width}x{self.height}")
        print(f"Output will be saved to: {self.output_path}")
        print("\nStarting... (click on image to begin)\n")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert screen coordinates to image coordinates
            # Calculate effective zoom (accounting for fit-to-window)
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
            
            # Add point to current lane (use original image coordinates)
            if 0 <= img_x < self.width and 0 <= img_y < self.height:
                self.current_lane.append((img_x, img_y))
                print(f"  Added point ({img_x}, {img_y}) to lane {self.current_lane_id + 1}")
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
                # Update pan position
                self.pan_x += x - self.pan_start_x
                self.pan_y += y - self.pan_start_y
                self.pan_start_x = x
                self.pan_start_y = y
                self._redraw()
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in/out
            zoom_delta = 0.1
            if flags > 0:  # Scroll up - zoom in
                self.zoom_factor = min(5.0, self.zoom_factor + zoom_delta)
            else:  # Scroll down - zoom out
                self.zoom_factor = max(0.2, self.zoom_factor - zoom_delta)
            self._redraw()
    
    def _redraw(self):
        """Redraw the image with current lanes."""
        # Create working copy
        working_image = self.image.copy()
        
        # Draw all completed lanes
        for lane_id, lane_points in enumerate(self.lanes):
            if len(lane_points) >= 2:
                # Draw lane line
                points = np.array(lane_points, dtype=np.int32)
                cv2.polylines(working_image, [points], False, (0, 255, 0), 2)
                
                # Draw lane number at start
                if len(lane_points) > 0:
                    cv2.putText(working_image, f"L{lane_id + 1}", 
                              (lane_points[0][0] + 10, lane_points[0][1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current lane being marked
        if len(self.current_lane) >= 2:
            points = np.array(self.current_lane, dtype=np.int32)
            cv2.polylines(working_image, [points], False, (255, 0, 0), 2)
        
        # Draw all points
        for lane_id, lane_points in enumerate(self.lanes):
            for point in lane_points:
                cv2.circle(working_image, point, 5, (0, 255, 0), -1)
        
        for point in self.current_lane:
            cv2.circle(working_image, point, 5, (255, 0, 0), -1)
        
        # Draw lane number for current lane
        if len(self.current_lane) > 0:
            cv2.putText(working_image, f"L{self.current_lane_id + 1} (current)", 
                      (self.current_lane[0][0] + 10, self.current_lane[0][1] - 10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Get actual window size
        try:
            win_rect = cv2.getWindowImageRect(self.window_name)
            if win_rect[2] > 0 and win_rect[3] > 0:
                self.win_width = win_rect[2]
                self.win_height = win_rect[3]
        except:
            pass  # Use default if can't get window size
        
        # Calculate scale to fit image in window initially
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
        
        # Draw legend on display image
        legend_y = 30
        cv2.putText(self.display_image, f"Zoom: {self.zoom_factor:.2f}x | Pan: ({self.pan_x}, {self.pan_y})", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.display_image, "Green = Completed lanes, Red = Current lane", 
                   (10, legend_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(self.display_image, f"Lanes marked: {len(self.lanes)}", 
                   (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(self.window_name, self.display_image)
    
    def _finish_current_lane(self):
        """Finish the current lane and start a new one."""
        if len(self.current_lane) >= 2:
            self.lanes.append(self.current_lane.copy())
            print(f"✓ Completed lane {self.current_lane_id + 1} with {len(self.current_lane)} points")
            self.current_lane_id += 1
            self.current_lane = []
            self._redraw()
        else:
            print("⚠ Need at least 2 points to complete a lane")
    
    def _clear_current_lane(self):
        """Clear the current lane."""
        self.current_lane = []
        print("Cleared current lane")
        self._redraw()
    
    def _undo_last_point(self):
        """Remove the last point from current lane."""
        if self.current_lane:
            removed = self.current_lane.pop()
            print(f"Removed point {removed}")
            self._redraw()
        else:
            print("No points to undo")
    
    def _save_lanes(self):
        """Save lanes to JSON file."""
        if not self.lanes:
            print("⚠ No lanes to save!")
            return
        
        # Prepare data structure
        lane_data = {
            'image_path': str(self.image_path),
            'image_width': self.width,
            'image_height': self.height,
            'lanes': []
        }
        
        for lane_id, lane_points in enumerate(self.lanes):
            lane_data['lanes'].append({
                'lane_id': lane_id + 1,
                'lane_number': lane_id + 1,  # 1 = bottom, 2 = next above, etc.
                'points': lane_points,
                'num_points': len(lane_points)
            })
        
        # Save to file
        output_path_obj = Path(self.output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_path, 'w') as f:
            json.dump(lane_data, f, indent=2)
        
        print(f"\n✅ Saved {len(self.lanes)} lanes to {self.output_path}")
        print(f"   Lane 1 = Most bottom (southernmost)")
        print(f"   Lane {len(self.lanes)} = Most top (northernmost)")
    
    def run(self):
        """Run the interactive marking tool."""
        self._redraw()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Quit
                if self.current_lane:
                    print("\n⚠ Current lane not finished. Press 'n' to finish it first.")
                else:
                    break
            
            elif key == ord('n'):
                # New lane
                self._finish_current_lane()
            
            elif key == ord('s'):
                # Save
                if self.current_lane:
                    self._finish_current_lane()
                self._save_lanes()
            
            elif key == ord('c'):
                # Clear current lane
                self._clear_current_lane()
            
            elif key == ord('u'):
                # Undo last point
                self._undo_last_point()
            
            elif key == ord('+') or key == ord('='):
                # Zoom in
                self.zoom_factor = min(5.0, self.zoom_factor + 0.1)
                self._redraw()
            
            elif key == ord('-') or key == ord('_'):
                # Zoom out
                self.zoom_factor = max(0.2, self.zoom_factor - 0.1)
                self._redraw()
            
            elif key == ord('r'):
                # Reset zoom and pan
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
        
        # Auto-save on quit if there are lanes
        if self.lanes:
            if self.current_lane:
                self._finish_current_lane()
            self._save_lanes()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive lane marking tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mark lanes on SiteA.jpg
  python mark_lanes.py --image eda/data/media/SiteA.jpg --output eda/data/lanes.json
  
Instructions:
  1. Click LEFT MOUSE to add points along a lane centerline
  2. Press 'n' to finish current lane and start a new one
  3. Press 's' to save lanes
  4. Press 'q' to quit
  
Lane Numbering:
  Lane 1 = Most bottom line (southernmost)
  Lane 2 = Next line above Lane 1
  Lane 3 = Next line above Lane 2
  ... mark from BOTTOM to TOP
        """
    )
    
    parser.add_argument('--image', required=True, help='Path to background image')
    parser.add_argument('--output', required=True, help='Output JSON file for lane data')
    
    args = parser.parse_args()
    
    try:
        marker = LaneMarker(args.image, args.output)
        marker.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

