#!/usr/bin/env python3
"""
Analyze ROI coordinates and create coordinate transformation
"""

import json
from pathlib import Path
from PIL import Image


def analyze_roi_and_image():
    """Analyze ROI coordinates and image to understand coordinate system."""
    
    # Load ROI
    roi_path = Path("eda/data/roi.json")
    with open(roi_path) as f:
        roi_data = json.load(f)
    
    # Load image
    image_path = Path("eda/data/media/SiteA.jpg")
    img = Image.open(image_path)
    image_width, image_height = img.size
    
    print("="*60)
    print("ROI and Image Analysis")
    print("="*60)
    print(f"\nImage size: {image_width}x{image_height}")
    
    # Extract ROI coordinates
    site_a_roi = roi_data.get("Site A", {}).get("A", {})
    
    all_x = []
    all_y = []
    
    print("\nROI regions:")
    for region_id, points in site_a_roi.items():
        print(f"  Region {region_id}: {len(points)} points")
        for point in points:
            all_x.append(point[0])
            all_y.append(point[1])
    
    print(f"\nROI coordinate ranges:")
    print(f"  X: {min(all_x):.0f} to {max(all_x):.0f} (range: {max(all_x) - min(all_x):.0f} pixels)")
    print(f"  Y: {min(all_y):.0f} to {max(all_y):.0f} (range: {max(all_y) - min(all_y):.0f} pixels)")
    
    # Calculate ROI center and bounds
    roi_center_x = (min(all_x) + max(all_x)) / 2
    roi_center_y = (min(all_y) + max(all_y)) / 2
    roi_width = max(all_x) - min(all_x)
    roi_height = max(all_y) - min(all_y)
    
    print(f"\nROI bounds:")
    print(f"  Center: ({roi_center_x:.0f}, {roi_center_y:.0f})")
    print(f"  Width: {roi_width:.0f} pixels")
    print(f"  Height: {roi_height:.0f} pixels")
    
    # Simulation world bounds (from road network)
    # The road network extends from -1500 to +1500 meters in each direction
    sim_min_x = -1500
    sim_max_x = 1500
    sim_min_y = -1500
    sim_max_y = 1500
    sim_width = sim_max_x - sim_min_x
    sim_height = sim_max_y - sim_min_y
    
    print(f"\nSimulation world bounds:")
    print(f"  X: {sim_min_x} to {sim_max_x} meters (range: {sim_width} meters)")
    print(f"  Y: {sim_min_y} to {sim_max_y} meters (range: {sim_height} meters)")
    
    # Calculate transformation
    # Scale factor: pixels per meter
    scale_x = roi_width / sim_width
    scale_y = roi_height / sim_height
    scale = min(scale_x, scale_y)  # Use smaller scale to maintain aspect ratio
    
    print(f"\nCoordinate transformation:")
    print(f"  Scale (pixels/meter): {scale:.4f}")
    print(f"  ROI center offset: ({roi_center_x:.0f}, {roi_center_y:.0f})")
    
    # Transformation function
    def world_to_image(world_x, world_y):
        """Transform world coordinates to image coordinates."""
        # Map world coordinates to ROI region
        # World origin (0,0) maps to ROI center
        image_x = roi_center_x + world_x * scale
        image_y = roi_center_y - world_y * scale  # Flip Y axis (world Y increases upward, image Y increases downward)
        return image_x, image_y
    
    # Test transformation
    print(f"\nTest transformations:")
    test_points = [
        (0, 0, "World origin"),
        (500, 0, "East 500m"),
        (-500, 0, "West 500m"),
        (0, 500, "North 500m"),
        (0, -500, "South 500m"),
    ]
    
    for wx, wy, desc in test_points:
        ix, iy = world_to_image(wx, wy)
        print(f"  {desc}: world({wx:.0f}, {wy:.0f}) -> image({ix:.0f}, {iy:.0f})")
    
    return {
        'image_width': image_width,
        'image_height': image_height,
        'roi_center_x': roi_center_x,
        'roi_center_y': roi_center_y,
        'roi_width': roi_width,
        'roi_height': roi_height,
        'scale': scale,
        'world_to_image': world_to_image
    }


def estimate_bbox_size_from_roi():
    """Estimate bounding box size based on ROI and typical car dimensions."""
    
    # Typical car dimensions in meters
    car_length_m = 4.5  # meters
    car_width_m = 2.0   # meters
    
    # Analyze ROI to estimate scale
    roi_path = Path("eda/data/roi.json")
    with open(roi_path) as f:
        roi_data = json.load(f)
    
    site_a_roi = roi_data.get("Site A", {}).get("A", {})
    
    # Get region 1 points to estimate scale
    region1_points = site_a_roi.get("1", [])
    if region1_points:
        # Calculate approximate lane width from ROI
        # Region 1 seems to be a lane, estimate its width
        x_coords = [p[0] for p in region1_points]
        y_coords = [p[1] for p in region1_points]
        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)
        
        # Estimate lane width in pixels (rough approximation)
        lane_width_pixels = min(x_range, y_range) / 2  # Rough estimate
        
        # Typical lane width is 3.5 meters
        # So pixels per meter = lane_width_pixels / 3.5
        pixels_per_meter = lane_width_pixels / 3.5
        
        car_length_pixels = car_length_m * pixels_per_meter
        car_width_pixels = car_width_m * pixels_per_meter
        
        print(f"\nEstimated bounding box sizes:")
        print(f"  Lane width (estimated): {lane_width_pixels:.1f} pixels")
        print(f"  Pixels per meter: {pixels_per_meter:.2f}")
        print(f"  Car length: {car_length_pixels:.1f} pixels")
        print(f"  Car width: {car_width_pixels:.1f} pixels")
        
        return car_length_pixels, car_width_pixels
    
    return None, None


if __name__ == "__main__":
    results = analyze_roi_and_image()
    bbox_length, bbox_width = estimate_bbox_size_from_roi()
    
    print("\n" + "="*60)
    print("RECOMMENDED TRANSFORMATION PARAMETERS")
    print("="*60)
    print(f"offset_x: {results['roi_center_x']:.0f}")
    print(f"offset_y: {results['roi_center_y']:.0f}")
    print(f"scale: {results['scale']:.6f}")
    print(f"image_width: {results['image_width']}")
    print(f"image_height: {results['image_height']}")
    if bbox_length and bbox_width:
        print(f"\nbbox_length_pixels: {bbox_length:.1f}")
        print(f"bbox_width_pixels: {bbox_width:.1f}")
    print("="*60)

