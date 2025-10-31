#!/usr/bin/env python3
"""
Calculate optimal scale to fit simulation within image bounds
"""

def calculate_optimal_scale():
    """Calculate scale that keeps all vehicles within image bounds."""
    
    # Image dimensions
    image_width = 3840
    image_height = 2160
    
    # ROI center (where world origin maps to)
    offset_x = 2064
    offset_y = 526
    
    # Simulation world bounds
    world_min_x = -1500
    world_max_x = 1500
    world_min_y = -1500
    world_max_y = 1500
    
    # Calculate available space around ROI center
    space_left = offset_x  # Pixels available to the left
    space_right = image_width - offset_x  # Pixels available to the right
    space_up = offset_y  # Pixels available above
    space_down = image_height - offset_y  # Pixels available below
    
    print("="*60)
    print("Optimal Scale Calculation")
    print("="*60)
    print(f"\nImage: {image_width}x{image_height}")
    print(f"ROI center: ({offset_x}, {offset_y})")
    print(f"\nAvailable space around center:")
    print(f"  Left: {space_left} pixels")
    print(f"  Right: {space_right} pixels")
    print(f"  Up: {space_up} pixels")
    print(f"  Down: {space_down} pixels")
    
    print(f"\nWorld bounds:")
    print(f"  X: {world_min_x} to {world_max_x} meters (range: {world_max_x - world_min_x} m)")
    print(f"  Y: {world_min_y} to {world_max_y} meters (range: {world_max_y - world_min_y} m)")
    
    # Calculate scale for each direction
    scale_x_positive = space_right / world_max_x  # For positive X (east)
    scale_x_negative = space_left / abs(world_min_x)  # For negative X (west)
    scale_y_positive = space_up / world_max_y  # For positive Y (north) - note: Y flips
    scale_y_negative = space_down / abs(world_min_y)  # For negative Y (south)
    
    print(f"\nScale calculations:")
    print(f"  X positive (east): {space_right} / {world_max_x} = {scale_x_positive:.4f} px/m")
    print(f"  X negative (west): {space_left} / {abs(world_min_x)} = {scale_x_negative:.4f} px/m")
    print(f"  Y positive (north): {space_up} / {world_max_y} = {scale_y_positive:.4f} px/m")
    print(f"  Y negative (south): {space_down} / {abs(world_min_y)} = {scale_y_negative:.4f} px/m")
    
    # Use the minimum to ensure everything fits
    optimal_scale = min(scale_x_positive, scale_x_negative, scale_y_positive, scale_y_negative)
    
    print(f"\nOptimal scale: {optimal_scale:.4f} pixels per meter")
    print(f"  (using minimum to ensure all directions fit)")
    
    # Test with optimal scale
    print(f"\nTest with optimal scale ({optimal_scale:.4f} px/m):")
    test_points = [
        (world_max_x, 0, "East edge"),
        (world_min_x, 0, "West edge"),
        (0, world_max_y, "North edge"),
        (0, world_min_y, "South edge"),
        (world_max_x, world_max_y, "Northeast corner"),
        (world_min_x, world_min_y, "Southwest corner"),
    ]
    
    for wx, wy, desc in test_points:
        img_x = offset_x + wx * optimal_scale
        img_y = offset_y - wy * optimal_scale  # Flip Y
        in_bounds = (0 <= img_x <= image_width and 0 <= img_y <= image_height)
        print(f"  {desc}: world({wx:.0f}, {wy:.0f}) -> image({img_x:.1f}, {img_y:.1f}) {'✓' if in_bounds else '✗'}")
    
    return optimal_scale


if __name__ == "__main__":
    scale = calculate_optimal_scale()
    print(f"\n" + "="*60)
    print(f"RECOMMENDED SCALE: {scale:.4f} pixels per meter")
    print("="*60)

