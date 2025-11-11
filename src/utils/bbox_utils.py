"""
Bounding Box Utilities

Functions to calculate bounding boxes from vehicle positions, angles, and dimensions.
These bounding boxes are compatible with the visualization format used in the EDA directory.
"""

import numpy as np
from typing import Tuple, List


def calculate_bounding_box_corners(
    center_x: float,
    center_y: float,
    length: float,
    width: float,
    angle: float
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    Calculate the 4 corners of a rotated bounding box.
    
    Args:
        center_x: X coordinate of the center of the bounding box
        center_y: Y coordinate of the center of the bounding box
        length: Length of the vehicle (along the direction of travel)
        width: Width of the vehicle (perpendicular to direction of travel)
        angle: Rotation angle in radians (counter-clockwise from positive x-axis)
    
    Returns:
        Tuple of (x1, y1, x2, y2, x3, y3, x4, y4) representing the 4 corners
        The corners are ordered counter-clockwise starting from the front-left corner
    """
    # Half dimensions
    half_length = length / 2.0
    half_width = width / 2.0
    
    # Define corners relative to center (before rotation)
    # Front-left, front-right, back-right, back-left
    corners_relative = np.array([
        [-half_length, -half_width],  # Front-left
        [-half_length,  half_width],   # Front-right
        [ half_length,  half_width],   # Back-right
        [ half_length, -half_width]    # Back-left
    ])
    
    # Rotation matrix
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rotation_matrix = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    # Rotate corners
    corners_rotated = corners_relative @ rotation_matrix.T
    
    # Translate to absolute position
    corners_absolute = corners_rotated + np.array([center_x, center_y])
    
    # Return as tuple (x1, y1, x2, y2, x3, y3, x4, y4)
    # Note: The visualization expects corners in this order, but we'll ensure
    # they form a proper rectangle
    return (
        float(corners_absolute[0, 0]), float(corners_absolute[0, 1]),  # x1, y1
        float(corners_absolute[1, 0]), float(corners_absolute[1, 1]),  # x2, y2
        float(corners_absolute[2, 0]), float(corners_absolute[2, 1]),  # x3, y3
        float(corners_absolute[3, 0]), float(corners_absolute[3, 1])   # x4, y4
    )


def world_to_image_coordinates(
    world_x: float,
    world_y: float,
    image_width: int,
    image_height: int,
    world_bounds: Tuple[float, float, float, float],
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    scale: float = 1.0
) -> Tuple[float, float]:
    """
    Transform world coordinates to image pixel coordinates.
    
    This uses a direct transformation where:
    - World origin (0, 0) maps to (offset_x, offset_y) in image space
    - Scale is pixels per meter
    - Y-axis is flipped (world Y increases upward, image Y increases downward)
    
    Args:
        world_x: World x coordinate (meters)
        world_y: World y coordinate (meters)
        image_width: Width of the target image in pixels (for bounds checking)
        image_height: Height of the target image in pixels (for bounds checking)
        world_bounds: Tuple of (min_x, min_y, max_x, max_y) - not used but kept for compatibility
        offset_x: X coordinate in image space where world origin (0,0) maps to
        offset_y: Y coordinate in image space where world origin (0,0) maps to
        scale: Scale factor in pixels per meter
    
    Returns:
        Tuple of (image_x, image_y) in pixel coordinates
    """
    # Direct transformation: world origin maps to (offset_x, offset_y)
    # Scale is pixels per meter
    # Y-axis is flipped (world Y+ = image Y-)
    image_x = offset_x + world_x * scale
    image_y = offset_y - world_y * scale  # Flip Y-axis
    
    return image_x, image_y


def get_world_bounds_from_road_network(road_network) -> Tuple[float, float, float, float]:
    """
    Calculate world bounds from a road network.
    
    Args:
        road_network: RoadNetwork object
    
    Returns:
        Tuple of (min_x, min_y, max_x, max_y)
    """
    if not road_network.all_lanes:
        return (-1000, -1000, 1000, 1000)
    
    all_x = []
    all_y = []
    
    for lane in road_network.all_lanes.values():
        all_x.append(lane.start_point.x)
        all_x.append(lane.end_point.x)
        all_y.append(lane.start_point.y)
        all_y.append(lane.end_point.y)
    
    min_x = min(all_x)
    max_x = max(all_x)
    min_y = min(all_y)
    max_y = max(all_y)
    
    # Add padding
    padding = 100.0
    return (min_x - padding, min_y - padding, max_x + padding, max_y + padding)

