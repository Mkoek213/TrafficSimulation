#!/usr/bin/env python3
"""
Analyze real DRIFT dataset to understand coordinate system and bounding box sizes
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_real_data(csv_path: str, image_path: str):
    """Analyze real data to understand coordinate system and bounding box sizes."""
    
    print("="*60)
    print("Analyzing Real DRIFT Dataset")
    print("="*60)
    
    # Load CSV
    print(f"\n1. Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Load image
    print(f"\n2. Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"   ERROR: Could not load image")
        return None
    
    image_height, image_width = image.shape[:2]
    print(f"   Image size: {image_width}x{image_height}")
    
    # Analyze coordinate ranges
    print("\n3. Analyzing coordinate ranges:")
    print(f"   center_x: min={df['center_x'].min():.2f}, max={df['center_x'].max():.2f}, mean={df['center_x'].mean():.2f}")
    print(f"   center_y: min={df['center_y'].min():.2f}, max={df['center_y'].max():.2f}, mean={df['center_y'].mean():.2f}")
    
    # Analyze bounding box sizes
    print("\n4. Analyzing bounding box sizes:")
    
    # Calculate bounding box width and height from corners
    # Assuming corners are in order: front-left, front-right, back-right, back-left
    def calculate_bbox_size(row):
        corners = np.array([
            [row['x1'], row['y1']],
            [row['x2'], row['y2']],
            [row['x3'], row['y3']],
            [row['x4'], row['y4']]
        ])
        
        # Calculate width (average of front and back)
        front_width = np.linalg.norm(corners[1] - corners[0])
        back_width = np.linalg.norm(corners[2] - corners[3])
        width = (front_width + back_width) / 2
        
        # Calculate length (average of left and right)
        left_length = np.linalg.norm(corners[3] - corners[0])
        right_length = np.linalg.norm(corners[2] - corners[1])
        length = (left_length + right_length) / 2
        
        return width, length
    
    widths = []
    lengths = []
    for _, row in df.head(100).iterrows():  # Sample first 100 rows
        w, l = calculate_bbox_size(row)
        widths.append(w)
        lengths.append(l)
    
    widths = np.array(widths)
    lengths = np.array(lengths)
    
    print(f"   Bounding box width (pixels): min={widths.min():.2f}, max={widths.max():.2f}, mean={widths.mean():.2f}")
    print(f"   Bounding box length (pixels): min={lengths.min():.2f}, max={lengths.max():.2f}, mean={lengths.mean():.2f}")
    
    # Analyze frame data
    print("\n5. Analyzing frame data:")
    print(f"   Frames: {df['frame'].min()} to {df['frame'].max()}")
    print(f"   Unique tracks: {df['track_id'].nunique()}")
    print(f"   Average vehicles per frame: {len(df) / df['frame'].nunique():.2f}")
    
    # Sample a frame to see actual positions
    sample_frame = df['frame'].iloc[0]
    frame_data = df[df['frame'] == sample_frame]
    print(f"\n6. Sample frame {sample_frame}:")
    print(f"   Vehicles: {len(frame_data)}")
    print(f"   Position ranges:")
    print(f"     center_x: {frame_data['center_x'].min():.2f} to {frame_data['center_x'].max():.2f}")
    print(f"     center_y: {frame_data['center_y'].min():.2f} to {frame_data['center_y'].max():.2f}")
    
    # Check if coordinates are within image bounds
    print("\n7. Coordinate validation:")
    in_bounds_x = ((df['center_x'] >= 0) & (df['center_x'] <= image_width)).sum()
    in_bounds_y = ((df['center_y'] >= 0) & (df['center_y'] <= image_height)).sum()
    print(f"   center_x in bounds [0, {image_width}]: {in_bounds_x}/{len(df)} ({100*in_bounds_x/len(df):.1f}%)")
    print(f"   center_y in bounds [0, {image_height}]: {in_bounds_y}/{len(df)} ({100*in_bounds_y/len(df):.1f}%)")
    
    return {
        'image_width': image_width,
        'image_height': image_height,
        'center_x_range': (df['center_x'].min(), df['center_x'].max()),
        'center_y_range': (df['center_y'].min(), df['center_y'].max()),
        'bbox_width_mean': widths.mean(),
        'bbox_length_mean': lengths.mean(),
        'bbox_width_range': (widths.min(), widths.max()),
        'bbox_length_range': (lengths.min(), lengths.max())
    }


if __name__ == "__main__":
    # Check if data exists
    csv_path = project_root / "eda/data/drift/site_A/drone_1.csv"
    image_path = project_root / "eda/data/media/SiteA.jpg"
    
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}")
        print("Please download data first:")
        print("  python eda/scripts/download_data.py --site A --files 1")
        sys.exit(1)
    
    if not image_path.exists():
        print(f"Image file not found: {image_path}")
        sys.exit(1)
    
    results = analyze_real_data(str(csv_path), str(image_path))
    
    if results:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Image dimensions: {results['image_width']}x{results['image_height']}")
        print(f"Coordinate ranges:")
        print(f"  X: {results['center_x_range'][0]:.2f} to {results['center_x_range'][1]:.2f}")
        print(f"  Y: {results['center_y_range'][0]:.2f} to {results['center_y_range'][1]:.2f}")
        print(f"Bounding box sizes (pixels):")
        print(f"  Width: {results['bbox_width_mean']:.2f} (range: {results['bbox_width_range'][0]:.2f} - {results['bbox_width_range'][1]:.2f})")
        print(f"  Length: {results['bbox_length_mean']:.2f} (range: {results['bbox_length_range'][0]:.2f} - {results['bbox_length_range'][1]:.2f})")
        print("="*60)

