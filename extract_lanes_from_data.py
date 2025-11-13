"""
Extract smooth lane centerlines from real car trajectories in drone data.

This script:
1. Reads the drone CSV data
2. Identifies vehicles that drive long, smooth paths
3. Extracts their trajectories as lane centerlines
4. Saves to JSON format for the simulation
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

# Configuration
DATA_PATH = "eda/data/drift/site_A/drone_1.csv"
OUTPUT_JSON = "data/custom_lanes.json"
IMAGE_WIDTH = 3840
IMAGE_HEIGHT = 2160

# ROI parameters (from roi.json)
OFFSET_X = 2064.0  # ROI center X
OFFSET_Y = 526.0   # ROI center Y
SCALE = 0.3507     # pixels per meter

# Parameters for lane extraction
MIN_TRAJECTORY_LENGTH = 100  # Minimum number of points
MIN_DISTANCE_TRAVELED = 150  # Minimum distance in meters
SMOOTHING_WINDOW = 15  # Savitzky-Golay filter window
POLYORDER = 3  # Polynomial order for smoothing

def load_data():
    """Load drone data."""
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} detections for {df['track_id'].nunique()} vehicles")
    return df

def calculate_trajectory_metrics(trajectory_df):
    """Calculate metrics for a trajectory."""
    # Sort by frame
    traj = trajectory_df.sort_values('frame')
    
    # Calculate total distance traveled
    centers = traj[['center_x', 'center_y']].values
    distances = np.sqrt(np.sum(np.diff(centers, axis=0)**2, axis=1))
    total_distance = np.sum(distances) / SCALE  # Convert to meters
    
    # Calculate smoothness (variance in direction changes)
    if len(centers) > 2:
        directions = np.diff(centers, axis=0)
        angles = np.arctan2(directions[:, 1], directions[:, 0])
        angle_changes = np.abs(np.diff(angles))
        # Wrap around 2*pi
        angle_changes = np.minimum(angle_changes, 2*np.pi - angle_changes)
        smoothness = np.std(angle_changes)
    else:
        smoothness = 999
    
    return {
        'length': len(traj),
        'distance_meters': total_distance,
        'smoothness': smoothness
    }

def smooth_trajectory(points, smoothing_window=SMOOTHING_WINDOW):
    """Smooth trajectory using Savitzky-Golay filter."""
    if len(points) < smoothing_window:
        return points
    
    # Apply Savitzky-Golay filter
    x_smooth = savgol_filter(points[:, 0], smoothing_window, POLYORDER)
    y_smooth = savgol_filter(points[:, 1], smoothing_window, POLYORDER)
    
    return np.column_stack([x_smooth, y_smooth])

def resample_trajectory(points, num_points=50):
    """Resample trajectory to have evenly spaced points."""
    # Calculate cumulative distance along trajectory
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate([[0], np.cumsum(distances)])
    
    # Create evenly spaced distances
    total_dist = cumulative_dist[-1]
    even_distances = np.linspace(0, total_dist, num_points)
    
    # Interpolate x and y at even distances
    x_interp = np.interp(even_distances, cumulative_dist, points[:, 0])
    y_interp = np.interp(even_distances, cumulative_dist, points[:, 1])
    
    return np.column_stack([x_interp, y_interp])

def find_best_trajectories(df, num_lanes=9):
    """Find the best trajectories to use as lane centerlines."""
    print(f"\nAnalyzing trajectories...")
    
    # Group by track_id
    vehicle_metrics = []
    for track_id, group in df.groupby('track_id'):
        metrics = calculate_trajectory_metrics(group)
        metrics['track_id'] = track_id
        vehicle_metrics.append(metrics)
    
    # Convert to DataFrame for easier filtering
    metrics_df = pd.DataFrame(vehicle_metrics)
    
    # Filter: long trajectories with good smoothness
    good_trajectories = metrics_df[
        (metrics_df['length'] >= MIN_TRAJECTORY_LENGTH) &
        (metrics_df['distance_meters'] >= MIN_DISTANCE_TRAVELED) &
        (metrics_df['smoothness'] < 0.5)  # Lower is smoother
    ].sort_values('distance_meters', ascending=False)
    
    print(f"  Found {len(good_trajectories)} good trajectories")
    print(f"  Top candidates:")
    for idx, row in good_trajectories.head(15).iterrows():
        print(f"    Track {row['track_id']}: {row['length']} points, "
              f"{row['distance_meters']:.1f}m, smoothness={row['smoothness']:.3f}")
    
    # Select top trajectories, ensuring diversity (different spatial regions)
    selected = []
    used_regions = []
    
    for idx, row in good_trajectories.iterrows():
        if len(selected) >= num_lanes:
            break
        
        track_id = row['track_id']
        traj = df[df['track_id'] == track_id].sort_values('frame')
        
        # Get average position
        avg_x = traj['center_x'].mean()
        avg_y = traj['center_y'].mean()
        
        # Check if this region is already covered
        too_close = False
        for used_x, used_y in used_regions:
            dist = np.sqrt((avg_x - used_x)**2 + (avg_y - used_y)**2)
            if dist < 200:  # Less than 200 pixels apart
                too_close = True
                break
        
        if not too_close:
            selected.append(track_id)
            used_regions.append((avg_x, avg_y))
            print(f"  ✓ Selected track {track_id} (region: {avg_x:.0f}, {avg_y:.0f})")
    
    return selected

def extract_lane_centerlines(df, track_ids):
    """Extract and process lane centerlines from selected tracks."""
    print(f"\nExtracting lane centerlines...")
    
    lanes = []
    for lane_id, track_id in enumerate(track_ids):
        traj = df[df['track_id'] == track_id].sort_values('frame')
        
        # Get center points
        points = traj[['center_x', 'center_y']].values
        
        # Smooth the trajectory
        points_smooth = smooth_trajectory(points)
        
        # Resample to have evenly spaced points
        points_resampled = resample_trajectory(points_smooth, num_points=50)
        
        # Convert to list of [x, y] pairs
        centerline_points = points_resampled.tolist()
        
        # Create lane object
        lane = {
            'lane_id': lane_id,
            'lane_number': lane_id,  # For visualizer compatibility
            'centerline_points': centerline_points,
            'points': centerline_points,  # For compatibility
            'spawn_enabled': True,  # Enable spawning on all lanes
            'traffic_lights': []  # Will add manually later if needed
        }
        
        lanes.append(lane)
        print(f"  ✓ Lane {lane_id} from track {track_id}: {len(centerline_points)} points")
    
    return lanes

def create_json_output(lanes):
    """Create JSON output structure."""
    output = {
        'image_width': IMAGE_WIDTH,
        'image_height': IMAGE_HEIGHT,
        'offset_x': OFFSET_X,
        'offset_y': OFFSET_Y,
        'scale': SCALE,
        'lanes': lanes,
        'lane_connections': []  # Can be added manually later
    }
    return output

def main():
    """Main execution."""
    print("="*60)
    print("Extracting Lane Centerlines from Real Car Data")
    print("="*60)
    
    # Load data
    df = load_data()
    
    # Find best trajectories
    track_ids = find_best_trajectories(df, num_lanes=9)
    
    if len(track_ids) == 0:
        print("\n❌ No suitable trajectories found!")
        print("   Try adjusting MIN_TRAJECTORY_LENGTH or MIN_DISTANCE_TRAVELED")
        return
    
    # Extract lane centerlines
    lanes = extract_lane_centerlines(df, track_ids)
    
    # Create JSON output
    output = create_json_output(lanes)
    
    # Save to file
    output_path = Path(OUTPUT_JSON)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Saved {len(lanes)} lanes to {OUTPUT_JSON}")
    print(f"\nNext steps:")
    print(f"  1. Review the lanes in the JSON file")
    print(f"  2. Add traffic lights manually if needed")
    print(f"  3. Add lane connections if needed")
    print(f"  4. Run simulation with: python run_sim.py")

if __name__ == "__main__":
    main()

