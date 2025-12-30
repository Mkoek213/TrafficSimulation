#!/usr/bin/env python3
"""
Compare Multiple Simulation Scenarios

This script reads multiple CSV output files from MESA simulations
and generates comparative statistics and plots.

Usage:
    python eda/scripts/compare_simulations.py \
      --scenarios "Scenario A=path/to/A.csv" "Scenario B=path/to/B.csv" \
      --output-dir eda/output/comparison_results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import sys

def load_scenario(name, path):
    """Load a single scenario CSV."""
    print(f"Loading {name} from {path}...")
    try:
        df = pd.read_csv(Path(path))
        df['Scenario'] = name
        return df
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

METERS_PER_PIXEL = 0.117  # From physics_analysis.py

def calculate_metrics(df, scenario_name):
    """Calculate aggregate metrics for a scenario DataFrame."""
    # Pre-calculate deltas for speed
    # Ensure sorted by track and timestamp/frame
    df = df.sort_values(['track_id', 'frame'])
    
    # Use pandas groupby operations for efficiency (like in friend's script)
    grouped = df.groupby('track_id')
    
    df['prev_x'] = grouped['center_x'].shift(1)
    df['prev_y'] = grouped['center_y'].shift(1)
    # Use timestamp for dt to be precise
    df['prev_time'] = grouped['timestamp'].shift(1)
    
    dx = df['center_x'] - df['prev_x']
    dy = df['center_y'] - df['prev_y']
    dt = df['timestamp'] - df['prev_time']
    
    dist_pixels = np.sqrt(dx**2 + dy**2)
    
    # Calculate raw speed in m/s
    # M_PER_PX = 0.117
    # speed (m/s) = (dist_px * 0.117) / dt
    
    # Filter valid dt
    valid_mask = (dt > 0.001)  # avoid div by zero
    
    df['speed_ms'] = np.nan
    df.loc[valid_mask, 'speed_ms'] = (dist_pixels[valid_mask] * METERS_PER_PIXEL) / dt[valid_mask]
    
    # Smoothing (Window=10, min_periods=3) - from physics_analysis.py
    df['speed_smooth_ms'] = grouped['speed_ms'].transform(lambda x: x.rolling(window=10, min_periods=3).mean())
    df['speed_kph'] = df['speed_smooth_ms'] * 3.6
    
    # Calculate Acceleration (m/s^2)
    # accel = delta_speed / dt
    df['prev_speed_ms'] = grouped['speed_smooth_ms'].shift(1)
    df['accel_ms2'] = (df['speed_smooth_ms'] - df['prev_speed_ms']) / dt
    
    # Extract significant braking/acceleration phases for stats
    # Braking: accel < -0.5
    braking_mask = (df['accel_ms2'] < -0.5) & (df['accel_ms2'] > -6.0) & (df['speed_smooth_ms'] > 2.0)
    decelerations = -df.loc[braking_mask, 'accel_ms2']
    
    # Acceleration: accel > 0.5
    accel_mask = (df['accel_ms2'] > 0.5) & (df['accel_ms2'] < 5.0) & (df['speed_smooth_ms'] > 1.0)
    accelerations = df.loc[accel_mask, 'accel_ms2']
    
    # 1. Active vehicles
    vehicles_per_frame = df.groupby('frame')['track_id'].nunique()
    
    # 2. Avg Speed per Frame (using smoothed kph)
    avg_speed_per_frame = df.groupby('frame')['speed_kph'].mean()
    
    # 3. Trip stats
    track_groups = df.groupby('track_id')
    vehicle_stats = []
    
    for track_id, group in track_groups:
        if len(group) < 2:
            continue
        t_start = group['timestamp'].min()
        t_end = group['timestamp'].max()
        duration = t_end - t_start
        if duration < 1.0:
            continue
            
        vehicle_stats.append({
            'track_id': track_id,
            'duration': duration,
        })
    
    vehicle_stats_df = pd.DataFrame(vehicle_stats)
    
    metrics = {
        'Scenario': scenario_name,
        'Total Vehicles': df['track_id'].nunique(),
        'Avg Travel Time (s)': vehicle_stats_df['duration'].mean() if not vehicle_stats_df.empty else 0,
        'Avg Speed (km/h)': df['speed_kph'].mean(), 
        'Avg Deceleration (m/s2)': decelerations.mean() if not decelerations.empty else 0,
        'Avg Acceleration (m/s2)': accelerations.mean() if not accelerations.empty else 0,
        'Throughput (veh/min)': (df['track_id'].nunique() / (df['timestamp'].max() - df['timestamp'].min())) * 60 if not df.empty else 0
    }
    
    return metrics, vehicle_stats_df, vehicles_per_frame, avg_speed_per_frame

def plot_comparisons(metrics_df, vehicle_stats_data, vehicles_per_frame_data, avg_speed_data, output_dir):
    """Generate and save comparison plots combined in one figure."""
    sns.set_style("whitegrid")
    
    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Traffic Simulation Comparative Analysis', fontsize=16)
    
    # 1. Bar Chart: Average Travel Time
    sns.barplot(data=metrics_df, x='Scenario', y='Avg Travel Time (s)', palette='viridis', ax=axes[0, 0])
    axes[0, 0].set_title('Average Travel Time')
    axes[0, 0].set_ylabel('Seconds')
    
    # 2. Line Chart: Average Speed over Time (NEW)
    for name, series in avg_speed_data.items():
        # Rolling average for smoothness
        smooth_series = series.rolling(window=10, min_periods=1).mean()
        axes[0, 1].plot(series.index, smooth_series.values, label=name, linewidth=2)
    
    axes[0, 1].set_title('Average Speed over Time')
    axes[0, 1].set_ylabel('Speed (km/h)')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(bottom=0)
    
    # 3. Line Plot: Active Vehicles over Time (Congestion)
    for name, series in vehicles_per_frame_data.items():
        axes[1, 0].plot(series.index, series.values, label=name, alpha=0.8, linewidth=2)
    
    axes[1, 0].set_title('Congestion Evolution (Active Vehicles)')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].legend()
    
    # 4. Bar Chart: Throughput
    sns.barplot(data=metrics_df, x='Scenario', y='Throughput (veh/min)', palette='magma', ax=axes[1, 1])
    axes[1, 1].set_title('Network Throughput')
    axes[1, 1].set_ylabel('Vehicles / Minute')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    # Save combined plot
    output_path = output_dir / 'combined_analysis.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved combined analysis plot to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare Traffic Simulation Scenarios")
    parser.add_argument('--scenarios', nargs='+', required=True, 
                        help='List of scenarios in "Name=Path" format, e.g., "Baseline=out/normal.csv"')
    parser.add_argument('--output-dir', required=True, help='Directory to save plots and stats')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_metrics = []
    valid_trips_data = {}
    vehicles_per_frame_data = {}
    
    avg_speed_data = {}
    
    print("-" * 50)
    print("Processing Scenarios...")
    print("-" * 50)
    
    for item in args.scenarios:
        if "=" not in item:
            print(f"Skipping invalid format: {item}. Use Name=Path")
            continue
            
        name, path = item.split('=', 1)
        df = load_scenario(name, path)
        
        if df is not None:
            met, trips, v_per_frame, avg_speed = calculate_metrics(df, name)
            all_metrics.append(met)
            valid_trips_data[name] = trips
            vehicles_per_frame_data[name] = v_per_frame
            avg_speed_data[name] = avg_speed
            print(f"Processed: {name}")
            print(f"  Avg Time: {met['Avg Travel Time (s)']:.1f}s")
            print(f"  Avg Speed: {met['Avg Speed (km/h)']:.1f} km/h")
            print(f"  Avg Accel: {met['Avg Acceleration (m/s2)']:.2f} m/s^2")
            print(f"  Avg Decel: {met['Avg Deceleration (m/s2)']:.2f} m/s^2")
            
    if not all_metrics:
        print("No valid scenarios processed.")
        return

    # Create Metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save Metrics to CSV
    metrics_csv_path = output_dir / 'comparison_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved metrics to {metrics_csv_path}")
    print(metrics_df.to_string(index=False))
    
    # Generate Plots
    print("\nGenerating Plots...")
    plot_comparisons(metrics_df, valid_trips_data, vehicles_per_frame_data, avg_speed_data, output_dir)
    print(f"Saved plots to {output_dir}")
    print("-" * 50)

if __name__ == "__main__":
    main()
