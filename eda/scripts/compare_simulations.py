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

PIXELS_PER_METER = 11.0  # Corrections based on user feedback (likely Scale=11 default)

def calculate_metrics(df, scenario_name):
    """Calculate aggregate metrics for a scenario DataFrame."""
    # Group by track to calculate total stats
    track_groups = df.groupby('track_id')
    vehicle_stats = []
    
    # Pre-calculate deltas for speed
    df = df.sort_values(['track_id', 'frame'])
    df['prev_x'] = df.groupby('track_id')['center_x'].shift(1)
    df['prev_y'] = df.groupby('track_id')['center_y'].shift(1)
    df['prev_time'] = df.groupby('track_id')['timestamp'].shift(1)
    
    # Calculate instantaneous speed (distance / time)
    # dist = sqrt((x2-x1)^2 + (y2-y1)^2)
    # speed = dist / dt
    dx = df['center_x'] - df['prev_x']
    dy = df['center_y'] - df['prev_y']
    dt = df['timestamp'] - df['prev_time']
    
    dist_pixels = np.sqrt(dx**2 + dy**2)
    dist_meters = dist_pixels / PIXELS_PER_METER
    
    # Filter out zero dt (shouldn't happen with valid frames but safety check)
    with np.errstate(divide='ignore', invalid='ignore'):
        speed_ms = dist_meters / dt
        speed_ms = np.where(dt > 0, speed_ms, np.nan)
    
    df['speed_kph'] = speed_ms * 3.6
    
    # 1. Active vehicles over time (per frame)
    vehicles_per_frame = df.groupby('frame')['track_id'].nunique()
    
    # 2. Avg Speed per Frame
    avg_speed_per_frame = df.groupby('frame')['speed_kph'].mean()
    
    # 3. Trip stats
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
            # We could sum incremental distances, but simple euclidian is decent approx for straight lanes
            # 'distance_m': ...
        })
    
    vehicle_stats_df = pd.DataFrame(vehicle_stats)
    
    metrics = {
        'Scenario': scenario_name,
        'Total Vehicles': df['track_id'].nunique(),
        'Avg Travel Time (s)': vehicle_stats_df['duration'].mean() if not vehicle_stats_df.empty else 0,
        'Avg Speed (km/h)': df['speed_kph'].mean(), # Global average of all instant speeds
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
            print(f"Processed: {name} (Avg Time: {met['Avg Travel Time (s)']:.1f}s, Avg Speed: {met['Avg Speed (km/h)']:.1f} km/h)")
            
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
