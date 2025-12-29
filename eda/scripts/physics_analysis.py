import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

def analyze_stopping_deceleration(df_list, m_per_px=0.117):
    """
    Calculates deceleration statistics by analyzing all significant braking phases.
    Improved to be more robust than just looking for full stops.
    """
    all_decelerations = []
    
    print("Analyzing continuous deceleration phases...")
    
    for df in df_list:
        if df.empty: continue
        
        df_calc = df.sort_values(['track_id', 'timestamp']).copy()
        df_calc['datetime'] = pd.to_datetime(df_calc['timestamp'], format='%H:%M:%S.%f')
        grouped = df_calc.groupby('track_id')
        
        df_calc['prev_x'] = grouped['center_x'].shift(1)
        df_calc['prev_y'] = grouped['center_y'].shift(1)
        df_calc['prev_time'] = grouped['datetime'].shift(1)
        
        dx = df_calc['center_x'] - df_calc['prev_x']
        dy = df_calc['center_y'] - df_calc['prev_y']
        dt = (df_calc['datetime'] - df_calc['prev_time']).dt.total_seconds()
        
        dist_px = np.sqrt(dx**2 + dy**2)
        valid_mask = (dt > 0.01) & (dt < 1.0)
        
        df_calc['speed'] = np.nan
        df_calc.loc[valid_mask, 'speed'] = (dist_px[valid_mask] * m_per_px) / dt[valid_mask]
        
        df_calc['speed_smooth'] = grouped['speed'].transform(lambda x: x.rolling(window=10, min_periods=3).mean())
        
        df_calc['prev_speed'] = grouped['speed_smooth'].shift(1)
        df_calc['accel'] = (df_calc['speed_smooth'] - df_calc['prev_speed']) / dt
        
        braking_mask = (df_calc['accel'] < -0.5) & \
                       (df_calc['accel'] > -6.0) & \
                       (df_calc['speed_smooth'] > 2.0)
        
        valid_decel = -df_calc.loc[braking_mask, 'accel'].values 
        all_decelerations.extend(valid_decel)

    if not all_decelerations:
        print("No significant deceleration detected.")
        return

    all_decelerations = np.array(all_decelerations)
    avg_decel = np.mean(all_decelerations)
    median_decel = np.median(all_decelerations)
    p95_decel = np.percentile(all_decelerations, 95)
    
    print(f"Analyzed {len(all_decelerations)} data points of braking.")
    print(f"Average Deceleration: {avg_decel:.2f} m/s²")
    print(f"Median Deceleration: {median_decel:.2f} m/s²")
    print(f"95th Percentile (Hard Braking): {p95_decel:.2f} m/s²")
    
    plt.figure(figsize=(12, 7))
    sns.histplot(all_decelerations, kde=True, bins=50, color='crimson', edgecolor=None, alpha=0.6)
    plt.axvline(avg_decel, color='blue', linestyle='--', linewidth=2, label=f'Mean: {avg_decel:.2f} m/s²')
    plt.axvline(median_decel, color='green', linestyle='-', linewidth=2, label=f'Median: {median_decel:.2f} m/s²')
    plt.title('Distribution of Vehicle Deceleration (Braking Phases)', fontsize=16)
    plt.xlabel('Deceleration Rate (m/s²)', fontsize=14)
    plt.ylabel('Frequency (Frames)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_acceleration_phases(df_list, m_per_px=0.117):
    """
    Calculates statistics for positive acceleration phases (speeding up).
    """
    all_accelerations = []
    
    print("Analyzing continuous acceleration phases...")
    
    for df in df_list:
        if df.empty: continue
        
        df_calc = df.sort_values(['track_id', 'timestamp']).copy()
        df_calc['datetime'] = pd.to_datetime(df_calc['timestamp'], format='%H:%M:%S.%f')
        grouped = df_calc.groupby('track_id')
        
        df_calc['prev_x'] = grouped['center_x'].shift(1)
        df_calc['prev_y'] = grouped['center_y'].shift(1)
        df_calc['prev_time'] = grouped['datetime'].shift(1)
        
        dx = df_calc['center_x'] - df_calc['prev_x']
        dy = df_calc['center_y'] - df_calc['prev_y']
        dt = (df_calc['datetime'] - df_calc['prev_time']).dt.total_seconds()
        
        dist_px = np.sqrt(dx**2 + dy**2)
        valid_mask = (dt > 0.01) & (dt < 1.0)
        
        df_calc['speed'] = np.nan
        df_calc.loc[valid_mask, 'speed'] = (dist_px[valid_mask] * m_per_px) / dt[valid_mask]
        
        df_calc['speed_smooth'] = grouped['speed'].transform(lambda x: x.rolling(window=10, min_periods=3).mean())
        
        df_calc['prev_speed'] = grouped['speed_smooth'].shift(1)
        df_calc['accel'] = (df_calc['speed_smooth'] - df_calc['prev_speed']) / dt
        
        accel_mask = (df_calc['accel'] > 0.5) & \
                     (df_calc['accel'] < 5.0) & \
                     (df_calc['speed_smooth'] > 1.0)
        
        valid_accel = df_calc.loc[accel_mask, 'accel'].values
        all_accelerations.extend(valid_accel)

    if not all_accelerations:
        print("No significant acceleration detected.")
        return

    all_accelerations = np.array(all_accelerations)
    avg_accel = np.mean(all_accelerations)
    median_accel = np.median(all_accelerations)
    p95_accel = np.percentile(all_accelerations, 95)
    
    print(f"Analyzed {len(all_accelerations)} data points of acceleration.")
    print(f"Average Acceleration: {avg_accel:.2f} m/s²")
    print(f"Median Acceleration: {median_accel:.2f} m/s²")
    print(f"95th Percentile: {p95_accel:.2f} m/s²")
    
    plt.figure(figsize=(12, 7))
    sns.histplot(all_accelerations, kde=True, bins=50, color='dodgerblue', edgecolor=None, alpha=0.6)
    plt.axvline(avg_accel, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_accel:.2f} m/s²')
    plt.axvline(median_accel, color='green', linestyle='-', linewidth=2, label=f'Median: {median_accel:.2f} m/s²')
    plt.title('Distribution of Vehicle Acceleration (Speeding Up)', fontsize=16)
    plt.xlabel('Acceleration Rate (m/s²)', fontsize=14)
    plt.ylabel('Frequency (Frames)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_standstill_distances(df_list, m_per_px=0.117, stop_speed_threshold=0.5):
    """
    Calculates center-to-center and bumper-to-bumper distances for stopped cars.
    """
    center_dists = []
    bumper_dists = []
    
    print("Analyzing standstill distances...")
    
    for df_idx, df in enumerate(df_list):
        if df.empty: continue
        
        df_calc = df.sort_values(['track_id', 'timestamp']).copy()
        df_calc['datetime'] = pd.to_datetime(df_calc['timestamp'], format='%H:%M:%S.%f')
        
        grouped = df_calc.groupby('track_id')
        dx = grouped['center_x'].diff()
        dy = grouped['center_y'].diff()
        dt = grouped['datetime'].diff().dt.total_seconds()
        
        dist_px = np.sqrt(dx**2 + dy**2)
        speed = (dist_px * m_per_px) / dt
        df_calc['speed'] = speed.fillna(0)
        
        stopped_df = df_calc[df_calc['speed'] < stop_speed_threshold].copy()
        
        if stopped_df.empty: continue
            
        timestamps = stopped_df['timestamp'].unique()
        timestamps = timestamps[::10] 
        
        for ts in timestamps:
            frame_cars = stopped_df[stopped_df['timestamp'] == ts]
            if len(frame_cars) < 2: continue
            
            coords = frame_cars[['center_x', 'center_y']].values
            angles = frame_cars['angle'].values
            ids = frame_cars['track_id'].values
            widths = frame_cars['width'].values
            heights = frame_cars['height'].values 
            
            tree = cKDTree(coords)
            dists, indices = tree.query(coords, k=2) 
            
            for i, (dist, idx_pair) in enumerate(zip(dists, indices)):
                neighbor_idx = idx_pair[1]
                if neighbor_idx == len(frame_cars): continue 
                
                dx = coords[neighbor_idx][0] - coords[i][0]
                dy = coords[neighbor_idx][1] - coords[i][1]
                
                heading_x = np.cos(angles[i])
                heading_y = np.sin(angles[i])
                
                longitudinal_dist = dx * heading_x + dy * heading_y
                lateral_dist = abs(-dx * heading_y + dy * heading_x)
                lateral_dist_m = lateral_dist * m_per_px
                
                if longitudinal_dist > 0 and lateral_dist_m < 2.5 and (longitudinal_dist * m_per_px) < 15.0:
                    c2c_px = np.sqrt(dx**2 + dy**2)
                    c2c_m = c2c_px * m_per_px
                    
                    length_rear = max(widths[i], heights[i]) 
                    length_front = max(widths[neighbor_idx], heights[neighbor_idx])
                    
                    gap_px = c2c_px - (length_rear/2 + length_front/2)
                    gap_m = gap_px * m_per_px
                    
                    if gap_m < -1.0: continue
                    gap_m = max(0.0, gap_m) 
                    
                    center_dists.append(c2c_m)
                    bumper_dists.append(gap_m)

    if not center_dists:
        print("No stopped cars found following each other.")
        return

    c2c = np.array(center_dists)
    b2b = np.array(bumper_dists)
    
    c2c = c2c[c2c < 20]
    b2b = b2b[b2b < 10]
    
    print(f"Analyzed {len(c2c)} stopped pairs.")
    print(f"Avg Center-to-Center: {np.mean(c2c):.2f} m (Median: {np.median(c2c):.2f} m)")
    print(f"Avg Bumper-to-Bumper: {np.mean(b2b):.2f} m (Median: {np.median(b2b):.2f} m)")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.histplot(c2c, kde=True, bins=40, color='teal', ax=axes[0])
    axes[0].set_title('Center-to-Center Distance (Standstill)', fontsize=14)
    axes[0].set_xlabel('Distance (m)', fontsize=12)
    axes[0].axvline(np.mean(c2c), color='red', linestyle='--', label=f'Mean: {np.mean(c2c):.2f} m')
    axes[0].legend()
    
    sns.histplot(b2b, kde=True, bins=40, color='orange', ax=axes[1])
    axes[1].set_title('Bumper-to-Bumper Gap (Standstill)', fontsize=14)
    axes[1].set_xlabel('Gap (m)', fontsize=12)
    axes[1].axvline(np.mean(b2b), color='red', linestyle='--', label=f'Mean: {np.mean(b2b):.2f} m')
    axes[1].axvline(np.median(b2b), color='blue', linestyle='-', label=f'Median: {np.median(b2b):.2f} m')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

