import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def visualize_entering_stats_multi(df_list, image_path, margin=200):
    """
    Aggregates statistics from multiple sessions and visualizes 8 entering lanes.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")
    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    all_starts = []
    total_duration = 0.0
    
    print(f"Processing {len(df_list)} sessions...")
    
    for df in df_list:
        if df.empty: continue
        times = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
        duration = (times.max() - times.min()).total_seconds()
        if duration <= 0: duration = 0.1 
        total_duration += duration
        
        df_sorted = df.sort_values('timestamp')
        starts = df_sorted.groupby('track_id').first().reset_index()
        all_starts.append(starts)
        
    if not all_starts:
        print("No valid data found.")
        return

    combined_starts = pd.concat(all_starts, ignore_index=True)
    
    print(f"{'='*55}")
    print(f"AGGREGATED TRAFFIC STATS (Total Duration: {total_duration:.2f}s)")
    print(f"{'='*55}")
    print(f"{'Lane':<10} | {'Side':<10} | {'Total Count':<12} | {'Freq (cars/s)':<15}")
    print(f"{'-'*60}")
    
    left_entries = combined_starts[combined_starts['center_x'] < margin].copy()
    top_entries = combined_starts[combined_starts['center_y'] < margin].copy()
    right_entries = combined_starts[combined_starts['center_x'] > (img_w - margin)].copy()
    
    def process_side(entries, n_lanes, coordinate_col, side_name, start_lane_id):
        if len(entries) < n_lanes:
            if len(entries) == 0:
                return start_lane_id + n_lanes
            actual_clusters = len(entries)
        else:
            actual_clusters = n_lanes

        coords = entries[coordinate_col].values.reshape(-1, 1)
        kmeans = KMeans(n_clusters=actual_clusters, random_state=42, n_init=10)
        kmeans.fit(coords)
        
        centers = kmeans.cluster_centers_.flatten()
        labels = kmeans.labels_
        sorted_indices = np.argsort(centers)
        
        for i in range(actual_clusters):
            cluster_idx = sorted_indices[i]
            center_val = centers[cluster_idx]
            count = np.sum(labels == cluster_idx)
            freq = count / total_duration
            
            current_lane_id = start_lane_id + i
            print(f"Lane {current_lane_id:<5} | {side_name:<10} | {count:<12} | {freq:.4f}")
            
            if side_name == 'Left':
                pos = (int(margin/2), int(center_val))
            elif side_name == 'Right':
                pos = (int(img_w - margin/2), int(center_val))
            elif side_name == 'Top':
                pos = (int(center_val), int(margin/2))
                
            cv2.circle(img_rgb, pos, 30, (255, 140, 0), -1) 
            text = str(current_lane_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            thickness = 3
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = pos[0] - text_size[0] // 2
            text_y = pos[1] + text_size[1] // 2
            cv2.putText(img_rgb, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            
        return start_lane_id + n_lanes

    next_id = process_side(left_entries, 2, 'center_y', 'Left', 1)
    next_id = process_side(top_entries, 3, 'center_x', 'Top', next_id)
    process_side(right_entries, 3, 'center_y', 'Right', next_id)
    
    print(f"{'='*60}")
    
    plt.figure(figsize=(18, 12))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f'Aggregate Entering Traffic (Total Duration: {total_duration/60:.1f} min)')
    plt.tight_layout()
    plt.show()

def visualize_speed_heatmap(df_list, image_path, m_per_px=0.117, bins=(100, 100), alpha=0.6, cmap='jet'):
    """
    Creates a heatmap of average vehicle speeds overlaid on the site image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found.")
    img_h, img_w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    all_speeds = []
    all_x = []
    all_y = []
    
    print("Calculating speeds across sessions...")
    
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
        
        current_speeds = (dist_px[valid_mask] * m_per_px) / dt[valid_mask]
        current_xs = df_calc.loc[valid_mask, 'center_x']
        current_ys = df_calc.loc[valid_mask, 'center_y']
        
        mask_clean = current_speeds < 40 
        
        if mask_clean.any():
            all_speeds.append(current_speeds[mask_clean].values)
            all_x.append(current_xs[mask_clean].values)
            all_y.append(current_ys[mask_clean].values)

    if not all_speeds:
        print("No valid speed data found.")
        return

    all_speeds = np.concatenate(all_speeds)
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    
    print(f"Computed speeds for {len(all_speeds)} points. Avg Speed: {np.mean(all_speeds):.2f} m/s")

    x_edges = np.linspace(0, img_w, bins[1] + 1)
    y_edges = np.linspace(0, img_h, bins[0] + 1)
    
    bin_x = np.digitize(all_x, x_edges) - 1
    bin_y = np.digitize(all_y, y_edges) - 1
    
    valid_bins = (bin_x >= 0) & (bin_x < bins[1]) & (bin_y >= 0) & (bin_y < bins[0])
    bin_x = bin_x[valid_bins]
    bin_y = bin_y[valid_bins]
    speeds = all_speeds[valid_bins]
    
    speed_sum = np.zeros(bins)
    count_grid = np.zeros(bins)
    flat_indices = bin_y * bins[1] + bin_x
    
    np.add.at(speed_sum.ravel(), flat_indices, speeds)
    np.add.at(count_grid.ravel(), flat_indices, 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_speed_grid = speed_sum / count_grid
        mean_speed_grid[count_grid == 0] = 0
        
    heatmap_full = cv2.resize(mean_speed_grid, (img_w, img_h), interpolation=cv2.INTER_CUBIC)
    mask = heatmap_full > 0.1
    
    vmax = np.percentile(all_speeds, 95)
    vmin = 0
    norm_heatmap = np.clip((heatmap_full - vmin) / (vmax - vmin), 0, 1)
    
    cmap_obj = plt.get_cmap(cmap)
    colored_heatmap = (cmap_obj(norm_heatmap)[:, :, :3] * 255).astype(np.uint8)
    
    overlay = img_rgb.copy()
    overlay[mask] = cv2.addWeighted(img_rgb[mask], 1 - alpha, colored_heatmap[mask], alpha, 0)
    
    plt.figure(figsize=(20, 12))
    ax = plt.gca()
    ax.imshow(overlay)
    ax.axis('off')
    ax.set_title(f'Average Vehicle Speed Heatmap (m/s)', fontsize=16)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label('Speed (m/s)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    plt.show()

