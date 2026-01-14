import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import os

def get_unique_filename(base_name, folder="generated_images"):
    """
    Generates a unique filename by appending v_1, v_2, etc.
    Ensures the output directory exists.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
        
    filename, ext = os.path.splitext(os.path.basename(base_name))
    
    counter = 1
    while True:
        new_filename = f"{filename}_v_{counter}{ext}"
        full_path = os.path.join(folder, new_filename)
        if not os.path.exists(full_path):
            return full_path
        counter += 1

def plot_lane_density(file_path):
    """
    Reads the simulation CSV and plots the number of cars on each lane over time.
    Generates 8 subplots, one for each lane.
    """
    # Load data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    # Filter for cars only (assuming class_id 1 is cars based on inspection)
    # Also ignore negative track_ids which are traffic lights
    cars_df = df[(df['class_id'] == 1) & (df['track_id'] >= 0)].copy()

    if cars_df.empty:
        print("No car data found in the file.")
        return
    
    # === NEW LOGIC: Focus on Jam Unloading Phase ===
    # Phase 2 starts at frame 350 (43.75s).
    # We redefine time so t=0 is the start of unloading.
    JAM_TRANSITION_FRAME = 350
    FPS = 8
    
    # Filter for unloading phase
    cars_df = cars_df[cars_df['frame'] >= JAM_TRANSITION_FRAME].copy()
    
    # Recalculate time relative to unloading start
    cars_df['time_sec'] = (cars_df['frame'] - JAM_TRANSITION_FRAME) / FPS
    
    # Limit analysis to first 130 seconds of unloading
    cars_df = cars_df[cars_df['time_sec'] <= 130]
        
    # Get all unique timestamps and sort them
    timestamps = sorted(cars_df['time_sec'].unique())
    max_time = 130 # User requested fixed 130s view
    
    # Get all unique lanes. Expecting 1-8 but will handle what's there.
    # We want 8 graphs, so we'll enforce lanes 1-8 if possible, or just unique ones.
    # The user asked for "one for each lane" and "8 graphs".
    target_lanes = range(1, 9)
    
    # Prepare plotting
    # Changed to 2 rows, 4 columns as requested
    fig, axes = plt.subplots(2, 4, figsize=(24, 10), sharex=False, sharey=True) # sharex=False so we can set limits manually
    axes = axes.flatten()
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # Define a color palette
    palette = sns.color_palette("husl", 8)

    print("Generating density plots...")

    for i, lane_id in enumerate(target_lanes):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Filter for specific lane
        lane_data = cars_df[cars_df['lane_id'] == lane_id]
        
        # Explicitly set x-axis limit to match global simulation time
        ax.set_xlim(0, max_time)
        
        if lane_data.empty:
            ax.text(0.5, 0.5, "No Data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title(f"Lane {lane_id}")
            continue

        # Group by timestamp and count unique cars (track_id)
        # Assuming one row per car per frame
        counts = lane_data.groupby('time_sec')['track_id'].count().reset_index()
        counts.columns = ['time_sec', 'car_count']
        
        # Plot
        sns.lineplot(data=counts, x='time_sec', y='car_count', ax=ax, color=palette[i], linewidth=2.5)
        
        # Fill area under the curve
        ax.fill_between(counts['time_sec'], counts['car_count'], alpha=0.3, color=palette[i])
        
        # Analyze drop to normal level
        # Start looking from 6th second
        # Find first time where count <= 2
        drop_condition = (counts['time_sec'] >= 6) & (counts['car_count'] <= 2)
        drop_points = counts[drop_condition]
        
        if not drop_points.empty:
            drop_time = drop_points.iloc[0]['time_sec']
            ax.axvline(x=drop_time, color='red', linestyle='--', linewidth=2, label=f'Relief: {drop_time:.1f}s')
            ax.legend(loc='upper right')
            print(f"Lane {lane_id}: Relief detected at {drop_time:.2f}s")
        
        ax.set_title(f"Lane {lane_id} Density", fontsize=14, fontweight='bold')
        if i % 4 == 0:
            ax.set_ylabel("Number of Cars")
        else:
            ax.set_ylabel("")
            
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add basic stats
        max_cars = counts['car_count'].max()
        avg_cars = counts['car_count'].mean()
        ax.text(0.02, 0.95, f"Max: {max_cars}\nAvg: {avg_cars:.1f}", 
                transform=ax.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.suptitle("Traffic Jam Relief - Car Density per Lane", fontsize=16, y=1.02)
    
    output_path = get_unique_filename('density_stats.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Density plot saved to {output_path}")
    # plt.show() # Uncomment if running interactively

def plot_lane_locations(file_path, image_path='eda/data/media/SiteA.jpg'):
    """
    Plots the location of each lane on the map image.
    """
    if not os.path.exists(image_path):
        # Try finding it relative to current script if defaults fail
        alt_path = os.path.join(os.path.dirname(__file__), 'eda/data/media/SiteA.jpg')
        if os.path.exists(alt_path):
            image_path = alt_path
        else:    
            print(f"Error: Image {image_path} not found.")
            return

    # Load data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return

    # Filter for cars
    cars_df = df[(df['class_id'] == 1) & (df['track_id'] >= 0)]
    
    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Prepare palette to match the graphs
    palette = sns.color_palette("husl", 8)
    
    target_lanes = range(1, 9)
    print("Generating lane map...")
    
    for i, lane_id in enumerate(target_lanes):
        lane_data = cars_df[cars_df['lane_id'] == lane_id]
        
        if lane_data.empty:
            continue
            
        # Calculate label position based on where cars START in this lane
        # This prevents the label from ending up in the middle of an intersection or trajectory
        start_positions = lane_data.sort_values('timestamp').groupby('track_id').first()
        
        center_x = start_positions['center_x'].mean()
        center_y = start_positions['center_y'].mean()
        
        # If for some reason mean is NaN (shouldn't be if data exists), fallback
        if np.isnan(center_x) or np.isnan(center_y):
             center_x = lane_data['center_x'].mean()
             center_y = lane_data['center_y'].mean()

        pos = (int(center_x), int(center_y))
        
        # Determine color (convert RGB float 0-1 tuple to BGR int 0-255)
        color_rgb = palette[i]
        color_bgr = (int(color_rgb[2]*255), int(color_rgb[1]*255), int(color_rgb[0]*255)) # OpenCV uses BGR
        
        # Draw Marker
        cv2.circle(img, pos, 40, color_bgr, -1)
        cv2.circle(img, pos, 40, (255, 255, 255), 2) # White border
        
        # Draw Text
        text = str(lane_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        thickness = 4
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = pos[0] - text_size[0] // 2
        text_y = pos[1] + text_size[1] // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)

    output_path = get_unique_filename('lane_map.png')
    cv2.imwrite(output_path, img)
    print(f"Lane map saved to {output_path}")

if __name__ == "__main__":
    import sys
    file_path = "jam_simulation.csv"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    plot_lane_density(file_path)
    plot_lane_locations(file_path, image_path='eda/data/media/SiteA.jpg')
