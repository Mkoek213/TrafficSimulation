import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from collections import defaultdict
from pathlib import Path
import argparse
from matplotlib.image import imread

def load_lanes_data(json_path):
    """Load lanes data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def visualize_lanes(data, background_image_path):
    """Visualize lanes on top of the background image."""
    # Read the background image
    img = imread(background_image_path)
    
    # Create figure and axis with the same dimensions as the image
    aspect_ratio = data['image_height'] / data['image_width']
    plt.figure(figsize=(15, 15 * aspect_ratio))
    ax = plt.gca()
    
    # Display the background image
    ax.imshow(img)
    
    # Plot each lane using a colorblind-friendly categorical colormap (tab10/tab20)
    # Use tab10 for up to 10 lanes; fallback to tab20 if more lanes exist.
    n_lanes = len(data['lanes'])
    cmap_name = 'tab10' if n_lanes <= 10 else 'tab20'
    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(i % cmap.N) for i in range(n_lanes)]

    for lane, color in zip(data['lanes'], colors):
        points = np.array(lane['centerline_points'])
        # Draw centerline
        ax.plot(points[:, 0], points[:, 1], '-', 
                label=f'Lane {lane["lane_number"]}', 
                color=color, 
                linewidth=2,
                alpha=0.9)

        # Mark and annotate each centerline point with its index
        for idx, (x, y) in enumerate(points):
            # small circular marker
            ax.plot(x, y, 'o', color=color, markersize=6, markeredgecolor='k', markeredgewidth=0.6)
            # add index text with a thin black stroke for readability on varied backgrounds
            txt = ax.text(x, y, str(idx), fontsize=7, color='white', ha='center', va='center')
            txt.set_path_effects([pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()])

        # Emphasize start and end points
        if len(points) > 0:
            ax.plot(points[0, 0], points[0, 1], 'D', color=color, markersize=7, markeredgecolor='k')
            ax.plot(points[-1, 0], points[-1, 1], 'X', color=color, markersize=7, markeredgecolor='k')

    # Group lanes by their start coordinates and annotate grouped labels
    start_map = defaultdict(list)
    for lane in data['lanes']:
        pts = lane.get('centerline_points', [])
        if not pts:
            continue
        try:
            sx, sy = int(pts[0][0]), int(pts[0][1])
        except Exception:
            # fallback to float->int
            try:
                sx, sy = int(float(pts[0][0])), int(float(pts[0][1]))
            except Exception:
                continue
        lane_id = lane.get('lane_id', lane.get('lane_number', None))
        current_there = next((x for x in start_map if (x[0] - sx)**2 + (x[1]- sy)**2 < 20), (sx, sy))
        start_map[current_there].append(lane_id)

    for (sx, sy), ids in start_map.items():
        try:
            if len(ids) == 1:
                label = f"lane: {ids[0]}"
            else:
                # show a sorted list for consistent ordering
                label = f"lanes: {sorted(ids)}"
            ann = ax.annotate(label, xy=(sx, sy), xytext=(-5, -5), textcoords='offset points',
                              fontsize=2.5, color='white', ha='left', va='bottom')
            ann.set_path_effects([pe.Stroke(linewidth=1.5, foreground='black'), pe.Normal()])
        except Exception:
            # don't fail the whole plotting if annotation crashes
            pass

    # Set plot limits to match image dimensions
    ax.set_xlim(0, data['image_width'])
    ax.set_ylim(data['image_height'], 0)  # Invert y-axis to match image coordinates
    
    # Add title and legend
    ax.set_title('Lane Centerlines Visualization')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove axis labels as they're not needed with the background image
    ax.set_xticks([])
    ax.set_yticks([])

    # Make layout tight
    plt.tight_layout()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize lanes over a background image.')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the background image file')
    parser.add_argument('--lanes', type=str,
                      help='Path to the lanes.json file (optional, defaults to eda/data/lanes.json)')
    parser.add_argument('--output', type=str,
                      help='Path to save the output image (optional, defaults to eda/data/lanes.jpg)',
                      default='eda/data/lanes.jpg')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Get the script's directory
    script_dir = Path(__file__).parent.parent
    
    # Use provided lanes.json path or default
    lanes_json_path = args.lanes if args.lanes else script_dir / 'data' / 'lanes.json'
    
    # Load and visualize the data
    data = load_lanes_data(lanes_json_path)
    visualize_lanes(data, args.image)
    
    # Save the plot to the specified output file
    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    print(f"Visualization saved to: {args.output}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()