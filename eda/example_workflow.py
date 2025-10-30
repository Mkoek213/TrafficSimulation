#!/usr/bin/env python3
"""
Example Workflow: From Download to Visualization

This script demonstrates the complete workflow:
1. Download data (optional - comment out if already downloaded)
2. Load and visualize traffic trajectories
3. Create video with auto-generated filename

Run from project root:
    python eda/example_workflow.py
"""

import sys
sys.path.append('.')

from eda.vis.traffic_visualizer import TrafficVisualizer

def main():
    print("="*60)
    print("Traffic Simulation - Example Workflow")
    print("="*60)
    
    # Configuration
    SITE = 'A'
    IMAGE_PATH = 'eda/data/media/SiteA.jpg'
    CSV_PATH = 'eda/data/drift/site_A/drone_1.csv'
    START_FRAME = 400
    END_FRAME = 800
    
    # Optional: Download data first (uncomment if needed)
    # import subprocess
    # print("\n1. Downloading data...")
    # subprocess.run(['python', 'eda/scripts/download_data.py', '--site', SITE, '--files', '2'])
    
    # Load visualizer
    print("\n1. Loading data...")
    viz = TrafficVisualizer(
        image_path=IMAGE_PATH,
        csv_path=CSV_PATH,
        site_name=SITE
    )
    
    # Show statistics
    viz.print_statistics()
    
    # Create a short video
    print("\n2. Creating visualization video...")
    print("   (This will auto-save to eda/data/vis_videos/)")
    
    viz.create_video(
        # output_path=None,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
        fps=30,
        show_trails=True,
        frame_skip=1
    )
    
    print("\n✅ Done! Check eda/data/vis_videos/ for your video.")
    print(f"   Video name: {SITE}_{START_FRAME}_{END_FRAME}.mp4")
    
    # Optional: Show a single frame interactively
    # print("\n3. Showing single frame...")
    # viz.show_frame_interactive(frame_num=400, show_trails=True)


if __name__ == "__main__":
    main()

