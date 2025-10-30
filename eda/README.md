# EDA - Traffic Simulation Data Analysis

This directory contains all exploratory data analysis (EDA) tools, scripts, and data for the DRIFT traffic dataset.

## 📁 Directory Structure

```
eda/
├── data/                      # All data files
│   ├── drift/                # Downloaded DRIFT dataset
│   │   └── site_A/          # Site A trajectory data
│   ├── media/               # Site images and videos
│   │   └── SiteA.jpg       # Site A background image
│   ├── roi.json            # Region of Interest definitions
│   └── vis_videos/         # Generated visualization videos
├── notebooks/              # Jupyter notebooks for analysis
│   ├── eda.ipynb
│   └── site_roi_vis.ipynb
├── scripts/               # Utility scripts
│   └── download_data.py   # Download DRIFT dataset
├── vis/                   # Visualization tools
│   └── traffic_visualizer.py  # Traffic trajectory visualizer
└── output/               # General output directory
```

## 🚀 Quick Start

### 1. Download Data

Download Site A data (or any other site):

```bash
# From project root
python eda/scripts/download_data.py --site A --files 2

# Download multiple sites
python eda/scripts/download_data.py --site A B C --files 5

# Download all files from Site A
python eda/scripts/download_data.py --site A
```

### 2. Visualize Traffic

Create traffic visualization videos:

```bash
# Auto-save to eda/data/vis_videos/ with name: A_300_500.mp4
python eda/vis/traffic_visualizer.py \
    --image eda/data/media/SiteA.jpg \
    --csv eda/data/drift/site_A/drone_1.csv \
    --start-frame 300 --end-frame 500

# Custom output path
python eda/vis/traffic_visualizer.py \
    --image eda/data/media/SiteA.jpg \
    --csv eda/data/drift/site_A/drone_1.csv \
    --output my_video.mp4 \
    --start-frame 300 --end-frame 500

# Show single frame (interactive)
python eda/vis/traffic_visualizer.py \
    --image eda/data/media/SiteA.jpg \
    --csv eda/data/drift/site_A/drone_1.csv \
    --show-only --start-frame 300
```

## 📊 Python API

### Download Data

```python
# Run from project root
import subprocess

subprocess.run([
    'python', 'eda/scripts/download_data.py',
    '--site', 'A',
    '--files', '2'
])
```

### Visualize Traffic

```python
import sys
sys.path.append('.')

from eda.vis.traffic_visualizer import TrafficVisualizer

# Create visualizer
viz = TrafficVisualizer(
    image_path='eda/data/media/SiteA.jpg',
    csv_path='eda/data/drift/site_A/drone_1.csv'
)

# Show statistics
viz.print_statistics()

# Create video (auto-saved to eda/data/vis_videos/A_300_500.mp4)
viz.create_video(
    start_frame=300,
    end_frame=500,
    fps=30,
    show_trails=True
)

# Show single frame
viz.show_frame_interactive(frame_num=300, show_trails=True)
```

## 📝 Notes

### Video Naming Convention

Videos are automatically saved with the format: `{site}_{start_frame}_{end_frame}.mp4`

Examples:
- `A_300_500.mp4` - Site A, frames 300-500
- `B_1000_2000.mp4` - Site B, frames 1000-2000

### Data Paths

All scripts assume you run them from the **project root** (`/Users/brader/Desktop/TrafficSimulation/`):

```bash
# Correct (from project root)
python eda/scripts/download_data.py --site A

# Incorrect (from eda/ directory)
cd eda
python scripts/download_data.py --site A  # This will create wrong paths!
```

### Site Names

Available sites in DRIFT dataset: A, B, C, D, E, F, G, H, I

Site A is the primary focus for this project.

## 🔧 Troubleshooting

**Issue**: Videos not saving to correct location
- **Solution**: Make sure you run scripts from the project root directory

**Issue**: Cannot find image/CSV files
- **Solution**: Check that data has been downloaded with `download_data.py` first

**Issue**: Import errors for TrafficVisualizer
- **Solution**: Make sure to add project root to path: `sys.path.append('.')`

