# Fixed Coordinate Transformation and Bounding Box Sizes

## Problem
The bounding boxes from the MESA simulation were:
1. **Wrong position**: Cars were spawning outside the roads
2. **Wrong size**: Bounding boxes were much too small compared to real data

## Solution

### 1. Coordinate Transformation Calibration
Ran `calibrate_coordinates.py` to analyze the ROI (Region of Interest) data and image:
- **Image size**: 3840x2160 pixels (SiteA.jpg)
- **ROI center**: (2064, 526) - this is where world origin (0,0) maps to
- **Scale**: 14 pixels per meter (calculated from lane width: 3.5m ≈ 49 pixels)

### 2. Fixed Coordinate Transformation
Updated `world_to_image_coordinates()` in `src/utils/bbox_utils.py`:
- Changed from normalized coordinate system to direct transformation
- Formula: `image_x = offset_x + world_x * scale`
- Formula: `image_y = offset_y - world_y * scale` (Y-axis flipped)

### 3. Fixed Bounding Box Sizes
Updated `simulation_to_csv.py`:
- Convert vehicle dimensions from meters to pixels: `length_pixels = length_meters * pixels_per_meter`
- Calculate bounding boxes directly in image coordinate space
- Default car size: 4.5m × 2.0m = **63 × 28 pixels** (matches real data)

### 4. Updated Default Parameters
Updated default parameters in:
- `run_mesa_simulation.py`: Defaults to calibrated values
- `example_mesa_simulation.py`: Uses calibrated values

## Calibrated Parameters

```python
image_width = 3840      # SiteA.jpg width
image_height = 2160    # SiteA.jpg height
offset_x = 2064.0       # ROI center X (where world origin maps to)
offset_y = 526.0       # ROI center Y (where world origin maps to)
scale = 14.0           # pixels per meter
```

## Usage

Now you can run the simulation and it will produce correctly positioned and sized bounding boxes:

```bash
# Run simulation
python run_mesa_simulation.py --output eda/data/mesa_simulation.csv --frames 300

# Visualize
python eda/vis/traffic_visualizer.py \
    --image eda/data/media/SiteA.jpg \
    --csv eda/data/mesa_simulation.csv
```

The cars should now:
- ✅ Appear on the actual roads (aligned with ROI)
- ✅ Have correct bounding box sizes (63×28 pixels for typical cars)
- ✅ Match the coordinate system of the real DRIFT dataset

