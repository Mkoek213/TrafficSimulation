# Lane Marking Guide

## Quick Start

1. **Mark lanes on the image:**
   ```bash
   python mark_lanes.py --image eda/data/media/SiteA.jpg --output eda/data/lanes.json
   ```

2. **Run simulation with custom lanes:**
   ```bash
   python run_mesa_simulation.py \
       --output eda/data/mesa_simulation.csv \
       --frames 300 \
       --custom-lanes eda/data/lanes.json
   ```

## Lane Numbering System

**IMPORTANT:** Mark lanes from **BOTTOM to TOP** in order:

- **Lane 1** = Most bottom line (southernmost on image)
- **Lane 2** = Next line above Lane 1
- **Lane 3** = Next line above Lane 2
- **Lane 4** = Next line above Lane 3
- ... and so on

The numbering helps the simulation understand lane relationships and allows vehicles to change lanes properly.

## Instructions for mark_lanes.py

1. **Click LEFT MOUSE** to add points along a lane centerline
   - Click multiple points to mark the full length of the lane
   - The lane will be drawn as a blue line while you're marking it

2. **Press 'n'** to finish current lane and start a new one
   - The completed lane will turn green
   - Lane number will be displayed

3. **Press 's'** to save lanes to file
   - Saves all completed lanes to the output JSON file

4. **Press 'q'** to quit
   - Automatically saves lanes before quitting

5. **Press 'c'** to clear current lane (if you make a mistake)

6. **Press 'u'** to undo last point

## Tips

- **Mark lane centerlines**, not edges
- **Use multiple points** for curved lanes (at least 2 points, more for curves)
- **Mark all lanes** you want vehicles to use
- **Start from bottom** and work your way up
- **Mark both directions** if you want bidirectional traffic

## Marking Special Lane Configurations

### Left-Turn Lanes at Intersections

When marking a **left-turn lane that splits from a main road** (e.g., Lane 3 splitting from Road 2):

1. **Mark the main road lane first** (Lane 2):
   - Start from the beginning of the road
   - Continue through the intersection (or mark up to where the split begins)
   - Mark the centerline of the through lane

2. **Mark the left-turn lane** (Lane 3):
   - **Start where the lane splits** from the main road (before the traffic light/intersection)
   - Mark points along the **centerline** of the left-turn lane
   - Continue through the turn until the lane ends or connects to another road
   - The lane should curve to show the left turn

**Example for Lane 3 (left-turn lane splitting from Road 2):**
```
Lane 2 (through lane): [start] --------|------ [continues straight]
                                          |
Lane 3 (left-turn):              [split]|----- [curves left] ----- [turns]
```

**Key Points:**
- Lane 3 should start **slightly before or at the split point** from Lane 2
- Use multiple points (5-10+) to accurately trace the curved left turn
- The split point should be visible - mark it clearly with points
- Mark the full curve of the turn, not just the beginning

### Visual Guide

```
Road 2 (horizontal, going right):
┌─────────────────────────────────────┐
│  Lane 2 (through) ──────────────── │
│                                      │
│  Lane 3 (left-turn) ──┐             │ ← Split point (before traffic light)
│                        │             │
│                        │ (curves)    │
│                        ↓             │
│                      [left turn]     │
└─────────────────────────────────────┘
```

The simulation will treat each lane independently, so vehicles on Lane 2 can continue straight, while vehicles can spawn on Lane 3 to make left turns.

## Example Workflow

```bash
# 1. Mark lanes interactively
python mark_lanes.py --image eda/data/media/SiteA.jpg --output eda/data/lanes.json

# 2. Generate simulation with custom lanes
python run_mesa_simulation.py \
    --output eda/data/mesa_simulation.csv \
    --frames 300 \
    --max-vehicles 20 \
    --custom-lanes eda/data/lanes.json

# 3. Visualize
python eda/vis/traffic_visualizer.py \
    --image eda/data/media/SiteA.jpg \
    --csv eda/data/mesa_simulation.csv
```

The vehicles will now spawn and move along your manually marked lanes!

