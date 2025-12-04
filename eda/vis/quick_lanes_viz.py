#!/usr/bin/env python3
"""Quick lane visualizer for lanes.json

Usage:
  python eda/vis/quick_lanes_viz.py --json eda/data/lanes.json --save lanes_viz.png

This script draws lane centerlines on top of the reference image (if present)
and labels each lane by its `lane_id`. It's intentionally small and dependency
light (matplotlib + pillow via matplotlib image read).
"""
import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def plot_lanes(data, ax):
    lanes = data.get('lanes', [])
    colors = plt.cm.tab20.colors
    for i, lane in enumerate(lanes):
        pts = lane.get('centerline_points', [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        color = colors[i % len(colors)]
        ax.plot(xs, ys, '-', color=color, linewidth=2)
        ax.scatter(xs, ys, s=8, color=color)
        # label lane_id near first point
        lane_id = lane.get('lane_id')
        ax.text(xs[0], ys[0], f"{lane_id}", color='k', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))


def main():
    parser = argparse.ArgumentParser(description='Quick visualizer for lanes.json')
    parser.add_argument('--json', '-j', default='eda/data/lanes.json', help='Path to lanes.json')
    parser.add_argument('--save', '-s', help='If set, save the figure to this path')
    parser.add_argument('--show', action='store_true', help='Display the figure')
    args = parser.parse_args()

    path = Path(args.json)
    if not path.exists():
        print(f"ERROR: {path} does not exist")
        return

    data = load_json(path)

    fig, ax = plt.subplots(figsize=(12, 7))

    # If image_path present, try to load and show as background
    image_path = data.get('image_path')
    if image_path:
        # image_path may be relative to repo root
        img_path = Path(image_path)
        if not img_path.exists():
            # try relative to json file
            img_path = path.parent / image_path
        if img_path.exists():
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(img.shape[0], 0)
        else:
            # fallback to image dimensions inside JSON if available
            w = data.get('image_width')
            h = data.get('image_height')
            if w and h:
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
    else:
        w = data.get('image_width')
        h = data.get('image_height')
        if w and h:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)

    plot_lanes(data, ax)

    ax.set_title(f"Lanes: {path}")
    ax.set_axis_off()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to {out}")

    if args.show or not args.save:
        plt.show()


if __name__ == '__main__':
    main()
