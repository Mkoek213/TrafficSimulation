"""
Traffic Trajectory Visualizer for DRIFT Dataset

Visualize vehicle movements with bounding boxes over time.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
from typing import Optional, Tuple, List
import colorsys


class TrafficVisualizer:
    """Visualize traffic trajectories from DRIFT dataset."""
    
    def __init__(self, image_path: str, csv_path: str, site_name: str = None):
        """
        Initialize visualizer with background image and trajectory data.
        
        Args:
            image_path: Path to site background image
            csv_path: Path to CSV file with trajectory data
            site_name: Site identifier (e.g., 'A', 'B'). Auto-detected if None.
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        print(f"Loading trajectory data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(self.df)} detections")
        print(f"✓ Tracks: {self.df['track_id'].nunique()}")
        print(f"✓ Frames: {self.df['frame'].min()} to {self.df['frame'].max()}")
        
        # Auto-detect site name from path or data
        if site_name is None:
            if 'site' in self.df.columns and len(self.df) > 0:
                site_name = self.df['site'].iloc[0].replace('Site ', '').strip()
            else:
                # Try to extract from path (e.g., site_A/drone_1.csv -> A)
                csv_path_parts = Path(csv_path).parts
                for part in csv_path_parts:
                    if part.startswith('site_'):
                        site_name = part.replace('site_', '').upper()
                        break
        self.site_name = site_name or "unknown"
        
        # Generate colors for each track
        self.track_colors = self._generate_colors(self.df['track_id'].nunique())
        
    def _generate_colors(self, n: int) -> dict:
        """Generate distinct colors for each track."""
        colors = {}
        unique_tracks = sorted(self.df['track_id'].unique())
        for i, track_id in enumerate(unique_tracks):
            hue = i / len(unique_tracks)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors[track_id] = tuple(int(c * 255) for c in rgb)
        return colors
    
    def draw_frame_opencv(self, frame_num: int, 
                         show_trails: bool = False,
                         trail_length: int = 30) -> np.ndarray:
        """
        Draw a single frame with bounding boxes using OpenCV.
        
        Args:
            frame_num: Frame number to render
            show_trails: Whether to show trajectory trails
            trail_length: Number of previous frames to show in trail
            
        Returns:
            Image array with drawn bounding boxes
        """
        img = self.image.copy()
        
        # Get detections for this frame
        frame_data = self.df[self.df['frame'] == frame_num]
        
        # Draw trails if requested
        if show_trails and frame_num > 0:
            trail_data = self.df[
                (self.df['frame'] > frame_num - trail_length) & 
                (self.df['frame'] < frame_num)
            ]
            for track_id in trail_data['track_id'].unique():
                track_trail = trail_data[trail_data['track_id'] == track_id].sort_values('frame')
                if len(track_trail) > 1:
                    points = track_trail[['center_x', 'center_y']].values.astype(np.int32)
                    color = self.track_colors.get(track_id, (128, 128, 128))
                    # Draw fading trail
                    for i in range(len(points) - 1):
                        alpha = (i + 1) / len(points)
                        faded_color = tuple(int(c * alpha) for c in color)
                        cv2.line(img, tuple(points[i]), tuple(points[i+1]), 
                               faded_color, 2, cv2.LINE_AA)
        
        # Draw current bounding boxes and traffic lights
        for _, row in frame_data.iterrows():
            # Traffic lights (class_id == 2) are drawn as simple colored circles
            class_id = int(row['class_id']) if 'class_id' in row and not pd.isna(row['class_id']) else 1
            if class_id == 2:
                state = row['tl_state'] if 'tl_state' in row and not pd.isna(row['tl_state']) else None
                if state == 'green':
                    tl_color = (0, 255, 0)
                elif state == 'yellow':
                    tl_color = (255, 255, 0)
                else:
                    tl_color = (255, 0, 0)

                center = (int(row['center_x']), int(row['center_y']))
                # Draw outer white border and colored center
                cv2.circle(img, center, 9, (255, 255, 255), -1)
                cv2.circle(img, center, 6, tl_color, -1)
                # Label as TL
                cv2.putText(img, "TL", (center[0] + 12, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                continue

            track_id = row['track_id']
            color = self.track_colors.get(track_id, (255, 255, 255))

            # Draw rotated bounding box using 4 corners
            corners = np.array([
                [row['x1'], row['y1']],
                [row['x2'], row['y2']],
                [row['x3'], row['y3']],
                [row['x4'], row['y4']]
            ], dtype=np.int32)

            # Draw filled polygon with transparency
            overlay = img.copy()
            cv2.fillPoly(overlay, [corners], color)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

            # Draw outline
            cv2.polylines(img, [corners], True, color, 2, cv2.LINE_AA)

            # Draw center point
            center = (int(row['center_x']), int(row['center_y']))
            cv2.circle(img, center, 4, color, -1)

            # Draw track ID
            cv2.putText(img, f"{int(track_id)}", 
                       (center[0] + 10, center[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        # Add frame info
        timestamp = frame_data['timestamp'].iloc[0] if len(frame_data) > 0 else ""
        cv2.putText(img, f"Frame: {frame_num} | Time: {timestamp}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"Vehicles: {len(frame_data)}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        return img
    
    def create_video(self, output_path: Optional[str] = None,
                    start_frame: Optional[int] = None,
                    end_frame: Optional[int] = None,
                    fps: int = 8,
                    show_trails: bool = True,
                    frame_skip: int = 1):
        """
        Create a video of traffic movements.
        
        Args:
            output_path: Output video file path (auto-generated if None)
            start_frame: Starting frame (None = first frame)
            end_frame: Ending frame (None = last frame)
            fps: Frames per second
            show_trails: Whether to show trajectory trails
            frame_skip: Skip every N frames for faster processing
        """
        if start_frame is None:
            start_frame = int(self.df['frame'].min())
        if end_frame is None:
            end_frame = int(self.df['frame'].max())
        
        # Auto-generate output path if not provided
        if output_path is None:
            # Use absolute path from project root
            import os
            # Assuming script is run from project root
            project_root = Path.cwd()
            output_dir = project_root / "eda" / "data" / "vis_videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            video_name = f"{self.site_name}_{start_frame}_{end_frame}.mp4"
            output_path = str(output_dir / video_name)
        else:
            # Ensure parent directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🎬 Creating video: {output_path}")
        print(f"   Site: {self.site_name}")
        print(f"   Frames: {start_frame} to {end_frame} (skip={frame_skip})")
        print(f"   FPS: {fps}")
        
        # Setup video writer
        height, width = self.image.shape[:2]
        
        # Use H.264 codec for better compatibility (especially on macOS)
        # Try different codecs in order of preference
        import platform
        system = platform.system()
        
        if system == "Darwin":  # macOS
            # H.264 codec works best on macOS
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
        else:
            # For Linux/Windows, try H.264 first, fallback to mp4v
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Check if video writer was opened successfully
        if not out.isOpened():
            print("⚠️  Warning: H.264 codec failed, trying MJPG...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            # Change extension to .avi for MJPG
            if output_path.endswith('.mp4'):
                output_path = output_path[:-4] + '.avi'
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Render frames
        frames = range(start_frame, end_frame + 1, frame_skip)
        for i, frame_num in enumerate(frames):
            if i % 10 == 0:
                print(f"   Processing frame {frame_num} ({i+1}/{len(frames)})...", end='\r')
            
            img = self.draw_frame_opencv(frame_num, show_trails=show_trails)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)
        
        out.release()
        print(f"\n✅ Video saved: {output_path}")
        print(f"   Codec: {'H.264 (avc1)' if system == 'Darwin' else 'H.264/MJPG'}")
    
    def show_frame_interactive(self, frame_num: int, show_trails: bool = True, save_path: str = None):
        """
        Display a single frame interactively using matplotlib.
        
        Args:
            frame_num: Frame number to display
            show_trails: Whether to show trajectory trails
            save_path: Optional path to save the frame instead of displaying
        """
        img = self.draw_frame_opencv(frame_num, show_trails=show_trails)
        
        plt.figure(figsize=(16, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Frame {frame_num}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Frame saved to: {save_path}")
        else:
            try:
                plt.show()
            except Exception as e:
                # If interactive display fails, save to a default location
                default_path = f"frame_{frame_num}.png"
                plt.savefig(default_path, dpi=150, bbox_inches='tight')
                print(f"⚠️  Could not display interactively. Frame saved to: {default_path}")
                print(f"   Error: {e}")
    
    def get_statistics(self) -> dict:
        """Get statistics about the trajectory data."""
        return {
            'total_detections': len(self.df),
            'unique_tracks': self.df['track_id'].nunique(),
            'frame_range': (int(self.df['frame'].min()), int(self.df['frame'].max())),
            'total_frames': int(self.df['frame'].max() - self.df['frame'].min() + 1),
            'avg_detections_per_frame': len(self.df) / self.df['frame'].nunique(),
            'classes': self.df['class_id'].value_counts().to_dict() if 'class_id' in self.df.columns else {}
        }
    
    def print_statistics(self):
        """Print statistics about the data."""
        stats = self.get_statistics()
        print("\n📊 Dataset Statistics:")
        print(f"   Total detections: {stats['total_detections']:,}")
        print(f"   Unique vehicles: {stats['unique_tracks']}")
        print(f"   Frame range: {stats['frame_range'][0]} to {stats['frame_range'][1]}")
        print(f"   Total frames: {stats['total_frames']}")
        print(f"   Avg vehicles/frame: {stats['avg_detections_per_frame']:.1f}")
        if stats['classes']:
            print(f"   Vehicle classes: {stats['classes']}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize DRIFT traffic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-save to eda/data/vis_videos with auto-generated name
  python eda/vis/traffic_visualizer.py \\
      --image eda/data/media/SiteA.jpg \\
      --csv eda/data/drift/site_A/drone_1.csv \\
      --start-frame 300 --end-frame 500

  # Custom output path
  python eda/vis/traffic_visualizer.py \\
      --image eda/data/media/SiteA.jpg \\
      --csv eda/data/drift/site_A/drone_1.csv \\
      --output my_video.mp4

  # Show single frame
  python eda/vis/traffic_visualizer.py \\
      --image eda/data/media/SiteA.jpg \\
      --csv eda/data/drift/site_A/drone_1.csv \\
      --show-only --start-frame 300
        """
    )
    parser.add_argument('--image', required=True, help='Path to site image')
    parser.add_argument('--csv', required=True, help='Path to trajectory CSV')
    parser.add_argument('--site', help='Site name (e.g., A, B). Auto-detected if not provided.')
    parser.add_argument('--output', help='Output video path (auto-generated if not provided)')
    parser.add_argument('--start-frame', type=int, help='Starting frame (default: first frame)')
    parser.add_argument('--end-frame', type=int, help='Ending frame (default: last frame)')
    parser.add_argument('--fps', type=int, default=8, help='Output FPS (default: 8)')
    parser.add_argument('--no-trails', action='store_true', help='Disable trajectory trails')
    parser.add_argument('--frame-skip', type=int, default=1, help='Skip every N frames (default: 1)')
    parser.add_argument('--show-only', action='store_true', help='Show single frame instead of video')
    parser.add_argument('--save-frame', help='Save frame to file instead of displaying (use with --show-only)')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = TrafficVisualizer(args.image, args.csv, site_name=args.site)
    viz.print_statistics()
    
    if args.show_only:
        # Show single frame
        frame_num = args.start_frame or int(viz.df['frame'].min())
        save_path = args.save_frame or f"frame_{frame_num}.png"
        viz.show_frame_interactive(frame_num, show_trails=not args.no_trails, save_path=save_path)
    else:
        # Create video
        viz.create_video(
            output_path=args.output,  # Will auto-generate if None
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            fps=args.fps,
            show_trails=not args.no_trails,
            frame_skip=args.frame_skip
        )


if __name__ == "__main__":
    main()

