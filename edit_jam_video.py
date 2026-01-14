import cv2
import math
import sys

def edit_video(input_path="traffic_jam.mp4", output_path="traffic_jam_presentation.mp4"):
    """
    Edits the traffic jam simulation video for presentation:
    1. Speeds up the filling phase (Phase 1).
    2. Speeds up the clearing phase (Phase 2) slightly.
    3. Adds a timer overlay.
    """
    
    if not os.path.exists(input_path):
        print(f"Error: Input video '{input_path}' not found.")
        return

    print(f"Processing video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    
    # Input Properties
    input_fps = cap.get(cv2.CAP_PROP_FPS) # Should be 8.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_input_frames / input_fps
    
    print(f"Input: {width}x{height} @ {input_fps} fps, Duration: {duration_sec:.1f}s, Frames: {total_input_frames}")

    # Output Configuration
    TARGET_FPS = 30.0
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, codec, TARGET_FPS, (width, height))
    
    # === EDITING CONFIGURATION ===
    # Phase 1: Filling (0 to 350 frames approx) -> SKIPPED
    # Phase 2: Clearing (350+ frames)
    PHASE_SPLIT_FRAME = 350
    
    # We start directly from the clearing phase
    current_input_frame = float(PHASE_SPLIT_FRAME)
    
    # Calculate required speed to fit ~1000 frames into 15 seconds
    # Remaining frames = total_input_frames - PHASE_SPLIT_FRAME (approx 1050)
    # Target duration = 15 seconds
    remaining_frames = total_input_frames - PHASE_SPLIT_FRAME
    if remaining_frames <= 0:
        print("Error: No frames left for clearing phase.")
        return

    # To get 15s output at TARGET_FPS (30), we need 15 * 30 = 450 output frames.
    # So we need to cover 'remaining_frames' input frames in '450' output frames.
    # input_frames_per_output_frame = remaining_frames / 450
    # advance_step = input_frames_per_output_frame
    
    target_duration = 20.0 # seconds
    target_output_frames = target_duration * TARGET_FPS
    advance_step = remaining_frames / target_output_frames
    
    print(f"Cutting filling phase. Compressing {remaining_frames} frames into {target_duration}s.")
    print(f"Speed multiplier: {advance_step * TARGET_FPS / input_fps:.2f}x")
    
    frames_written = 0
    
    print("Editing in progress...")
    
    while True:
        # Read the frame at current index
        idx = int(current_input_frame)
        if idx >= total_input_frames:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
            
        # Draw Timer
        # Calculate simulation time (zeroed at start of video)
        sim_time_sec = max(0.0, (idx - PHASE_SPLIT_FRAME) / input_fps)
        minutes = int(sim_time_sec // 60)
        seconds = int(sim_time_sec % 60)
        
        # Format text
        text = f"T: {minutes:02d}:{seconds:02d}"
        font_scale = 5.0
        thickness = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Position
        x, y = 60, height - 60
        
        # Draw Black Box Background
        # Add padding
        pad = 30
        cv2.rectangle(frame, (x - pad, y + pad), (x + text_w + pad, y - text_h - pad), (0, 0, 0), -1)
        
        # Draw Main Text (White)
        cv2.putText(frame, text, (x, y), font, 
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # Write to output
        out.write(frame)
        frames_written += 1
        
        # Advance
        current_input_frame += advance_step
        
        if frames_written % 100 == 0:
            print(f"  Processed {int(current_input_frame)}/{total_input_frames} input frames -> {frames_written} output frames", end='\r')

    print(f"\nDone! Saved to: {output_path}")
    print(f"Output duration: {frames_written / TARGET_FPS:.1f}s")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import os
    edit_video()
