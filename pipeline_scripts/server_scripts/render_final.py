#!/usr/bin/env python3
import cv2
import numpy as np
import json
import argparse
import gc
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from visualization import VideoVisualizer


def check_nvenc_available():
    """Check if ffmpeg supports NVENC hardware encoding."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return 'h264_nvenc' in result.stdout
    except:
        return False


def render_final_video(tracks_file, frames_dirs, output_video, fps=30.0, show_trajectories=True, start_frame=None, end_frame=None):
    """
    Render final annotated video from tracks and frames.
    
    Args:
        tracks_file: Tracks JSON with bbox and track_id per frame
        frames_dirs: List of directories containing frames (batches)
        output_video: Output video path
        fps: Video FPS
        show_trajectories: Draw trajectories
        start_frame: Start frame (inclusive), None for beginning
        end_frame: End frame (exclusive), None for end
        
    Returns:
        output_video: Path to rendered video
    """
    # Load tracks
    with open(tracks_file, 'r') as f:
        data = json.load(f)
    
    tracks_by_frame = {t['frame']: t['tracks'] for t in data['tracks']}
    
    # Load first frame to get dimensions
    frames_dirs = [Path(d) for d in frames_dirs]
    first_frame_path = None
    for fdir in frames_dirs:
        frames = sorted(fdir.glob('frame_*.jpg'))
        if frames:
            first_frame_path = frames[0]
            break
    
    if not first_frame_path:
        raise ValueError("No frames found")
    
    first_frame = cv2.imread(str(first_frame_path))
    height, width = first_frame.shape[:2]
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    print(f"Total frames: {data['total_frames']}")
    print(f"Total tracks: {data['total_tracks']}")
    
    # Initialize visualizer
    viz = VideoVisualizer()
    
    # A100 doesn't support NVENC (compute GPU, encoder disabled)
    # Use optimized CPU encoding instead
    print("Using optimized CPU encoding (libx264 ultrafast)...")
    ffmpeg_cmd = [
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
        '-c:v', 'libx264', 
        '-preset', 'ultrafast',  # Fastest CPU preset
        '-crf', '23',  # Slightly lower quality for speed
        '-threads', '0',  # Use all CPU threads
        '-pix_fmt', 'yuv420p', 
        str(output_video)
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=10**8)
    writer = None
    use_nvenc = False
    
    # Track history for trajectories
    track_history = defaultdict(list)
    track_last_seen = {}  # Track when each ID was last seen to clean up old tracks
    max_history_len = 50
    max_unseen_frames = 150  # Remove tracks not seen for 150 frames to prevent memory leak
    
    # Collect all frames from all batches
    all_frames = []
    for fdir in frames_dirs:
        all_frames.extend(sorted(fdir.glob('frame_*.jpg')))
    
    all_frames.sort()  # Sort by name (frame_000000, frame_000001, ...)
    
    # Build frame index mapping
    frame_idx_map = {}
    for frame_path in all_frames:
        frame_name = frame_path.stem
        frame_idx = int(frame_name.split('_')[1])
        frame_idx_map[frame_idx] = frame_path
    
    # Apply frame range filter if specified
    if start_frame is not None or end_frame is not None:
        start = start_frame if start_frame is not None else 0
        end = end_frame if end_frame is not None else data['total_frames']
        print(f"Rendering frames {start} to {end-1} (range specified)")
        frame_idx_map = {idx: path for idx, path in frame_idx_map.items() 
                        if start <= idx < end}
        total_frames_to_render = len(frame_idx_map)
    else:
        total_frames_to_render = data['total_frames']
    
    print(f"Total frames to render: {total_frames_to_render}")
    
    # Batch-load frames for faster rendering  
    print("Rendering video with batch loading...")
    batch_size = 2000  # Increased to 2000 frames per batch for better CPU utilization
    total_batches = (total_frames_to_render + batch_size - 1) // batch_size
    
    # Get frame indices to render
    frame_indices = sorted(frame_idx_map.keys())
    
    for batch_num in range(total_batches):
        batch_start_idx = batch_num * batch_size
        batch_end_idx = min(batch_start_idx + batch_size, len(frame_indices))
        
        # Pre-load batch into memory
        batch_frames = frame_indices[batch_start_idx:batch_end_idx]
        print(f"Batch {batch_num + 1}/{total_batches}: Loading {len(batch_frames)} frames...", flush=True)
        batch_frame_data = {}
        frames_loaded = 0
        for frame_idx in batch_frames:
            frame = cv2.imread(str(frame_idx_map[frame_idx]))
            if frame is not None:
                batch_frame_data[frame_idx] = frame
                frames_loaded += 1
        
        print(f"Batch {batch_num + 1}/{total_batches}: Loaded {frames_loaded} frames, rendering...", flush=True)
        
        # Process loaded batch (no tqdm for better CPU utilization)
        frames_rendered = 0
        for frame_idx in batch_frames:
            frame = batch_frame_data.get(frame_idx)
            if frame is None:
                continue
            
            # Get tracks for this frame
            frame_tracks = tracks_by_frame.get(frame_idx, [])
            
            # Draw bounding boxes and IDs
            for track in frame_tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                
                x1, y1, x2, y2 = map(int, bbox)
                
                # Random color per track
                np.random.seed(track_id)
                color = tuple(np.random.randint(100, 255, 3).tolist())
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID
                label = f"ID:{track_id}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 2, cv2.LINE_AA)
                
                # Update trajectory
                cx = int((x1 + x2) / 2)
                cy = int(y2)  # Bottom center
                track_history[track_id].append((cx, cy))
                track_last_seen[track_id] = frame_idx  # Update last seen time
                
                # Limit history length
                if len(track_history[track_id]) > max_history_len:
                    track_history[track_id].pop(0)
            
            # Draw trajectories only for visible tracks (much faster than iterating all 636 tracks)
            if show_trajectories:
                for track in frame_tracks:
                    track_id = track['track_id']
                    history = track_history.get(track_id, [])
                    if len(history) > 1:
                        np.random.seed(track_id)
                        color = tuple(np.random.randint(100, 255, 3).tolist())
                        
                        # Use polylines for faster drawing
                        points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [points], False, color, 2)
            
            # Info panel
            info_text = f"Frame: {frame_idx} | Players: {len(frame_tracks)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Write frame
            if ffmpeg_proc is not None:
                try:
                    ffmpeg_proc.stdin.write(frame.tobytes())
                    frames_rendered += 1
                except BrokenPipeError:
                    print("\nError: ffmpeg process terminated unexpectedly")
                    break
            else:
                writer.write(frame)
                frames_rendered += 1
        
        # Progress update
        print(f"Batch {batch_num + 1}/{total_batches}: Rendered {frames_rendered} frames ({batch_end_idx}/{total_frames_to_render} total)", flush=True)
        
        # Clean up old tracks to prevent memory leak (remove tracks not seen in 150 frames)
        if batch_frames:
            current_frame = batch_frames[-1]
            tracks_to_remove = [tid for tid, last_seen in track_last_seen.items() 
                               if current_frame - last_seen > max_unseen_frames]
            for tid in tracks_to_remove:
                if tid in track_history:
                    del track_history[tid]
                del track_last_seen[tid]
            
            if tracks_to_remove:
                print(f"  Cleaned up {len(tracks_to_remove)} old tracks (keeping {len(track_history)} active)", flush=True)
        
        # Explicitly clear batch from memory and force garbage collection
        del batch_frame_data
        import gc
        gc.collect()
        
        # Flush ffmpeg pipe
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.flush()
            except:
                pass
    
    # Cleanup
    if ffmpeg_proc is not None:
        try:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait(timeout=60)
            if ffmpeg_proc.returncode != 0:
                stderr = ffmpeg_proc.stderr.read().decode('utf-8', errors='ignore')
                print(f"\nWarning: ffmpeg returned code {ffmpeg_proc.returncode}")
                print(f"ffmpeg stderr (last 500 chars): {stderr[-500:]}")
        except Exception as e:
            print(f"\nWarning during ffmpeg cleanup: {e}")
            if ffmpeg_proc:
                ffmpeg_proc.kill()
    elif writer is not None:
        writer.release()
    
    print(f"\n✓ Video rendered: {output_video}")
    print(f"  Duration: {data['total_frames'] / fps:.2f} seconds")
    
    return output_video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render final annotated video')
    parser.add_argument('tracks_file', type=str, help='Tracks JSON file')
    parser.add_argument('frames_dirs', type=str, nargs='+', 
                       help='Directories containing frames (can be multiple batch dirs)')
    parser.add_argument('--output', type=str, default='output_local/final_output.mp4',
                       help='Output video file')
    parser.add_argument('--fps', type=float, default=30.0, help='Video FPS')
    parser.add_argument('--no-trajectories', action='store_true',
                       help='Disable trajectory drawing')
    parser.add_argument('--start-frame', type=int, default=None,
                       help='Start frame (inclusive)')
    parser.add_argument('--end-frame', type=int, default=None,
                       help='End frame (exclusive)')
    
    args = parser.parse_args()
    
    render_final_video(args.tracks_file, args.frames_dirs, args.output, 
                      args.fps, not args.no_trajectories, 
                      args.start_frame, args.end_frame)
