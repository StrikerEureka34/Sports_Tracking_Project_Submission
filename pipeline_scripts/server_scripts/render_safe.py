#!/usr/bin/env python3
"""Video renderer with batch processing to prevent memory issues."""

import cv2
import numpy as np
import json
import argparse
import time
import subprocess
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def render_video_safe(tracks_file, frames_dirs, output_video, fps=30.0, 
                      show_trajectories=False, start_frame=None, end_frame=None,
                      batch_size=500, target_height=720, crf=28):
    """Render video in small batches with encoder restart per batch."""
    
    print("=" * 60)
    print("SAFE VIDEO RENDERER (deadlock-resistant)")
    print("=" * 60)
    
    # Load tracks
    print(f"\nLoading tracks from {tracks_file}...")
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
    orig_height, orig_width = first_frame.shape[:2]
    
    # Scale down if needed
    if target_height and target_height < orig_height:
        scale = target_height / orig_height
        width = int(orig_width * scale)
        height = target_height
        print(f"Scaling: {orig_width}x{orig_height} -> {width}x{height} ({scale:.2f}x)")
    else:
        width, height = orig_width, orig_height
        scale = 1.0
    
    print(f"Video: {width}x{height} @ {fps} FPS")
    print(f"Batch size: {batch_size} frames (SMALL for safety)")
    print(f"CRF: {crf} (higher = faster, lower quality)")
    print(f"Total frames: {data['total_frames']}")
    print(f"Total tracks: {data['total_tracks']}")
    
    # Build frame index mapping
    frame_idx_map = {}
    for fdir in frames_dirs:
        for frame_file in sorted(fdir.glob('frame_*.jpg')):
            frame_num = int(frame_file.stem.split('_')[1])
            frame_idx_map[frame_num] = frame_file
    
    # Apply frame range filter
    if start_frame is not None or end_frame is not None:
        start = start_frame if start_frame is not None else 0
        end = end_frame if end_frame is not None else data['total_frames']
        print(f"Rendering frames {start} to {end-1}")
        frame_idx_map = {idx: path for idx, path in frame_idx_map.items() 
                        if start <= idx < end}
    
    frame_indices = sorted(frame_idx_map.keys())
    total_frames = len(frame_indices)
    print(f"Total frames to render: {total_frames}")
    
    # Split into batches
    total_batches = (total_frames + batch_size - 1) // batch_size
    print(f"Total batches: {total_batches}")
    
    # Output directory for batch segments
    output_path = Path(output_video)
    segments_dir = output_path.parent / f"{output_path.stem}_segments"
    segments_dir.mkdir(exist_ok=True)
    
    segment_files = []
    
    # Track history
    track_history = defaultdict(list)
    max_history_len = 50
    
    print("\n" + "=" * 60)
    print("RENDERING BATCHES (with encoder reopen per batch)")
    print("=" * 60)
    
    for batch_num in range(total_batches):
        batch_start = batch_num * batch_size
        batch_end = min(batch_start + batch_size, total_frames)
        batch_frames = frame_indices[batch_start:batch_end]
        
        segment_file = segments_dir / f"segment_{batch_num:04d}.mp4"
        segment_files.append(str(segment_file))
        
        print(f"\nBatch {batch_num + 1}/{total_batches}: Frames {batch_frames[0]}-{batch_frames[-1]} ({len(batch_frames)} frames)")
        print(f"  Output: {segment_file.name}")
        
        # REOPEN ENCODER FOR EACH BATCH (key anti-deadlock measure)
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', str(crf),
            '-threads', '4',  # Limited threads per batch
            '-pix_fmt', 'yuv420p',
            str(segment_file)
        ]
        
        try:
            proc = subprocess.Popen(
                ffmpeg_cmd, 
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL,  # Suppress ffmpeg noise
                bufsize=10**7  # 10MB buffer (smaller than before)
            )
            
            batch_start_time = time.time()
            frames_written = 0
            
            # Render this batch
            for idx, frame_idx in enumerate(batch_frames):
                frame_path = frame_idx_map.get(frame_idx)
                if not frame_path:
                    continue
                
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    continue
                
                # Scale if needed
                if scale != 1.0:
                    frame = cv2.resize(frame, (width, height))
                
                # Get tracks
                frame_tracks = tracks_by_frame.get(frame_idx, [])
                
                # Draw annotations
                for track in frame_tracks:
                    track_id = track['track_id']
                    bbox = track['bbox']
                    
                    # Scale bbox
                    x1, y1, x2, y2 = bbox
                    if scale != 1.0:
                        x1, y1, x2, y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
                    else:
                        x1, y1, x2, y2 = map(int, bbox)
                    
                    # Color
                    np.random.seed(track_id)
                    color = tuple(np.random.randint(100, 255, 3).tolist())
                    
                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Trajectory
                    if show_trajectories:
                        cx = int((x1 + x2) / 2)
                        cy = int(y2)
                        track_history[track_id].append((cx, cy))
                        if len(track_history[track_id]) > max_history_len:
                            track_history[track_id].pop(0)
                        
                        history = track_history[track_id]
                        if len(history) > 1:
                            points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                            cv2.polylines(frame, [points], False, color, 2)
                
                # Info
                info = f"Frame: {frame_idx} | Players: {len(frame_tracks)}"
                cv2.putText(frame, info, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write to pipe
                try:
                    proc.stdin.write(frame.tobytes())
                    frames_written += 1
                except BrokenPipeError:
                    print(f"  ERROR: Pipe broken at frame {idx}/{len(batch_frames)}")
                    break
                
                # Progress every 100 frames
                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - batch_start_time
                    fps_current = (idx + 1) / elapsed if elapsed > 0 else 0
                    print(f"    {idx+1}/{len(batch_frames)} frames ({fps_current:.1f} FPS)", end='\r', flush=True)
            
            # Close encoder for this batch
            proc.stdin.close()
            proc.wait(timeout=30)  # Wait max 30s for encoder to finish
            
            batch_elapsed = time.time() - batch_start_time
            print(f"  ✓ Batch {batch_num + 1} complete: {frames_written} frames in {batch_elapsed:.1f}s ({frames_written/batch_elapsed:.1f} FPS)")
            
        except subprocess.TimeoutExpired:
            print(f"  ERROR: Batch {batch_num + 1} encoder timeout!")
            proc.kill()
            continue
        except Exception as e:
            print(f"  ERROR: Batch {batch_num + 1} failed: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("CONCATENATING SEGMENTS")
    print("=" * 60)
    
    # Concatenate all segments
    concat_file = segments_dir / "concat_list.txt"
    with open(concat_file, 'w') as f:
        for seg in segment_files:
            if Path(seg).exists():
                f.write(f"file '{Path(seg).name}'\n")
    
    print(f"Concatenating {len(segment_files)} segments...")
    concat_cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(concat_file),
        '-c', 'copy',  # Stream copy (no re-encoding)
        str(output_video)
    ]
    
    result = subprocess.run(concat_cmd, capture_output=True)
    if result.returncode == 0:
        print(f"✓ Final video: {output_video}")
        
        # Get video info
        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,nb_frames',
            '-of', 'default=noprint_wrappers=1',
            str(output_video)
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        print(probe_result.stdout)
        
        # Cleanup segments (optional - commented out for safety)
        # print(f"Cleaning up {len(segment_files)} segment files...")
        # for seg in segment_files:
        #     Path(seg).unlink(missing_ok=True)
        # concat_file.unlink(missing_ok=True)
        
        print(f"\n✓ Rendering complete!")
        print(f"  Output: {output_video}")
        print(f"  Segments kept in: {segments_dir} (can delete manually)")
        
    else:
        print(f"ERROR: Concatenation failed!")
        print(result.stderr.decode())
        print(f"Segments saved in: {segments_dir}")
        print(f"You can manually concatenate or use individual segments")
    
    return str(output_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Safe video renderer with deadlock prevention')
    parser.add_argument('tracks_file', help='Tracks JSON file')
    parser.add_argument('frames_dirs', nargs='+', help='Directories containing frames')
    parser.add_argument('--output', '-o', required=True, help='Output video file')
    parser.add_argument('--fps', type=float, default=30.0, help='Video FPS (default: 30)')
    parser.add_argument('--start-frame', type=int, help='Start frame (inclusive)')
    parser.add_argument('--end-frame', type=int, help='End frame (exclusive)')
    parser.add_argument('--trajectories', action='store_true', help='Show trajectories')
    parser.add_argument('--batch-size', type=int, default=500, help='Frames per batch (default: 500)')
    parser.add_argument('--height', type=int, default=720, help='Target height (default: 720)')
    parser.add_argument('--crf', type=int, default=28, help='CRF value (default: 28, higher=faster)')
    
    args = parser.parse_args()
    
    render_video_safe(
        args.tracks_file,
        args.frames_dirs,
        args.output,
        fps=args.fps,
        show_trajectories=args.trajectories,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        batch_size=args.batch_size,
        target_height=args.height,
        crf=args.crf
    )
