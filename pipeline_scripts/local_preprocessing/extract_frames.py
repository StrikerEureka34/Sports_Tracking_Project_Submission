#!/usr/bin/env python3
import cv2
import argparse
import json
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path, output_dir, resize=None, max_frames=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    print(f"Video: {width}x{height} @ {fps:.2f} FPS")
    print(f"Extracting {total_frames} frames to {output_dir}/")
    
    if resize:
        print(f"Resizing to {resize[0]}x{resize[1]}")
    
    frame_idx = 0
    pbar = tqdm(total=total_frames, desc="Extracting frames")
    
    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break
        
        if resize:
            frame = cv2.resize(frame, resize)
        
        # Save frame with zero-padded filename
        frame_filename = output_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        frame_idx += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    
    # Save metadata
    metadata = {
        'total_frames': frame_idx,
        'original_fps': fps,
        'original_size': (width, height),
        'resize': resize,
        'video_path': str(video_path)
    }
    
    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Extracted {frame_idx} frames")
    print(f"✓ Metadata saved to {output_dir}/metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from video')
    parser.add_argument('video', type=str, help='Input video file')
    parser.add_argument('--output', type=str, default='frames', help='Output directory')
    parser.add_argument('--resize', type=str, default=None, 
                       help='Resize frames (format: WIDTHxHEIGHT, e.g., 1280x720)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum number of frames to extract')
    
    args = parser.parse_args()
    
    resize = None
    if args.resize:
        w, h = map(int, args.resize.split('x'))
        resize = (w, h)
    
    extract_frames(args.video, args.output, resize, args.max_frames)
