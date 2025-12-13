#!/usr/bin/env python3
import shutil
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def split_into_batches(frames_dir, batch_size=500, output_base='batches'):
    frames_dir = Path(frames_dir)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Get all frames
    frames = sorted(frames_dir.glob('frame_*.jpg'))
    if not frames:
        raise ValueError(f"No frames found in {frames_dir}")
    
    print(f"Total frames: {len(frames)}")
    print(f"Batch size: {batch_size}")
    print(f"Output: {output_base}/")
    
    num_batches = (len(frames) + batch_size - 1) // batch_size
    print(f"Creating {num_batches} batches...")
    
    batch_info = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(frames))
        batch_frames = frames[start_idx:end_idx]
        
        # Create batch directory
        batch_dir = output_base / f"batch_{batch_idx+1:03d}"
        batch_dir.mkdir(exist_ok=True)
        
        # Copy frames to batch
        print(f"\nBatch {batch_idx+1}/{num_batches}: {len(batch_frames)} frames")
        for frame in tqdm(batch_frames, desc=f"  Copying"):
            shutil.copy2(frame, batch_dir / frame.name)
        
        # Copy metadata and mask if they exist
        for aux_file in ['metadata.json', 'exclusion_mask.png', 'exclusion_polygon.json']:
            src = frames_dir / aux_file
            if src.exists():
                shutil.copy2(src, batch_dir / aux_file)
        
        batch_info.append({
            'batch_id': batch_idx + 1,
            'batch_name': f"batch_{batch_idx+1:03d}",
            'num_frames': len(batch_frames),
            'start_frame': start_idx,
            'end_frame': end_idx - 1
        })
    
    # Save batch manifest
    import json
    manifest = {
        'total_frames': len(frames),
        'batch_size': batch_size,
        'num_batches': num_batches,
        'batches': batch_info
    }
    
    manifest_path = output_base / 'batch_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Created {num_batches} batches in {output_base}/")
    print(f"✓ Manifest saved to {manifest_path}")
    
    return num_batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split frames into batches')
    parser.add_argument('frames_dir', type=str, help='Directory containing frames')
    parser.add_argument('--batch-size', type=int, default=500, 
                       help='Number of frames per batch (default: 500)')
    parser.add_argument('--output', type=str, default='batches',
                       help='Output base directory (default: batches)')
    
    args = parser.parse_args()
    
    split_into_batches(args.frames_dir, args.batch_size, args.output)
