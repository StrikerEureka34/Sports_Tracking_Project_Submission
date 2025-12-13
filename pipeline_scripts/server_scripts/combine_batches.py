#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def combine_batch_detections(detections_dir, output_file):
    """Merge all batch detection files into one sorted file."""
    detections_dir = Path(detections_dir)
    
    # Load all batch detections
    batch_files = sorted(detections_dir.glob('batch_*_detections.json'))
    if not batch_files:
        raise ValueError(f"No detection files found in {detections_dir}")
    
    print(f"Found {len(batch_files)} batch detection files")
    
    all_detections = []
    total_frames = 0
    total_detections = 0
    
    for batch_file in batch_files:
        print(f"Loading {batch_file.name}...")
        with open(batch_file, 'r') as f:
            batch_data = json.load(f)
        
        all_detections.extend(batch_data['detections'])
        total_frames += batch_data['num_frames']
        total_detections += batch_data['num_detections']
    
    # Sort by frame number
    all_detections.sort(key=lambda x: x['frame'])
    
    # Combine into single structure
    combined = {
        'total_frames': total_frames,
        'total_detections': total_detections,
        'num_batches': len(batch_files),
        'detections': all_detections
    }
    
    # Save combined file
    output_file = Path(output_file)
    with open(output_file, 'w') as f:
        json.dump(combined, f)
    
    print(f"\n✓ Combined {len(batch_files)} batches")
    print(f"  Total frames: {total_frames}")
    print(f"  Total detections: {total_detections}")
    print(f"  Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine batch detections')
    parser.add_argument('detections_dir', type=str, 
                       help='Directory containing batch detection JSON files')
    parser.add_argument('--output', type=str, 
                       default='output_local/combined_detections.json',
                       help='Output combined detections file')
    
    args = parser.parse_args()
    
    combine_batch_detections(args.detections_dir, args.output)
