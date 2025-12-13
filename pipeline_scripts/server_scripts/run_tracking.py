#!/usr/bin/env python3
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from tracking import PlayerTracker
from enhanced_tracking import EnhancedPlayerTracker


def run_tracking_on_detections(detections_file, output_file, batch_dirs=None, use_enhanced=True):
    """Apply tracking algorithm to combined detections."""
    import cv2
    
    # Load detections
    with open(detections_file, 'r') as f:
        data = json.load(f)
    
    detections_by_frame = data['detections']
    
    print(f"Total frames: {len(detections_by_frame)}")
    print(f"Total detections: {data['total_detections']}")
    
    # Build frame index -> file path mapping if batch dirs provided
    frame_to_path = {}
    if batch_dirs:
        print(f"Loading frame paths from {len(batch_dirs)} batches...")
        for batch_dir in batch_dirs:
            batch_path = Path(batch_dir)
            for frame_file in sorted(batch_path.glob('frame_*.jpg')):
                # Extract frame number from filename
                frame_num = int(frame_file.stem.split('_')[1])
                frame_to_path[frame_num] = str(frame_file)
        print(f"Found {len(frame_to_path)} frame files")
    
    # Initialize tracker
    if use_enhanced:
        tracker = EnhancedPlayerTracker()
        print("Using enhanced tracker with Kalman+Mahalanobis+Hungarian")
    else:
        tracker = PlayerTracker()
        print("Using standard ByteTrack")
    
    # Track
    all_tracks = []
    
    print("Running tracking...")
    for frame_data in detections_by_frame:
        frame_idx = frame_data['frame']
        dets = frame_data['detections']
        
        if not dets:
            all_tracks.append({
                'frame': frame_idx,
                'tracks': []
            })
            continue
        
        # Convert to arrays
        bboxes = np.array([d['bbox'] for d in dets])
        confidences = np.array([d['confidence'] for d in dets])
        class_ids = np.array([d['class_id'] for d in dets])
        
        # Convert to supervision format
        detections_sv = PlayerTracker.detections_to_supervision(bboxes, confidences, class_ids)
        
        # Load frame for appearance matching (if available)
        frame = None
        if use_enhanced and frame_idx in frame_to_path:
            frame = cv2.imread(frame_to_path[frame_idx])
        
        # Track
        if use_enhanced:
            tracked = tracker.update(detections_sv, frame_idx, frame)
        else:
            tracked = tracker.update(detections_sv)
        
        # Store tracks
        frame_tracks = {
            'frame': frame_idx,
            'tracks': []
        }
        
        if tracked.tracker_id is not None:
            for i, track_id in enumerate(tracked.tracker_id):
                frame_tracks['tracks'].append({
                    'track_id': int(track_id),
                    'bbox': tracked.xyxy[i].tolist(),
                    'confidence': float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
                })
        
        all_tracks.append(frame_tracks)
    
    # Save tracks
    output_data = {
        'total_frames': len(all_tracks),
        'total_tracks': len(set(t['track_id'] for ft in all_tracks for t in ft['tracks'])),
        'tracks': all_tracks
    }
    
    output_file = Path(output_file)
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
    
    print(f"\n✓ Tracking complete")
    print(f"  Total tracks: {output_data['total_tracks']}")
    print(f"  Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ByteTrack on combined detections')
    parser.add_argument('detections_file', type=str, help='Combined detections JSON')
    parser.add_argument('batch_dirs', nargs='*', help='Batch directories for loading frames')
    parser.add_argument('--output', type=str, default='output_local/tracks.json',
                       help='Output tracks file')
    parser.add_argument('--standard-tracking', action='store_true',
                       help='Use standard ByteTrack (default: enhanced)')
    
    args = parser.parse_args()
    
    batch_dirs = [Path(d) for d in args.batch_dirs] if args.batch_dirs else None
    
    run_tracking_on_detections(args.detections_file, args.output, 
                               batch_dirs=batch_dirs,
                               use_enhanced=not args.standard_tracking)
