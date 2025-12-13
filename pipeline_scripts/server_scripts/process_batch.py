#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from detection import PlayerDetector


def load_mask(batch_dir):
    """Load exclusion mask if present."""
    mask_path = Path(batch_dir) / 'exclusion_mask.png'
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask
    return None


def filter_detections_by_mask(bboxes, confidences, class_ids, mask):
    """
    Filter detections: remove those inside exclusion zone.
    
    Args:
        bboxes: (N, 4) bounding boxes [x1, y1, x2, y2]
        confidences: (N,) confidence scores
        class_ids: (N,) class IDs
        mask: Binary mask (255 = exclude, 0 = keep)
        
    Returns:
        Filtered bboxes, confidences, class_ids
    """
    if mask is None or len(bboxes) == 0:
        return bboxes, confidences, class_ids
    
    keep_indices = []
    
    for i, bbox in enumerate(bboxes):
        # Check bottom center of bbox
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int(bbox[3])  # Bottom of bbox
        
        # Keep if outside exclusion zone (mask value = 0)
        if 0 <= cy < mask.shape[0] and 0 <= cx < mask.shape[1]:
            if mask[cy, cx] == 0:  # Not in exclusion zone
                keep_indices.append(i)
        else:
            # Outside frame bounds, keep it
            keep_indices.append(i)
    
    if not keep_indices:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)
    
    return bboxes[keep_indices], confidences[keep_indices], class_ids[keep_indices]


def process_batch(batch_dir, output_dir, detector=None, batch_size=64):
    """
    Run YOLO inference on all frames in batch with mask filtering.
    
    Args:
        batch_dir: Directory containing batch frames
        output_dir: Output directory for detections
        detector: PlayerDetector instance (or create new one)
        batch_size: Batch size for YOLO inference
        
    Returns:
        detections_file: Path to saved detections JSON
    """
    batch_dir = Path(batch_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load mask
    mask = load_mask(batch_dir)
    if mask is not None:
        print(f"✓ Loaded exclusion mask: {mask.shape}")
    else:
        print("⚠ No exclusion mask found")
    
    # Initialize detector if not provided
    if detector is None:
        print("Initializing YOLOv8 detector...")
        detector = PlayerDetector()
    
    # Get all frames
    frame_paths = sorted(batch_dir.glob('frame_*.jpg'))
    if not frame_paths:
        raise ValueError(f"No frames found in {batch_dir}")
    
    print(f"Processing {len(frame_paths)} frames from {batch_dir.name} in batches of {batch_size}...")
    
    all_detections = []
    
    # Process in batches for efficiency
    for batch_start in tqdm(range(0, len(frame_paths), batch_size), desc="YOLO inference"):
        batch_end = min(batch_start + batch_size, len(frame_paths))
        batch_frame_paths = frame_paths[batch_start:batch_end]
        
        # Load batch of frames
        frames_batch = []
        frame_indices = []
        valid_indices = []
        
        for idx, frame_path in enumerate(batch_frame_paths):
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                frames_batch.append(frame)
                # Extract frame number from filename
                frame_name = frame_path.stem  # e.g., "frame_000123"
                frame_idx = int(frame_name.split('_')[1])
                frame_indices.append(frame_idx)
                valid_indices.append(idx)
        
        if not frames_batch:
            continue
        
        # Run YOLO on batch (much faster with larger batch size!)
        batch_detections = detector.detect_batch(frames_batch, batch_size=len(frames_batch))
        
        # Process each frame's detections
        for i, (bboxes, confidences, class_ids) in enumerate(batch_detections):
            frame_idx = frame_indices[i]
            
            # Filter by mask
            if mask is not None:
                bboxes, confidences, class_ids = filter_detections_by_mask(
                    bboxes, confidences, class_ids, mask
                )
            
            # Store detections
            frame_detections = {
                'frame': frame_idx,
                'frame_name': f'frame_{frame_idx:06d}',
                'batch': batch_dir.name,
                'detections': []
            }
            
            for bbox, conf, cls_id in zip(bboxes, confidences, class_ids):
                frame_detections['detections'].append({
                    'bbox': bbox.tolist(),
                    'confidence': float(conf),
                    'class_id': int(cls_id)
                })
            
            all_detections.append(frame_detections)
    
    # Save detections
    batch_name = batch_dir.name
    detections_file = output_dir / f"{batch_name}_detections.json"
    
    with open(detections_file, 'w') as f:
        json.dump({
            'batch': batch_name,
            'num_frames': len(frame_paths),
            'num_detections': sum(len(d['detections']) for d in all_detections),
            'detections': all_detections
        }, f)
    
    print(f"✓ Saved detections to {detections_file}")
    
    return detections_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process batch with YOLO + mask filtering')
    parser.add_argument('batch_dir', type=str, help='Batch directory')
    parser.add_argument('--output', type=str, default='output_local/detections',
                       help='Output directory for detections')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for YOLO inference (default: 64)')
    
    args = parser.parse_args()
    
    process_batch(args.batch_dir, args.output, batch_size=args.batch_size)
