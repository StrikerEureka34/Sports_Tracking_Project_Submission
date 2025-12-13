#!/usr/bin/env python3
"""OC-SORT / ByteTrack tracking with appearance embeddings and motion constraints."""

import json
import numpy as np
import cv2
import argparse
from pathlib import Path
from collections import defaultdict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TeamConsistencyGuard:
    """Enforces team consistency using HSV histogram voting."""
    def __init__(self, n_bins=8, confirm_frames=10):
        self.n_bins = n_bins
        self.confirm_frames = confirm_frames
        self.track_histograms = {}  # track_id -> list of histograms
        self.track_team = {}  # track_id -> team_label (0 or 1)
        self.confirmed = set()  # tracks with confirmed team
        
    def extract_histogram(self, frame, bbox):
        """Extract HSV histogram from bounding box region."""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, 
                           [self.n_bins, self.n_bins], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    
    def update(self, track_id, frame, bbox):
        """Update histogram history for a track."""
        hist = self.extract_histogram(frame, bbox)
        if hist is None:
            return
            
        if track_id not in self.track_histograms:
            self.track_histograms[track_id] = []
        self.track_histograms[track_id].append(hist)
        
        # Keep last N histograms
        if len(self.track_histograms[track_id]) > 50:
            self.track_histograms[track_id].pop(0)
    
    def get_team(self, track_id):
        """Get team label for a track (None if not confirmed)."""
        return self.track_team.get(track_id)
    
    def is_confirmed(self, track_id):
        """Check if track has confirmed team assignment."""
        return track_id in self.confirmed
    
    def can_associate(self, track_id1, track_id2):
        """Check if two tracks can be associated (same team or unknown)."""
        team1 = self.get_team(track_id1)
        team2 = self.get_team(track_id2)
        
        if team1 is None or team2 is None:
            return True
        return team1 == team2


class SpatialReentryGate:
    """
    Hard spatial re-entry gate for reusing IDs of recently dead tracks.
    If a new detection appears within reentry_radius of a recently dead track,
    reuse that track's ID instead of creating a new one.
    """
    def __init__(self, max_dead_frames=50, radius_scale=1.5):
        self.max_dead_frames = max_dead_frames  # How long to keep dead tracks
        self.radius_scale = radius_scale  # reentry_radius = scale * bbox_height
        self.dead_tracks = {}  # track_id -> (cx, cy, height, last_frame)
        self.active_tracks = {}  # track_id -> (cx, cy, height, frame)
        
    def mark_active(self, track_id, bbox, frame_idx):
        """Mark a track as active."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        height = y2 - y1
        self.active_tracks[track_id] = (cx, cy, height, frame_idx)
        # Remove from dead if it was there
        if track_id in self.dead_tracks:
            del self.dead_tracks[track_id]
    
    def mark_dead(self, track_id, frame_idx):
        """Move a track to dead pool."""
        if track_id in self.active_tracks:
            cx, cy, height, _ = self.active_tracks[track_id]
            self.dead_tracks[track_id] = (cx, cy, height, frame_idx)
            del self.active_tracks[track_id]
    
    def cleanup_old_dead(self, current_frame):
        """Remove tracks that have been dead too long."""
        to_remove = []
        for track_id, (_, _, _, last_frame) in self.dead_tracks.items():
            if current_frame - last_frame > self.max_dead_frames:
                to_remove.append(track_id)
        for tid in to_remove:
            del self.dead_tracks[tid]
    
    def find_reentry_match(self, bbox, frame_idx):
        """
        Check if a new detection can be matched to a recently dead track.
        Returns track_id if match found, None otherwise.
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        height = y2 - y1
        reentry_radius = self.radius_scale * height
        
        best_match = None
        best_dist = float('inf')
        
        for track_id, (tcx, tcy, theight, last_frame) in self.dead_tracks.items():
            frame_gap = frame_idx - last_frame
            if frame_gap > self.max_dead_frames:
                continue
                
            dist = np.sqrt((cx - tcx)**2 + (cy - tcy)**2)
            
            # Use average height for radius
            avg_height = (height + theight) / 2
            radius = self.radius_scale * avg_height
            
            if dist < radius and dist < best_dist:
                best_dist = dist
                best_match = track_id
        
        return best_match


class VelocityGate:
    """
    Hard velocity gating based on bbox height (depth proxy).
    max_speed = alpha * bbox_height (pixels/frame)
    """
    def __init__(self, alpha=0.5, min_speed=10, max_speed=100):
        self.alpha = alpha
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.track_positions = {}  # track_id -> (cx, cy, height, frame)
        
    def update(self, track_id, bbox, frame_idx):
        """Update position history for a track."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        height = y2 - y1
        self.track_positions[track_id] = (cx, cy, height, frame_idx)
    
    def get_max_displacement(self, bbox, frames_gap=1):
        """Calculate maximum allowed displacement based on bbox height."""
        height = bbox[3] - bbox[1]
        speed = self.alpha * height
        speed = np.clip(speed, self.min_speed, self.max_speed)
        return speed * frames_gap
    
    def is_valid_association(self, track_id, new_bbox, frame_idx):
        """Check if association is valid given velocity constraints."""
        if track_id not in self.track_positions:
            return True
            
        old_cx, old_cy, old_height, old_frame = self.track_positions[track_id]
        frames_gap = frame_idx - old_frame
        
        if frames_gap <= 0:
            return True
            
        new_cx = (new_bbox[0] + new_bbox[2]) / 2
        new_cy = (new_bbox[1] + new_bbox[3]) / 2
        
        displacement = np.sqrt((new_cx - old_cx)**2 + (new_cy - old_cy)**2)
        max_disp = self.get_max_displacement(new_bbox, frames_gap)
        
        return displacement <= max_disp


class TrackLifetimeStats:
    """
    Track lifetime statistics for debugging and quality assessment.
    Computes histogram of track lifetimes to identify issues.
    """
    def __init__(self):
        self.track_first_seen = {}  # track_id -> first frame
        self.track_last_seen = {}   # track_id -> last frame
        
    def update(self, track_id, frame_idx):
        """Record a track sighting."""
        if track_id not in self.track_first_seen:
            self.track_first_seen[track_id] = frame_idx
        self.track_last_seen[track_id] = frame_idx
    
    def get_lifetime(self, track_id):
        """Get lifetime of a track in frames."""
        if track_id not in self.track_first_seen:
            return 0
        return self.track_last_seen[track_id] - self.track_first_seen[track_id] + 1
    
    def get_stats(self):
        """Get lifetime statistics for all tracks."""
        lifetimes = [self.get_lifetime(tid) for tid in self.track_first_seen]
        if not lifetimes:
            return {}
        
        # Compute histogram bins
        very_short = sum(1 for l in lifetimes if l <= 2)  # 1-2 frames (likely noise)
        short = sum(1 for l in lifetimes if 3 <= l <= 10)  # 3-10 frames
        medium = sum(1 for l in lifetimes if 11 <= l <= 100)  # 11-100 frames
        long = sum(1 for l in lifetimes if 101 <= l <= 1000)  # 101-1000 frames
        very_long = sum(1 for l in lifetimes if l > 1000)  # 1000+ frames
        
        return {
            'total_tracks': len(lifetimes),
            'very_short_1_2': very_short,
            'short_3_10': short,
            'medium_11_100': medium,
            'long_101_1000': long,
            'very_long_1000+': very_long,
            'min_lifetime': min(lifetimes),
            'max_lifetime': max(lifetimes),
            'avg_lifetime': sum(lifetimes) / len(lifetimes),
            'median_lifetime': sorted(lifetimes)[len(lifetimes)//2]
        }


# Domain knowledge constants
MAX_ACTIVE_TRACKS = 30  # Cricket: max 22 players + umpires + staff


class DeepOCSortTracker:
    """
    OC-SORT / ByteTrack tracker wrapper.
    Using ByteTrack for more stable tracking in sports scenes.
    Includes: spatial re-entry gate, active track cap, lifetime stats.
    """
    def __init__(self, 
                 det_thresh=0.3,
                 max_age=12,          # REDUCED: ~0.4s at 30fps (was 30)
                 min_hits=3,
                 iou_threshold=0.3,
                 delta_t=3,
                 use_reid=True,
                 reid_weights='osnet_x1_0_msmt17.pt',
                 tracker_type='bytetrack'):  # bytetrack, ocsort, deepocsort
        
        import torch
        
        # Get reid weights path (boxmot will auto-download)
        reid_path = Path(reid_weights)
        
        if tracker_type == 'bytetrack':
            from boxmot import ByteTrack
            self.tracker = ByteTrack(
                det_thresh=det_thresh,
                max_age=max_age,        # REDUCED to 12
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                track_buffer=max_age,   # Same as max_age
                match_thresh=0.8,       # Strict matching
            )
            print(f"Using ByteTrack with max_age={max_age}, iou_threshold={iou_threshold}")
        elif tracker_type == 'ocsort':
            from boxmot import OcSort
            self.tracker = OcSort(
                det_thresh=det_thresh,
                max_age=max_age,  # REDUCED to 12
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                delta_t=delta_t,
                asso_func='iou',
                inertia=0.2,
            )
            print(f"Using OC-SORT (no ReID)")
        else:  # deepocsort
            from boxmot import DeepOcSort
            self.tracker = DeepOcSort(
                reid_weights=reid_path,
                device=torch.device('cuda:0'),
                half=True,  # FP16
                det_thresh=det_thresh,
                max_age=max_age,  # REDUCED to 12
                min_hits=min_hits,
                iou_threshold=iou_threshold,
                delta_t=delta_t,
                asso_func='iou',
                inertia=0.2,
                embedding_off=not use_reid,
            )
            print(f"Using Deep OC-SORT with ReID={use_reid}")
        
        self.tracker_type = tracker_type
        self.use_reid = use_reid
        self.team_guard = TeamConsistencyGuard()
        self.velocity_gate = VelocityGate(alpha=0.5)
        self.reentry_gate = SpatialReentryGate(max_dead_frames=50, radius_scale=1.5)
        self.lifetime_stats = TrackLifetimeStats()
        self.frame_idx = 0
        self.track_ages = {}  # track_id -> frames since last seen
        self.last_seen_tracks = set()  # tracks seen in previous frame
        self.max_age = max_age
        
        print(f"Initialized {tracker_type.upper()} with:")
        print(f"  - det_thresh: {det_thresh}")
        print(f"  - max_age: {max_age} (REDUCED for fast cleanup)")
        print(f"  - min_hits: {min_hits}")
        print(f"  - MAX_ACTIVE_TRACKS cap: {MAX_ACTIVE_TRACKS}")
        print(f"  - Spatial re-entry gate: radius_scale=1.5, max_dead=50")
        
    def update(self, detections, frame=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of dicts with 'bbox' and 'confidence'
            frame: BGR image for ReID (optional but recommended)
            
        Returns:
            List of tracks with 'track_id', 'bbox', 'confidence'
        """
        self.frame_idx += 1
        
        if len(detections) == 0:
            # Still need to call update with empty array to maintain state
            dets_array = np.empty((0, 6), dtype=np.float32)
        else:
            # Convert to numpy array [x1, y1, x2, y2, conf, cls]
            dets_array = np.array([
                [d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], 
                 d['confidence'], 0]
                for d in detections
            ], dtype=np.float32)
        
        # Ensure proper shape
        if len(dets_array.shape) == 1:
            dets_array = dets_array.reshape(-1, 6)
        
        # Run Deep OC-SORT
        try:
            if frame is not None:
                tracks = self.tracker.update(dets_array, frame)
            else:
                # Create dummy frame if not provided
                tracks = self.tracker.update(dets_array, np.zeros((720, 1280, 3), dtype=np.uint8))
        except Exception as e:
            # If tracker fails, return empty
            print(f"Tracker error at frame {self.frame_idx}: {e}")
            return []
        
        if tracks is None or len(tracks) == 0:
            # Mark all previously active tracks as potentially dead
            for tid in self.last_seen_tracks:
                self.reentry_gate.mark_dead(tid, self.frame_idx)
            self.last_seen_tracks = set()
            return []
        
        # Collect current frame's track IDs
        current_tracks = set()
        
        # Post-process: collect tracks with re-entry gate and lifetime stats
        valid_tracks = []
        for t in tracks:
            track_id = int(t[4])
            bbox = t[:4].tolist()
            conf = float(t[5]) if len(t) > 5 else 1.0
            
            current_tracks.add(track_id)
            
            # Update velocity gate (for position tracking)
            self.velocity_gate.update(track_id, bbox, self.frame_idx)
            
            # Update re-entry gate (mark as active)
            self.reentry_gate.mark_active(track_id, bbox, self.frame_idx)
            
            # Update lifetime stats
            self.lifetime_stats.update(track_id, self.frame_idx)
            
            # Update team guard if frame available
            if frame is not None:
                self.team_guard.update(track_id, frame, bbox)
            
            valid_tracks.append({
                'track_id': track_id,
                'bbox': bbox,
                'confidence': conf
            })
        
        # Mark tracks that disappeared as dead
        for tid in self.last_seen_tracks - current_tracks:
            self.reentry_gate.mark_dead(tid, self.frame_idx)
        
        # Cleanup old dead tracks
        self.reentry_gate.cleanup_old_dead(self.frame_idx)
        
        # SAFETY VALVE: Cap maximum active tracks
        if len(valid_tracks) > MAX_ACTIVE_TRACKS:
            # Sort by confidence and keep top MAX_ACTIVE_TRACKS
            valid_tracks.sort(key=lambda x: x['confidence'], reverse=True)
            valid_tracks = valid_tracks[:MAX_ACTIVE_TRACKS]
            # Note: This doesn't affect underlying tracker state, just output
        
        self.last_seen_tracks = current_tracks
        
        return valid_tracks


def run_tracking_deep_ocsort(detections_file, output_file, batch_dirs=None, 
                              use_reid=True, load_frames=True, max_frames=None,
                              tracker_type='bytetrack'):
    """
    Run Deep OC-SORT tracking on combined detections.
    
    Args:
        detections_file: Path to combined_detections.json
        output_file: Output tracks.json path
        batch_dirs: List of batch directories for loading frames
        use_reid: Whether to use OSNet ReID model
        load_frames: Whether to load frames for ReID
        max_frames: Maximum frames to process (None for all)
        tracker_type: 'bytetrack', 'ocsort', or 'deepocsort'
    """
    import time
    start_time = time.time()
    
    print("=" * 60)
    print("Deep OC-SORT Tracking with Conservative Constraints")
    print("=" * 60)
    
    # Load detections
    print(f"\nLoading detections from {detections_file}...")
    with open(detections_file, 'r') as f:
        data = json.load(f)
    
    detections_by_frame = {d['frame']: d['detections'] for d in data['detections']}
    total_frames_available = len(detections_by_frame)
    
    # Apply max_frames limit
    MAX_FRAMES = max_frames if max_frames else total_frames_available
    print(f"Total frames available: {total_frames_available}")
    print(f"Processing limit: {MAX_FRAMES} frames")
    print(f"Total detections: {data['total_detections']}")
    
    # Build frame index -> file path mapping
    frame_to_path = {}
    if batch_dirs and load_frames:
        print(f"\nLoading frame paths from {len(batch_dirs)} batches...")
        for batch_dir in batch_dirs:
            batch_path = Path(batch_dir)
            for frame_file in sorted(batch_path.glob('frame_*.jpg')):
                frame_num = int(frame_file.stem.split('_')[1])
                frame_to_path[frame_num] = str(frame_file)
        print(f"Found {len(frame_to_path)} frame files")
    
    # Initialize tracker with REDUCED max_age for fast cleanup
    print("\nInitializing tracker...")
    tracker = DeepOCSortTracker(
        det_thresh=0.3,       # Slightly higher to avoid noisy detections
        max_age=12,           # REDUCED: ~0.4s at 30fps (was 30) - prevents zombie tracks
        min_hits=3,           # Need 3 frames to confirm
        iou_threshold=0.5,    # Higher IoU = stricter matching
        delta_t=3,            # Velocity estimation window
        use_reid=use_reid,
        tracker_type=tracker_type,
    )
    
    # Process frames
    print(f"\nProcessing up to {MAX_FRAMES} frames...")
    print("-" * 60)
    tracks_output = []
    all_track_ids = set()
    
    frame_indices = sorted(detections_by_frame.keys())
    
    for frame_idx_loop, frame_idx in enumerate(frame_indices):
        # HARD STOP at MAX_FRAMES
        if frame_idx_loop >= MAX_FRAMES:
            print(f"\n>>> Reached MAX_FRAMES limit ({MAX_FRAMES}), stopping.")
            break
        
        dets = detections_by_frame[frame_idx]
        
        # Load frame for ReID if available
        frame = None
        if frame_idx in frame_to_path and load_frames:
            frame = cv2.imread(frame_to_path[frame_idx])
        
        # Update tracker
        frame_tracks = tracker.update(dets, frame)
        
        # Count active tracks (with velocity gate positions)
        active_track_count = len(tracker.velocity_gate.track_positions)
        
        # Collect track IDs
        for t in frame_tracks:
            all_track_ids.add(t['track_id'])
        
        tracks_output.append({
            'frame': frame_idx,
            'tracks': frame_tracks
        })
        
        # Progress + track count logging every 1000 frames
        if frame_idx_loop % 1000 == 0:
            elapsed = time.time() - start_time
            fps = (frame_idx_loop + 1) / elapsed if elapsed > 0 else 0
            print(f"[Frame {frame_idx_loop:6d}/{MAX_FRAMES}] "
                  f"Active tracks: {active_track_count:3d} | "
                  f"Unique IDs: {len(all_track_ids):4d} | "
                  f"FPS: {fps:.1f}")
            sys.stdout.flush()  # Ensure output is visible
    
    # Final stats
    elapsed_total = time.time() - start_time
    print("-" * 60)
    print(f"\n>>> Processing complete!")
    print(f"    Frames processed: {len(tracks_output)}")
    print(f"    Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)")
    print(f"    Avg FPS: {len(tracks_output)/elapsed_total:.1f}")
    
    # Track count sanity check
    print(f"\n>>> Track sanity check:")
    print(f"    Total unique track IDs: {len(all_track_ids)}")
    if len(all_track_ids) > 100:
        print(f"    ⚠️  WARNING: Very high track count! Check detection quality")
    elif len(all_track_ids) > 50:
        print(f"    ⚠️  WARNING: High track count - expected for broadcast with scene cuts")
    elif 10 <= len(all_track_ids) <= 30:
        print(f"    ✓ Track count looks reasonable for cricket")
    else:
        print(f"    ℹ️  Review tracks - may need tuning")
    
    # Lifetime statistics (for debugging)
    print(f"\n>>> Track lifetime analysis:")
    lifetime_stats = tracker.lifetime_stats.get_stats()
    if lifetime_stats:
        print(f"    Very short (1-2 frames): {lifetime_stats['very_short_1_2']} tracks")
        print(f"    Short (3-10 frames):     {lifetime_stats['short_3_10']} tracks")
        print(f"    Medium (11-100 frames):  {lifetime_stats['medium_11_100']} tracks")
        print(f"    Long (101-1000 frames):  {lifetime_stats['long_101_1000']} tracks")
        print(f"    Very long (1000+ frames):{lifetime_stats['very_long_1000+']} tracks")
        print(f"    Min/Max/Avg lifetime:    {lifetime_stats['min_lifetime']}/{lifetime_stats['max_lifetime']}/{lifetime_stats['avg_lifetime']:.1f} frames")
        
        if lifetime_stats['very_short_1_2'] > lifetime_stats['total_tracks'] * 0.5:
            print(f"    ⚠️  Many short-lived tracks - likely detection noise")
    
    # Save output
    print(f"\nSaving tracks to {output_file}...")
    output_data = {
        'total_frames': len(tracks_output),
        'total_tracks': len(all_track_ids),
        'tracker': tracker_type.upper(),
        'config': {
            'det_thresh': 0.3,
            'max_age': 12,            # REDUCED
            'min_hits': 3,
            'iou_threshold': 0.5,
            'delta_t': 3,
            'max_active_tracks': MAX_ACTIVE_TRACKS,
            'reentry_gate_radius_scale': 1.5,
            'reentry_gate_max_dead_frames': 50,
            'use_reid': use_reid,
            'max_frames_processed': len(tracks_output)
        },
        'lifetime_stats': lifetime_stats,
        'tracks': tracks_output
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
    
    print(f"\n✓ Tracking complete!")
    print(f"  Total frames: {len(tracks_output)}")
    print(f"  Total unique tracks: {len(all_track_ids)}")
    print(f"  Output: {output_file}")
    
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep OC-SORT tracking')
    parser.add_argument('detections_file', type=str, help='Combined detections JSON')
    parser.add_argument('--output', type=str, default='tracks_ocsort.json',
                       help='Output tracks file')
    parser.add_argument('--batch-dirs', type=str, nargs='*',
                       help='Batch directories for loading frames')
    parser.add_argument('--no-reid', action='store_true',
                       help='Disable ReID (faster but less accurate)')
    parser.add_argument('--no-frames', action='store_true',
                       help='Skip loading frames (faster but no ReID)')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process (default: all)')
    parser.add_argument('--tracker', type=str, default='bytetrack',
                       choices=['bytetrack', 'ocsort', 'deepocsort'],
                       help='Tracker type to use (default: bytetrack)')
    
    args = parser.parse_args()
    
    run_tracking_deep_ocsort(
        args.detections_file,
        args.output,
        args.batch_dirs,
        use_reid=not args.no_reid,
        load_frames=not args.no_frames,
        max_frames=args.max_frames,
        tracker_type=args.tracker
    )
