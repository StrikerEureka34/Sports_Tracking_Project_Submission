#!/usr/bin/env python3
"""Bird's eye view analytics using homography projection."""

import sys
import json
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "project_2" / "data_video"))

from projection import TopViewProjector
from analytics import TrajectoryAnalytics
from visualization import TopViewVisualizer

# Paths (use project_2 directory)
PROJECT_DIR = Path("/data/ayushkumar/project_2")
TRACKS_FILE = PROJECT_DIR / "output_local" / "tracks_full.json"
OUTPUT_DIR = PROJECT_DIR / "analytics_output_v2"
SAMPLE_FRAME_DIR = PROJECT_DIR / "incoming_batches"  # For getting frame dimensions

# Cricket field parameters (for homography)
CRICKET_FIELD_WIDTH = 137.2  # meters (boundary to boundary typical)
CRICKET_FIELD_HEIGHT = 137.2  # meters (circular field approximation)
BIRDS_EYE_WIDTH = 1370  # pixels (10 pixels per meter)
BIRDS_EYE_HEIGHT = 1370  # pixels

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def get_frame_dimensions():
    """Get dimensions from first available frame"""
    batch_dirs = sorted(SAMPLE_FRAME_DIR.glob("batch_*"))
    if not batch_dirs:
        return 1920, 1080  # Default HD
    
    first_batch = batch_dirs[0]
    frame_files = sorted(first_batch.glob("*.jpg"))
    if not frame_files:
        return 1920, 1080
    
    sample_frame = cv2.imread(str(frame_files[0]))
    if sample_frame is None:
        return 1920, 1080
    
    h, w = sample_frame.shape[:2]
    print(f"Frame dimensions: {w}x{h}")
    return w, h

def setup_cricket_homography(frame_width: int, frame_height: int) -> TopViewProjector:
    """
    Setup homography for cricket field projection
    Using typical broadcast camera view to top-down mapping
    """
    # Define source points (camera view coordinates)
    # These are typical points for a cricket broadcast:
    # - Bottom center (near boundary)
    # - Top corners (far boundary)
    # - Side boundaries
    source_points = np.array([
        [frame_width * 0.1, frame_height * 0.95],   # Bottom left boundary
        [frame_width * 0.9, frame_height * 0.95],   # Bottom right boundary
        [frame_width * 0.95, frame_height * 0.2],   # Top right boundary
        [frame_width * 0.05, frame_height * 0.2],   # Top left boundary
    ], dtype=np.float32)
    
    # Destination points (bird's eye view)
    margin = 50
    dest_points = np.array([
        [margin, BIRDS_EYE_HEIGHT - margin],           # Bottom left
        [BIRDS_EYE_WIDTH - margin, BIRDS_EYE_HEIGHT - margin],  # Bottom right
        [BIRDS_EYE_WIDTH - margin, margin],            # Top right
        [margin, margin],                               # Top left
    ], dtype=np.float32)
    
    projector = TopViewProjector(
        source_points=source_points,
        dest_points=dest_points,
        field_width=CRICKET_FIELD_WIDTH,
        field_height=CRICKET_FIELD_HEIGHT,
        output_width=BIRDS_EYE_WIDTH,
        output_height=BIRDS_EYE_HEIGHT
    )
    
    return projector

def load_tracks_data():
    """Load tracks from JSON"""
    print(f"Loading tracks from {TRACKS_FILE}...")
    with open(TRACKS_FILE, 'r') as f:
        tracks_data = json.load(f)
    
    print(f"Loaded {len(tracks_data['tracks'])} tracks across {tracks_data['total_frames']} frames")
    return tracks_data

def convert_tracks_to_trajectories(tracks_data: dict, projector: TopViewProjector) -> Tuple[TrajectoryAnalytics, Dict]:
    """
    Convert tracks to trajectory analytics with projection
    Uses optimized approach: collect all points first, project in batch
    """
    analytics = TrajectoryAnalytics(
        field_width=BIRDS_EYE_WIDTH,
        field_height=BIRDS_EYE_HEIGHT,
        heatmap_bins=100,  # High resolution for bird's eye view
        smoothing_window=5,
        min_track_length=10
    )
    
    print("Projecting positions to bird's eye view...")
    projected_tracks = defaultdict(list)
    
    # Process frame by frame (data is already organized by frame)
    for frame_data in tracks_data['tracks']:
        frame_idx = frame_data['frame']
        frame_tracks = frame_data['tracks']
        
        if len(frame_tracks) == 0:
            continue
        
        # Collect all positions for this frame
        positions = []
        track_ids = []
        confidences = []
        
        for track in frame_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            confidence = track['confidence']
            
            # Calculate feet position (bottom center of bbox)
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = y1 + 0.9 * (y2 - y1)  # 90% down (same as heatmap generation)
            
            positions.append([cx, cy])
            track_ids.append(track_id)
            confidences.append(confidence)
        
        positions = np.array(positions)
        track_ids = np.array(track_ids)
        confidences = np.array(confidences)
        
        # Project all positions at once (efficient)
        if len(positions) > 0:
            projected_positions = projector.project_points(positions)
            
            # Filter out-of-bounds positions
            projected_positions, valid_indices = projector.filter_out_of_bounds(
                projected_positions, 
                np.arange(len(projected_positions))
            )
            
            if valid_indices is not None and len(valid_indices) > 0:
                track_ids = track_ids[valid_indices]
                confidences = confidences[valid_indices]
                
                # Add to analytics
                analytics.add_trajectory_data(
                    frame_idx,
                    track_ids,
                    projected_positions,
                    confidences
                )
                
                # Store for trajectory visualization
                for i, track_id in enumerate(track_ids):
                    projected_tracks[int(track_id)].append(projected_positions[i])
        
        if frame_idx % 1000 == 0:
            print(f"  Processed frame {frame_idx}/{tracks_data['total_frames']}")
    
    # Convert projected tracks to numpy arrays
    projected_trajectories = {}
    for track_id, positions in projected_tracks.items():
        projected_trajectories[track_id] = np.array(positions)
    
    print(f"Projected {len(projected_trajectories)} valid trajectories")
    return analytics, projected_trajectories

def generate_birds_eye_heatmap(analytics: TrajectoryAnalytics, visualizer: TopViewVisualizer):
    """Generate bird's eye view heatmap (optimized)"""
    print("Generating bird's eye view heatmap...")
    
    # Use fast heatmap computation from analytics
    heatmap = analytics.compute_heatmap(gaussian_sigma=3.0)
    
    # Render with cricket field overlay
    field_viz = visualizer.draw_heatmap(heatmap, colormap='hot', alpha=0.7)
    
    # Save
    output_path = OUTPUT_DIR / "birds_eye_heatmap.png"
    cv2.imwrite(str(output_path), field_viz)
    print(f"Saved: {output_path}")
    
    return heatmap

def generate_trajectory_visualization(trajectories: Dict, visualizer: TopViewVisualizer):
    """Generate trajectory visualization for top tracks"""
    print("Generating trajectory visualization...")
    
    # Get top 20 longest tracks
    track_lengths = {tid: len(traj) for tid, traj in trajectories.items()}
    top_tracks = sorted(track_lengths.items(), key=lambda x: x[1], reverse=True)[:20]
    top_track_ids = [tid for tid, _ in top_tracks]
    
    # Filter trajectories
    top_trajectories = {tid: trajectories[tid] for tid in top_track_ids}
    
    # Visualize
    field_viz = visualizer.draw_trajectories(top_trajectories, thickness=2)
    
    # Save
    output_path = OUTPUT_DIR / "birds_eye_trajectories_top20.png"
    cv2.imwrite(str(output_path), field_viz)
    print(f"Saved: {output_path} (top 20 tracks)")
    
    return top_trajectories

def generate_speed_analytics(analytics: TrajectoryAnalytics, projector: TopViewProjector):
    """Generate speed and movement analytics"""
    print("Generating speed analytics...")
    
    # Get all statistics
    meters_per_pixel = CRICKET_FIELD_WIDTH / BIRDS_EYE_WIDTH
    stats_df = analytics.get_all_statistics(fps=30.0, meters_per_pixel=meters_per_pixel)
    
    if len(stats_df) == 0:
        print("No valid tracks for statistics")
        return
    
    # Save statistics
    stats_path = OUTPUT_DIR / "trajectory_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")
    
    # Generate visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cricket Player Movement Analytics', fontsize=16, fontweight='bold')
    
    # 1. Speed distribution
    ax = axes[0, 0]
    speeds = stats_df['avg_speed'].values
    ax.hist(speeds, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Average Speed (m/s)', fontsize=11)
    ax.set_ylabel('Number of Tracks', fontsize=11)
    ax.set_title('Speed Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Distance traveled
    ax = axes[0, 1]
    distances = stats_df['total_distance'].values
    ax.hist(distances, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Total Distance (meters)', fontsize=11)
    ax.set_ylabel('Number of Tracks', fontsize=11)
    ax.set_title('Distance Traveled Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Duration distribution
    ax = axes[1, 0]
    durations = stats_df['duration_seconds'].values
    ax.hist(durations, bins=30, color='mediumseagreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Track Duration (seconds)', fontsize=11)
    ax.set_ylabel('Number of Tracks', fontsize=11)
    ax.set_title('Track Duration Distribution', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Speed vs Distance scatter
    ax = axes[1, 1]
    ax.scatter(stats_df['avg_speed'], stats_df['total_distance'], 
               alpha=0.5, c=stats_df['duration_seconds'], cmap='viridis', s=50)
    ax.set_xlabel('Average Speed (m/s)', fontsize=11)
    ax.set_ylabel('Total Distance (meters)', fontsize=11)
    ax.set_title('Speed vs Distance', fontweight='bold')
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Duration (s)', fontsize=10)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "movement_analytics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    # Print summary statistics
    print("\n=== Movement Analytics Summary ===")
    print(f"Total valid tracks: {len(stats_df)}")
    print(f"Average speed: {speeds.mean():.2f} ± {speeds.std():.2f} m/s")
    print(f"Max speed: {speeds.max():.2f} m/s")
    print(f"Average distance: {distances.mean():.2f} ± {distances.std():.2f} m")
    print(f"Average duration: {durations.mean():.2f} ± {durations.std():.2f} s")

def generate_zone_heatmap(analytics: TrajectoryAnalytics, visualizer: TopViewVisualizer):
    """Generate heatmap showing field zones with activity levels"""
    print("Generating zone-based heatmap...")
    
    # Compute heatmap
    heatmap = analytics.compute_heatmap(gaussian_sigma=2.0)
    
    # Create figure with cricket field zones
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Draw field template
    field_template = visualizer.field_template
    ax.imshow(cv2.cvtColor(field_template, cv2.COLOR_BGR2RGB), extent=[0, BIRDS_EYE_WIDTH, BIRDS_EYE_HEIGHT, 0])
    
    # Overlay heatmap
    heatmap_norm = heatmap / heatmap.max() if heatmap.max() > 0 else heatmap
    im = ax.imshow(heatmap_norm, cmap='hot', alpha=0.6, extent=[0, BIRDS_EYE_WIDTH, BIRDS_EYE_HEIGHT, 0])
    
    # Add zone labels
    zones = [
        ("Deep Field", BIRDS_EYE_WIDTH // 2, BIRDS_EYE_HEIGHT // 6),
        ("Infield", BIRDS_EYE_WIDTH // 2, BIRDS_EYE_HEIGHT // 2),
        ("Close Field", BIRDS_EYE_WIDTH // 2, BIRDS_EYE_HEIGHT * 5 // 6),
    ]
    
    for zone_name, x, y in zones:
        ax.text(x, y, zone_name, fontsize=14, fontweight='bold', 
                color='white', ha='center', 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title('Cricket Field Activity Zones (Bird\'s Eye View)', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activity Density', fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "birds_eye_zones.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")

def main():
    print("=" * 60)
    print("Enhanced Analytics with Bird's Eye View")
    print("=" * 60)
    
    # Get frame dimensions
    frame_width, frame_height = get_frame_dimensions()
    
    # Setup homography projection
    print("\nSetting up homography projection...")
    projector = setup_cricket_homography(frame_width, frame_height)
    
    # Setup visualizer
    visualizer = TopViewVisualizer(
        field_width=BIRDS_EYE_WIDTH,
        field_height=BIRDS_EYE_HEIGHT,
        trajectory_thickness=2,
        heatmap_alpha=0.7
    )
    
    # Load tracks
    tracks_data = load_tracks_data()
    
    # Convert to bird's eye view trajectories
    print("\nConverting to bird's eye view...")
    analytics, projected_trajectories = convert_tracks_to_trajectories(tracks_data, projector)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Bird's eye heatmap
    generate_birds_eye_heatmap(analytics, visualizer)
    
    # 2. Trajectory visualization
    generate_trajectory_visualization(projected_trajectories, visualizer)
    
    # 3. Speed and movement analytics
    generate_speed_analytics(analytics, projector)
    
    # 4. Zone heatmap
    generate_zone_heatmap(analytics, visualizer)
    
    print("\n" + "=" * 60)
    print(f"✓ All enhanced analytics saved to {OUTPUT_DIR}")
    print("=" * 60)
    print("\nGenerated files:")
    for file in sorted(OUTPUT_DIR.glob("*")):
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    main()
