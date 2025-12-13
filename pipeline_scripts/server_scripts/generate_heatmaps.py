#!/usr/bin/env python3
import json
import numpy as np
import cv2
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def generate_heatmaps(tracks_file, output_dir, width=1280, height=720):
    """Generate heatmaps and analytics from tracking data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HEATMAP & ANALYTICS GENERATOR")
    print("=" * 60)
    
    # Load tracks
    print(f"\nLoading tracks from {tracks_file}...")
    with open(tracks_file, 'r') as f:
        data = json.load(f)
    
    tracks_by_frame = {t['frame']: t['tracks'] for t in data['tracks']}
    total_frames = data['total_frames']
    total_tracks = data['total_tracks']
    
    print(f"Total frames: {total_frames}")
    print(f"Total unique tracks: {total_tracks}")
    
    # Initialize heatmap accumulators
    heatmap_all = np.zeros((height, width), dtype=np.float32)
    heatmap_by_track = defaultdict(lambda: np.zeros((height, width), dtype=np.float32))
    
    # Track statistics
    track_positions = defaultdict(list)  # track_id -> list of (x, y, frame)
    track_velocities = defaultdict(list)  # track_id -> list of velocities
    track_frame_counts = defaultdict(int)  # track_id -> frame count
    
    print("\nProcessing frames for heatmaps (FAST: increment only, blur later)...")
    for frame_idx, tracks in enumerate(tracks_by_frame.values()):
        if frame_idx % 1000 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames", end='\r', flush=True)
        
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            # Get feet position (stable projection, not just bbox bottom)
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int(y1 + 0.9 * (y2 - y1))  # 90% down = feet, stable on zoom
            
            # Clip to frame bounds
            cx = np.clip(cx, 0, width - 1)
            cy = np.clip(cy, 0, height - 1)
            
            # FAST: Just increment pixel (blur applied once later)
            heatmap_all[cy, cx] += 1
            heatmap_by_track[track_id][cy, cx] += 1
            track_frame_counts[track_id] += 1
            
            # Store position for velocity calculation
            track_positions[track_id].append((cx, cy, frame_idx))
    
    print(f"\n  Processed {total_frames}/{total_frames} frames")
    
    # Apply Gaussian blur ONCE (much faster than per-detection)
    print("\nApplying Gaussian blur to heatmaps...")
    heatmap_all = cv2.GaussianBlur(heatmap_all, (0, 0), sigmaX=15, sigmaY=15)
    
    # Normalize by time exposure (frames) - shows tendency, not camera bias
    print("Normalizing by frame exposure...")
    heatmap_all = heatmap_all / total_frames if total_frames > 0 else heatmap_all
    
    # Blur and normalize per-track heatmaps
    print("Processing per-track heatmaps...")
    for track_id in heatmap_by_track:
        heatmap_by_track[track_id] = cv2.GaussianBlur(
            heatmap_by_track[track_id], (0, 0), sigmaX=15, sigmaY=15
        )
        # Normalize by track's frame count
        frame_count = track_frame_counts[track_id]
        if frame_count > 0:
            heatmap_by_track[track_id] = heatmap_by_track[track_id] / frame_count
    
    # Calculate velocities
    print("\nCalculating track velocities...")
    for track_id, positions in track_positions.items():
        if len(positions) < 2:
            continue
        
        for i in range(1, len(positions)):
            x1, y1, f1 = positions[i-1]
            x2, y2, f2 = positions[i]
            
            frame_gap = f2 - f1
            if frame_gap > 0:
                dx = x2 - x1
                dy = y2 - y1
                dist = np.sqrt(dx**2 + dy**2)
                velocity = dist / frame_gap  # pixels per frame
                track_velocities[track_id].append(velocity)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Overall heatmap with cricket field overlay
    print("  1. Overall position heatmap (cricket field overlay)...")
    
    # Create cricket field background (simplified)
    cricket_field = np.ones((height, width, 3), dtype=np.uint8) * 34  # Dark green
    
    # Draw pitch (center rectangle)
    pitch_width = int(width * 0.15)
    pitch_height = int(height * 0.6)
    pitch_x = (width - pitch_width) // 2
    pitch_y = (height - pitch_height) // 2
    cv2.rectangle(cricket_field, (pitch_x, pitch_y), 
                  (pitch_x + pitch_width, pitch_y + pitch_height), 
                  (139, 90, 43), -1)  # Brown pitch
    
    # Draw boundary circle
    center = (width // 2, height // 2)
    boundary_radius = int(min(width, height) * 0.45)
    cv2.circle(cricket_field, center, boundary_radius, (200, 200, 200), 2)
    
    # Draw 30-yard circles
    cv2.circle(cricket_field, center, boundary_radius // 2, (150, 150, 150), 1)
    
    # Convert to matplotlib figure
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Show cricket field
    ax.imshow(cv2.cvtColor(cricket_field, cv2.COLOR_BGR2RGB), alpha=0.3)
    
    # Overlay heatmap
    heatmap_normalized = heatmap_all / heatmap_all.max() if heatmap_all.max() > 0 else heatmap_all
    im = ax.imshow(heatmap_normalized, cmap='hot', interpolation='bilinear', 
                   aspect='auto', alpha=0.7, vmin=0, vmax=heatmap_normalized.max())
    
    plt.colorbar(im, ax=ax, label='Player Density (normalized by time)')
    ax.set_title(f'Player Position Heatmap - Cricket Field\n{total_tracks} players, {total_frames} frames', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_overall.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Top 10 most active tracks
    print("  2. Top 10 active player heatmaps...")
    track_activity = {tid: np.sum(hmap) for tid, hmap in heatmap_by_track.items()}
    top_tracks = sorted(track_activity.items(), key=lambda x: x[1], reverse=True)[:10]
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (track_id, activity) in enumerate(top_tracks):
        hmap = heatmap_by_track[track_id]
        hmap_norm = hmap / hmap.max() if hmap.max() > 0 else hmap
        
        axes[idx].imshow(hmap_norm, cmap='hot', interpolation='bilinear', aspect='auto')
        axes[idx].set_title(f'Track ID {track_id}\nActivity: {activity:.0f}')
        axes[idx].axis('off')
    
    plt.suptitle('Top 10 Most Active Players', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'heatmap_top10.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Velocity distribution
    print("  3. Velocity distribution...")
    all_velocities = []
    for vels in track_velocities.values():
        all_velocities.extend(vels)
    
    if all_velocities:
        plt.figure(figsize=(12, 6))
        plt.hist(all_velocities, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Velocity (pixels/frame)')
        plt.ylabel('Frequency')
        plt.title(f'Player Velocity Distribution\nMean: {np.mean(all_velocities):.2f} px/frame, Max: {np.max(all_velocities):.2f} px/frame')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'velocity_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 4. Track lifetime distribution
    print("  4. Track lifetime analysis...")
    if 'lifetime_stats' in data:
        stats = data['lifetime_stats']
        
        categories = ['Very Short\n(1-2 frames)', 'Short\n(3-10)', 'Medium\n(11-100)', 
                     'Long\n(101-1000)', 'Very Long\n(1000+)']
        counts = [
            stats.get('very_short_1_2', 0),
            stats.get('short_3_10', 0),
            stats.get('medium_11_100', 0),
            stats.get('long_101_1000', 0),
            stats.get('very_long_1000+', 0)
        ]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(categories, counts, edgecolor='black', alpha=0.7, color=plt.cm.viridis(np.linspace(0.2, 0.8, 5)))
        plt.xlabel('Track Lifetime Category')
        plt.ylabel('Number of Tracks')
        plt.title(f'Track Lifetime Distribution\nTotal Tracks: {stats.get("total_tracks", 0)}, Avg Lifetime: {stats.get("avg_lifetime", 0):.1f} frames')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'track_lifetime_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # 5. Movement flow vectors (using 10% sampling for clarity)
    print("  5. Movement flow visualization...")
    plt.figure(figsize=(16, 9))
    
    # Sample positions for flow vectors
    sample_rate = max(1, len(track_positions) // 50)  # Sample ~50 tracks
    sampled_tracks = list(track_positions.items())[::sample_rate]
    
    for track_id, positions in sampled_tracks:
        if len(positions) < 10:
            continue
        
        # Sample every 10th position for this track
        sampled_pos = positions[::10]
        
        xs = [p[0] for p in sampled_pos]
        ys = [p[1] for p in sampled_pos]
        
        # Draw trajectory
        plt.plot(xs, ys, alpha=0.5, linewidth=1)
        
        # Draw arrows for direction
        for i in range(0, len(sampled_pos) - 1, 5):
            x1, y1, _ = sampled_pos[i]
            x2, y2, _ = sampled_pos[i + 1]
            plt.arrow(x1, y1, x2 - x1, y2 - y1, 
                     head_width=10, head_length=10, fc='blue', ec='blue', alpha=0.3)
    
    plt.xlim(0, width)
    plt.ylim(height, 0)  # Invert Y axis
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Player Movement Flow (Sampled Trajectories)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'movement_flow.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate statistics report
    print("\n  6. Generating statistics report...")
    stats_report = {
        'total_frames': total_frames,
        'total_tracks': total_tracks,
        'avg_tracks_per_frame': total_frames / total_tracks if total_tracks > 0 else 0,
        'avg_velocity': np.mean(all_velocities) if all_velocities else 0,
        'max_velocity': np.max(all_velocities) if all_velocities else 0,
        'top_10_active_tracks': [tid for tid, _ in top_tracks],
        'config': data.get('config', {})
    }
    
    if 'lifetime_stats' in data:
        stats_report['lifetime_stats'] = data['lifetime_stats']
    
    # Save statistics
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats_report, f, indent=2)
    
    print(f"\n✓ Heatmaps and analytics saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - heatmap_overall.png: Overall player density")
    print("  - heatmap_top10.png: Individual heatmaps for top 10 players")
    print("  - velocity_distribution.png: Player speed analysis")
    print("  - track_lifetime_distribution.png: Track quality metrics")
    print("  - movement_flow.png: Movement patterns and trajectories")
    print("  - statistics.json: Detailed analytics")
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heatmaps and analytics from tracking data')
    parser.add_argument('tracks_file', help='Path to tracks JSON file')
    parser.add_argument('--output', '-o', default='heatmaps', help='Output directory (default: heatmaps)')
    parser.add_argument('--width', type=int, default=1280, help='Frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Frame height (default: 720)')
    
    args = parser.parse_args()
    
    generate_heatmaps(args.tracks_file, args.output, args.width, args.height)
