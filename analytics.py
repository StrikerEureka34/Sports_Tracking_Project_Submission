import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scipy.ndimage import gaussian_filter
import config


class TrajectoryAnalytics:
    def __init__(
        self,
        field_width: int = config.HOMOGRAPHY_CONFIG["output_width"],
        field_height: int = config.HOMOGRAPHY_CONFIG["output_height"],
        heatmap_bins: int = config.ANALYTICS_CONFIG["heatmap_bins"],
        smoothing_window: int = config.ANALYTICS_CONFIG["smoothing_window"],
        min_track_length: int = config.ANALYTICS_CONFIG["min_track_length"]
    ):
        self.field_width = field_width
        self.field_height = field_height
        self.heatmap_bins = heatmap_bins
        self.smoothing_window = smoothing_window
        self.min_track_length = min_track_length
        self.trajectories: Dict[int, np.ndarray] = {}
        self.frame_data: List[Dict] = []

    def add_trajectory_data(
        self,
        frame_idx: int,
        track_ids: np.ndarray,
        positions: np.ndarray,
        confidences: Optional[np.ndarray] = None
    ):
        if len(track_ids) == 0:
            return
        
        for i, track_id in enumerate(track_ids):
            track_id = int(track_id)
            
            self.frame_data.append({
                'frame': frame_idx,
                'track_id': track_id,
                'x': positions[i, 0],
                'y': positions[i, 1],
                'confidence': confidences[i] if confidences is not None else 1.0
            })
            
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            
            self.trajectories[track_id].append({
                'frame': frame_idx,
                'position': positions[i]
            })

    def get_dataframe(self) -> pd.DataFrame:
        if not self.frame_data:
            return pd.DataFrame(columns=['frame', 'track_id', 'x', 'y', 'confidence'])
        return pd.DataFrame(self.frame_data)

    def get_trajectory_array(self, track_id: int) -> np.ndarray:
        if track_id not in self.trajectories:
            return np.empty((0, 2))
        return np.array([entry['position'] for entry in self.trajectories[track_id]])

    def smooth_trajectory(self, trajectory: np.ndarray, window_size: Optional[int] = None) -> np.ndarray:
        if window_size is None:
            window_size = self.smoothing_window
        
        if len(trajectory) < window_size:
            return trajectory
        
        smoothed = np.copy(trajectory).astype(float)
        for i in range(len(trajectory)):
            start = max(0, i - window_size // 2)
            end = min(len(trajectory), i + window_size // 2 + 1)
            smoothed[i] = np.mean(trajectory[start:end], axis=0)
        
        return smoothed

    def compute_heatmap(self, track_ids: Optional[List[int]] = None, gaussian_sigma: float = 2.0) -> np.ndarray:
        all_positions = []
        
        if track_ids is None:
            track_ids = list(self.trajectories.keys())
        
        for track_id in track_ids:
            if track_id in self.trajectories:
                positions = np.array([entry['position'] for entry in self.trajectories[track_id]])
                valid_mask = (
                    (positions[:, 0] >= 0) &
                    (positions[:, 0] < self.field_width) &
                    (positions[:, 1] >= 0) &
                    (positions[:, 1] < self.field_height)
                )
                all_positions.append(positions[valid_mask])
        
        if not all_positions:
            return np.zeros((self.heatmap_bins, self.heatmap_bins))
        
        all_positions = np.vstack(all_positions)
        
        heatmap, _, _ = np.histogram2d(
            all_positions[:, 0],
            all_positions[:, 1],
            bins=self.heatmap_bins,
            range=[[0, self.field_width], [0, self.field_height]]
        )
        
        heatmap = gaussian_filter(heatmap, sigma=gaussian_sigma)
        return heatmap.T

    def compute_speed(self, track_id: int, fps: float = 30.0, meters_per_pixel: Optional[float] = None) -> np.ndarray:
        trajectory = self.get_trajectory_array(track_id)
        
        if len(trajectory) < 2:
            return np.array([])
        
        displacements = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(displacements, axis=1)
        speeds = distances * fps
        
        if meters_per_pixel is not None:
            speeds *= meters_per_pixel
        
        return speeds

    def compute_distance_traveled(self, track_id: int, meters_per_pixel: Optional[float] = None) -> float:
        trajectory = self.get_trajectory_array(track_id)
        
        if len(trajectory) < 2:
            return 0.0
        
        displacements = np.diff(trajectory, axis=0)
        total_distance = np.sum(np.linalg.norm(displacements, axis=1))
        
        if meters_per_pixel is not None:
            total_distance *= meters_per_pixel
        
        return float(total_distance)

    def get_track_statistics(self, track_id: int, fps: float = 30.0, meters_per_pixel: Optional[float] = None) -> Dict:
        trajectory = self.get_trajectory_array(track_id)
        
        if len(trajectory) < 2:
            return {
                'track_id': track_id,
                'num_points': len(trajectory),
                'duration_frames': 0,
                'duration_seconds': 0.0,
                'total_distance': 0.0,
                'avg_speed': 0.0,
                'max_speed': 0.0
            }
        
        speeds = self.compute_speed(track_id, fps, meters_per_pixel)
        distance = self.compute_distance_traveled(track_id, meters_per_pixel)
        
        return {
            'track_id': track_id,
            'num_points': len(trajectory),
            'duration_frames': len(trajectory),
            'duration_seconds': len(trajectory) / fps,
            'total_distance': distance,
            'avg_speed': np.mean(speeds) if len(speeds) > 0 else 0.0,
            'max_speed': np.max(speeds) if len(speeds) > 0 else 0.0,
            'min_x': np.min(trajectory[:, 0]),
            'max_x': np.max(trajectory[:, 0]),
            'min_y': np.min(trajectory[:, 1]),
            'max_y': np.max(trajectory[:, 1])
        }

    def get_all_statistics(self, fps: float = 30.0, meters_per_pixel: Optional[float] = None) -> pd.DataFrame:
        stats_list = []
        
        for track_id in self.trajectories.keys():
            if len(self.trajectories[track_id]) >= self.min_track_length:
                stats = self.get_track_statistics(track_id, fps, meters_per_pixel)
                stats_list.append(stats)
        
        if not stats_list:
            return pd.DataFrame()
        
        return pd.DataFrame(stats_list)

    def clear(self):
        self.trajectories.clear()
        self.frame_data.clear()
