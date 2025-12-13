import numpy as np
import supervision as sv
from typing import Optional, Dict, List
import config


class PlayerTracker:
    def __init__(
        self,
        track_thresh: float = config.TRACKING_CONFIG["track_thresh"],
        track_buffer: int = config.TRACKING_CONFIG["track_buffer"],
        match_thresh: float = config.TRACKING_CONFIG["match_thresh"],
        frame_rate: int = config.TRACKING_CONFIG["frame_rate"]
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh,
            frame_rate=frame_rate
        )
        print("ByteTrack tracker initialized")

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(detections)

    def reset(self):
        self.tracker.reset()

    @staticmethod
    def detections_to_supervision(
        bboxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray
    ) -> sv.Detections:
        if len(bboxes) == 0:
            return sv.Detections.empty()
        
        return sv.Detections(
            xyxy=bboxes,
            confidence=confidences,
            class_id=class_ids
        )


class TrackHistory:
    def __init__(self, max_length: int = 1000):
        self.max_length = max_length
        self.tracks: Dict[int, List[Dict]] = {}

    def update(
        self,
        detections: sv.Detections,
        frame_idx: int,
        positions: Optional[np.ndarray] = None
    ):
        if detections.tracker_id is None:
            return
        
        if positions is None:
            positions = self._get_bottom_centers(detections.xyxy)
        
        for i, track_id in enumerate(detections.tracker_id):
            track_id = int(track_id)
            
            if track_id not in self.tracks:
                self.tracks[track_id] = []
            
            self.tracks[track_id].append({
                'frame': frame_idx,
                'position': positions[i],
                'bbox': detections.xyxy[i],
                'confidence': detections.confidence[i] if detections.confidence is not None else 1.0
            })
            
            if len(self.tracks[track_id]) > self.max_length:
                self.tracks[track_id].pop(0)

    def get_track(self, track_id: int) -> List[Dict]:
        return self.tracks.get(track_id, [])

    def get_recent_positions(self, track_id: int, n: int = 30) -> np.ndarray:
        track = self.tracks.get(track_id, [])
        if not track:
            return np.empty((0, 2))
        
        recent = track[-n:]
        return np.array([entry['position'] for entry in recent])

    def get_all_positions(self, track_id: int) -> np.ndarray:
        track = self.tracks.get(track_id, [])
        if not track:
            return np.empty((0, 2))
        return np.array([entry['position'] for entry in track])

    def get_active_tracks(self) -> List[int]:
        return list(self.tracks.keys())

    @staticmethod
    def _get_bottom_centers(bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return np.empty((0, 2))
        return np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,
            bboxes[:, 3]
        ])
