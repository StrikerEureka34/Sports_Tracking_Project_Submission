import numpy as np
import cv2
from typing import Tuple, Optional
import config


class TopViewProjector:
    def __init__(
        self,
        source_points: Optional[np.ndarray] = None,
        dest_points: Optional[np.ndarray] = None,
        field_width: float = config.HOMOGRAPHY_CONFIG["field_width"],
        field_height: float = config.HOMOGRAPHY_CONFIG["field_height"],
        output_width: int = config.HOMOGRAPHY_CONFIG["output_width"],
        output_height: int = config.HOMOGRAPHY_CONFIG["output_height"]
    ):
        self.field_width = field_width
        self.field_height = field_height
        self.output_width = output_width
        self.output_height = output_height
        self.source_points = source_points
        self.dest_points = dest_points
        self.homography_matrix = None
        
        if source_points is not None and dest_points is not None:
            self.set_homography(source_points, dest_points)

    def set_homography(self, source_points: np.ndarray, dest_points: Optional[np.ndarray] = None):
        self.source_points = np.array(source_points, dtype=np.float32)
        
        if dest_points is None:
            dest_points = np.array([
                [0, 0],
                [self.output_width, 0],
                [self.output_width, self.output_height],
                [0, self.output_height]
            ], dtype=np.float32)
        else:
            dest_points = np.array(dest_points, dtype=np.float32)
        
        self.dest_points = dest_points
        self.homography_matrix = cv2.getPerspectiveTransform(self.source_points, self.dest_points)
        print("Homography matrix computed")

    def project_point(self, point: np.ndarray) -> np.ndarray:
        if self.homography_matrix is None:
            raise ValueError("Homography not set")
        
        point = np.array([[point]], dtype=np.float32)
        projected = cv2.perspectiveTransform(point, self.homography_matrix)
        return projected[0][0]

    def project_points(self, points: np.ndarray) -> np.ndarray:
        if self.homography_matrix is None:
            raise ValueError("Homography not set")
        
        if len(points) == 0:
            return np.empty((0, 2))
        
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(points, self.homography_matrix)
        return projected.reshape(-1, 2)

    def project_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.homography_matrix is None:
            raise ValueError("Homography not set")
        
        return cv2.warpPerspective(
            frame,
            self.homography_matrix,
            (self.output_width, self.output_height)
        )

    def create_field_template(
        self,
        bg_color: Tuple[int, int, int] = (34, 139, 34),
        line_color: Tuple[int, int, int] = (255, 255, 255),
        line_thickness: int = 2
    ) -> np.ndarray:
        field = np.full((self.output_height, self.output_width, 3), bg_color, dtype=np.uint8)
        
        cv2.rectangle(field, (0, 0), (self.output_width - 1, self.output_height - 1), line_color, line_thickness)
        
        mid_x = self.output_width // 2
        cv2.line(field, (mid_x, 0), (mid_x, self.output_height), line_color, line_thickness)
        
        center = (mid_x, self.output_height // 2)
        radius = int(self.output_height * 0.15)
        cv2.circle(field, center, radius, line_color, line_thickness)
        
        return field

    def pixels_to_meters(self, pixel_distance: float) -> float:
        scale_x = self.field_width / self.output_width
        scale_y = self.field_height / self.output_height
        return pixel_distance * (scale_x + scale_y) / 2

    def meters_to_pixels(self, meter_distance: float) -> float:
        scale_x = self.output_width / self.field_width
        scale_y = self.output_height / self.field_height
        return meter_distance * (scale_x + scale_y) / 2

    def is_point_in_bounds(self, point: np.ndarray) -> bool:
        x, y = point
        return 0 <= x < self.output_width and 0 <= y < self.output_height

    def filter_out_of_bounds(
        self,
        points: np.ndarray,
        track_ids: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(points) == 0:
            return points, track_ids
        
        if track_ids is not None and len(points) != len(track_ids):
            min_len = min(len(points), len(track_ids))
            points = points[:min_len]
            track_ids = track_ids[:min_len]
        
        mask = (
            (points[:, 0] >= 0) &
            (points[:, 0] < self.output_width) &
            (points[:, 1] >= 0) &
            (points[:, 1] < self.output_height)
        )
        
        filtered_points = points[mask]
        filtered_track_ids = track_ids[mask] if track_ids is not None else None
        
        return filtered_points, filtered_track_ids
