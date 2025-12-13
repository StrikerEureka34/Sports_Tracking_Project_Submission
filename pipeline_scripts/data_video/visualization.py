import numpy as np
import cv2
import supervision as sv
from matplotlib import cm
from typing import Optional, Tuple, List
import config


class VideoVisualizer:
    """Annotates video frames with detections, labels, and trajectories."""
    
    def __init__(self, bbox_thickness=None, text_scale=None, trajectory_length=None):
        bbox_thickness = bbox_thickness or config.VISUALIZATION_CONFIG["bbox_thickness"]
        text_scale = text_scale or config.VISUALIZATION_CONFIG["text_scale"]
        trajectory_length = trajectory_length or config.VISUALIZATION_CONFIG["trajectory_length"]
        self.bbox_thickness = bbox_thickness
        self.text_scale = text_scale
        self.trajectory_length = trajectory_length
        
        self.bbox_annotator = sv.BoxAnnotator(thickness=bbox_thickness)
        self.label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=bbox_thickness)
        self.trace_annotator = sv.TraceAnnotator(thickness=bbox_thickness, trace_length=trajectory_length)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        show_labels: bool = True,
        show_trajectories: bool = True
    ) -> np.ndarray:
        annotated = frame.copy()
        
        if show_trajectories and detections.tracker_id is not None:
            annotated = self.trace_annotator.annotate(scene=annotated, detections=detections)
        
        annotated = self.bbox_annotator.annotate(scene=annotated, detections=detections)
        
        if show_labels and detections.tracker_id is not None:
            labels = [f"ID: {track_id}" for track_id in detections.tracker_id]
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        return annotated

    def draw_info_panel(
        self,
        frame: np.ndarray,
        frame_idx: int,
        num_detections: int,
        fps: Optional[float] = None
    ) -> np.ndarray:
        annotated = frame.copy()
        
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        
        y_offset = 35
        cv2.putText(annotated, f"Frame: {frame_idx}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y_offset += 25
        cv2.putText(annotated, f"Detections: {num_detections}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if fps is not None:
            y_offset += 25
            cv2.putText(annotated, f"FPS: {fps:.1f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated


class TopViewVisualizer:
    def __init__(
        self,
        field_width: int = config.HOMOGRAPHY_CONFIG["output_width"],
        field_height: int = config.HOMOGRAPHY_CONFIG["output_height"],
        trajectory_thickness: int = config.VISUALIZATION_CONFIG["trajectory_thickness"],
        heatmap_alpha: float = config.VISUALIZATION_CONFIG["heatmap_alpha"]
    ):
        self.field_width = field_width
        self.field_height = field_height
        self.trajectory_thickness = trajectory_thickness
        self.heatmap_alpha = heatmap_alpha
        self.field_template = self._create_field_template()

    def _create_field_template(self) -> np.ndarray:
        field = np.full((self.field_height, self.field_width, 3), (34, 139, 34), dtype=np.uint8)
        
        cv2.rectangle(field, (5, 5), (self.field_width - 5, self.field_height - 5), (255, 255, 255), 2)
        
        mid_x = self.field_width // 2
        cv2.line(field, (mid_x, 5), (mid_x, self.field_height - 5), (255, 255, 255), 2)
        
        center = (mid_x, self.field_height // 2)
        radius = int(self.field_height * 0.15)
        cv2.circle(field, center, radius, (255, 255, 255), 2)
        
        return field

    def draw_positions(
        self,
        positions: np.ndarray,
        track_ids: Optional[np.ndarray] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        radius: int = 5
    ) -> np.ndarray:
        field = self.field_template.copy()
        
        if len(positions) == 0:
            return field
        
        for i, pos in enumerate(positions):
            x, y = int(pos[0]), int(pos[1])
            
            if colors is not None and i < len(colors):
                color = colors[i]
            else:
                color = (0, 0, 255)
            
            cv2.circle(field, (x, y), radius, color, -1)
            cv2.circle(field, (x, y), radius + 2, (255, 255, 255), 2)
            
            if track_ids is not None and i < len(track_ids):
                cv2.putText(field, str(int(track_ids[i])), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return field

    def draw_trajectories(
        self,
        trajectories: dict,
        colors: Optional[dict] = None,
        thickness: Optional[int] = None
    ) -> np.ndarray:
        field = self.field_template.copy()
        
        if thickness is None:
            thickness = self.trajectory_thickness
        
        for track_id, trajectory in trajectories.items():
            if len(trajectory) < 2:
                continue
            
            if colors is not None and track_id in colors:
                color = colors[track_id]
            else:
                np.random.seed(int(track_id))
                color = tuple(np.random.randint(0, 255, 3).tolist())
            
            points = trajectory.astype(np.int32)
            for j in range(len(points) - 1):
                cv2.line(field, tuple(points[j]), tuple(points[j + 1]), color, thickness)
            
            end_point = tuple(points[-1])
            cv2.circle(field, end_point, 6, color, -1)
            cv2.circle(field, end_point, 8, (255, 255, 255), 2)
        
        return field

    def draw_heatmap(
        self,
        heatmap: np.ndarray,
        colormap: str = config.VISUALIZATION_CONFIG["heatmap_colormap"],
        alpha: Optional[float] = None
    ) -> np.ndarray:
        field = self.field_template.copy()
        
        if alpha is None:
            alpha = self.heatmap_alpha
        
        if heatmap.max() > 0:
            heatmap_norm = heatmap / heatmap.max()
        else:
            return field
        
        heatmap_resized = cv2.resize(heatmap_norm, (self.field_width, self.field_height), interpolation=cv2.INTER_LINEAR)
        
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)
        heatmap_colored = (heatmap_colored[:, :, :3] * 255).astype(np.uint8)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR)
        
        mask = heatmap_resized > 0.01
        mask = mask.astype(np.uint8)
        
        field_float = field.astype(float)
        heatmap_float = heatmap_colored.astype(float)
        
        for c in range(3):
            field_float[:, :, c] = np.where(
                mask,
                field_float[:, :, c] * (1 - alpha) + heatmap_float[:, :, c] * alpha,
                field_float[:, :, c]
            )
        
        return field_float.astype(np.uint8)

    def draw_combined(
        self,
        heatmap: np.ndarray,
        trajectories: Optional[dict] = None,
        current_positions: Optional[np.ndarray] = None,
        track_ids: Optional[np.ndarray] = None
    ) -> np.ndarray:
        field = self.draw_heatmap(heatmap)
        
        if trajectories is not None:
            for track_id, trajectory in trajectories.items():
                if len(trajectory) < 2:
                    continue
                
                np.random.seed(int(track_id))
                color = tuple(np.random.randint(100, 255, 3).tolist())
                
                points = trajectory.astype(np.int32)
                for j in range(len(points) - 1):
                    cv2.line(field, tuple(points[j]), tuple(points[j + 1]), color, self.trajectory_thickness)
        
        if current_positions is not None and len(current_positions) > 0:
            for i, pos in enumerate(current_positions):
                x, y = int(pos[0]), int(pos[1])
                cv2.circle(field, (x, y), 6, (0, 0, 255), -1)
                cv2.circle(field, (x, y), 8, (255, 255, 255), 2)
                
                if track_ids is not None and i < len(track_ids):
                    cv2.putText(field, str(int(track_ids[i])), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return field
