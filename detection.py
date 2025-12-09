import numpy as np
from ultralytics import YOLO
from typing import List, Tuple
import config


class PlayerDetector:
    def __init__(
        self,
        model_name: str = config.DETECTION_CONFIG["model_name"],
        conf_threshold: float = config.DETECTION_CONFIG["conf_threshold"],
        iou_threshold: float = config.DETECTION_CONFIG["iou_threshold"],
        device: str = config.DETECTION_CONFIG["device"],
        classes: List[int] = config.DETECTION_CONFIG["classes"],
        imgsz: int = config.DETECTION_CONFIG["imgsz"]
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes
        self.imgsz = imgsz
        
        print(f"Loading YOLOv8 model: {model_name}")
        self.model = YOLO(model_name)
        self.model.to(device)
        print(f"Model loaded on {device}")

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        results = self.model.predict(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            bboxes = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            return bboxes, confidences, class_ids
        
        return np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)

    def detect_batch(self, frames: List[np.ndarray], batch_size: int = 8) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        results_list = self.model.predict(
            frames,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            batch=batch_size
        )
        
        detections = []
        for results in results_list:
            if results.boxes is not None:
                boxes = results.boxes
                bboxes = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)
                detections.append((bboxes, confidences, class_ids))
            else:
                detections.append((np.empty((0, 4)), np.empty((0,)), np.empty((0,), dtype=int)))
        
        return detections

    def get_bbox_centers(self, bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return np.empty((0, 2))
        
        centers = np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,
            (bboxes[:, 1] + bboxes[:, 3]) / 2
        ])
        return centers

    def get_bbox_bottom_centers(self, bboxes: np.ndarray) -> np.ndarray:
        if len(bboxes) == 0:
            return np.empty((0, 2))
        
        bottom_centers = np.column_stack([
            (bboxes[:, 0] + bboxes[:, 2]) / 2,
            bboxes[:, 3]
        ])
        return bottom_centers
