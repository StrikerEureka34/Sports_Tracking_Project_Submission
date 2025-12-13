from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DETECTION_CONFIG = {
    "model_name": "yolov8x.pt",
    "conf_threshold": 0.3,
    "iou_threshold": 0.5,
    "device": "cuda:0",
    "classes": [0],
    "imgsz": 1280,
}

TRACKING_CONFIG = {
    "track_thresh": 0.45,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "frame_rate": 30,
}

HOMOGRAPHY_CONFIG = {
    "source_points": None,
    "dest_points": None,
    "field_width": 105,
    "field_height": 68,
    "output_width": 1050,
    "output_height": 680,
}

ANALYTICS_CONFIG = {
    "heatmap_bins": 50,
    "smoothing_window": 5,
    "min_track_length": 10,
}

VISUALIZATION_CONFIG = {
    "bbox_thickness": 2,
    "bbox_color": (0, 255, 0),
    "text_color": (255, 255, 255),
    "text_scale": 0.6,
    "trajectory_thickness": 2,
    "trajectory_length": 30,
    "heatmap_alpha": 0.6,
    "heatmap_colormap": "hot",
}

VIDEO_CONFIG = {
    "output_fps": 30,
    "output_codec": "mp4v",
    "skip_frames": 1,
}
