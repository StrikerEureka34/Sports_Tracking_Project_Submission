import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import time

from detection import PlayerDetector
from tracking import PlayerTracker, TrackHistory
from projection import TopViewProjector
from analytics import TrajectoryAnalytics
from visualization import VideoVisualizer, TopViewVisualizer
import config


class PlayerTrackingPipeline:
    def __init__(
        self,
        video_path: str,
        output_dir: str = "output",
        source_points: np.ndarray = None,
        save_video: bool = True,
        save_analytics: bool = True
    ):
        self.video_path = video_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.save_video = save_video
        self.save_analytics = save_analytics
        
        print("Initializing pipeline...")
        self.detector = PlayerDetector()
        self.tracker = PlayerTracker()
        self.track_history = TrackHistory()
        self.video_viz = VideoVisualizer()
        self.topview_viz = TopViewVisualizer()
        
        self.projector = None
        self.analytics = None
        if source_points is not None:
            self.projector = TopViewProjector()
            self.projector.set_homography(source_points)
            self.analytics = TrajectoryAnalytics()
            print("Top-view projection enabled")
        
        print("Pipeline ready")

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
        
        writers = {}
        if self.save_video:
            writers['main'] = cv2.VideoWriter(
                str(self.output_dir / 'output_main.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (width, height)
            )
            
            if self.projector is not None:
                writers['topview'] = cv2.VideoWriter(
                    str(self.output_dir / 'output_topview.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps,
                    (self.projector.output_width, self.projector.output_height)
                )
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing")
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            bboxes, confidences, class_ids = self.detector.detect(frame)
            detections = PlayerTracker.detections_to_supervision(bboxes, confidences, class_ids)
            detections = self.tracker.update(detections)
            bottom_centers = self.detector.get_bbox_bottom_centers(bboxes)
            self.track_history.update(detections, frame_idx, bottom_centers)
            
            annotated_frame = self.video_viz.annotate_frame(frame, detections, show_labels=True, show_trajectories=True)
            processing_fps = (frame_idx + 1) / (time.time() - start_time)
            annotated_frame = self.video_viz.draw_info_panel(annotated_frame, frame_idx, len(detections), processing_fps)
            
            topview_frame = None
            if self.projector is not None and len(bottom_centers) > 0:
                projected_positions = self.projector.project_points(bottom_centers)
                projected_positions, valid_track_ids = self.projector.filter_out_of_bounds(projected_positions, detections.tracker_id)
                
                if len(projected_positions) > 0 and valid_track_ids is not None:
                    if detections.confidence is not None and detections.tracker_id is not None:
                        conf_dict = {tid: conf for tid, conf in zip(detections.tracker_id, detections.confidence)}
                        confidences_filtered = np.array([conf_dict.get(tid, 1.0) for tid in valid_track_ids])
                    else:
                        confidences_filtered = None
                    
                    self.analytics.add_trajectory_data(frame_idx, valid_track_ids, projected_positions, confidences_filtered)
                
                recent_trajectories = {}
                if detections.tracker_id is not None:
                    for track_id in detections.tracker_id:
                        recent_traj = self.track_history.get_recent_positions(int(track_id), n=30)
                        if len(recent_traj) > 1:
                            projected_traj = self.projector.project_points(recent_traj)
                            recent_trajectories[int(track_id)] = projected_traj
                
                topview_frame = self.topview_viz.draw_positions(
                    projected_positions,
                    valid_track_ids if len(projected_positions) > 0 else None
                )
                
                if recent_trajectories:
                    for track_id, traj in recent_trajectories.items():
                        if len(traj) < 2:
                            continue
                        np.random.seed(int(track_id))
                        color = tuple(np.random.randint(100, 255, 3).tolist())
                        points = traj.astype(np.int32)
                        for j in range(len(points) - 1):
                            cv2.line(topview_frame, tuple(points[j]), tuple(points[j + 1]), color, 2)
            
            if self.save_video:
                writers['main'].write(annotated_frame)
                if topview_frame is not None:
                    writers['topview'].write(topview_frame)
            
            frame_idx += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        for writer in writers.values():
            writer.release()
        
        print(f"\nProcessing complete: {frame_idx} frames at {frame_idx / (time.time() - start_time):.2f} FPS")
        
        if self.analytics is not None:
            self._generate_analytics()

    def _generate_analytics(self):
        print("Generating analytics...")
        
        if self.save_analytics:
            df = self.analytics.get_dataframe()
            df.to_csv(self.output_dir / 'trajectories.csv', index=False)
            print(f"Saved {len(df)} trajectory records")
            
            stats_df = self.analytics.get_all_statistics(fps=30.0)
            if len(stats_df) > 0:
                stats_df.to_csv(self.output_dir / 'track_statistics.csv', index=False)
                print(f"Saved statistics for {len(stats_df)} tracks")
        
        heatmap = self.analytics.compute_heatmap(gaussian_sigma=3.0)
        heatmap_img = self.topview_viz.draw_heatmap(heatmap)
        cv2.imwrite(str(self.output_dir / 'heatmap.png'), heatmap_img)
        
        all_trajectories = {}
        for track_id in self.analytics.trajectories.keys():
            traj = self.analytics.get_trajectory_array(track_id)
            if len(traj) >= config.ANALYTICS_CONFIG["min_track_length"]:
                all_trajectories[track_id] = traj
        
        combined_img = self.topview_viz.draw_combined(heatmap, trajectories=all_trajectories)
        cv2.imwrite(str(self.output_dir / 'combined_analysis.png'), combined_img)
        print("Analytics saved")


def main():
    parser = argparse.ArgumentParser(description='Player Tracking Pipeline')
    parser.add_argument('video', type=str, help='Path to input video')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--no-save-video', action='store_true', help='Skip video output')
    parser.add_argument('--no-save-analytics', action='store_true', help='Skip analytics output')
    parser.add_argument('--source-points', type=str, default=None, help='Homography points (x1,y1,x2,y2,...)')
    
    args = parser.parse_args()
    
    source_points = None
    if args.source_points:
        coords = [float(x) for x in args.source_points.split(',')]
        if len(coords) != 8:
            raise ValueError("source_points must have 8 values")
        source_points = np.array(coords).reshape(4, 2)
    
    pipeline = PlayerTrackingPipeline(
        video_path=args.video,
        output_dir=args.output,
        source_points=source_points,
        save_video=not args.no_save_video,
        save_analytics=not args.no_save_analytics
    )
    
    pipeline.process_video()
    print("Done")


if __name__ == "__main__":
    main()
