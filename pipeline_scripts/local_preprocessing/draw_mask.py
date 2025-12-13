#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import json
from pathlib import Path


class MaskDrawer:
    """Interactive polygon mask drawer for defining exclusion zones."""
    
    def __init__(self, image):
        self.image = image.copy()
        self.original = image.copy()
        self.points = []
        self.drawing = False
        self.mask = None
        
    def draw_polygon(self):
        """Interactive drawing loop. Left-click to add points, 'c' to close, 's' to save."""
        window_name = 'Draw Exclusion Mask'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("MASK DRAWING INSTRUCTIONS:")
        print("="*60)
        print("Left Click: Add polygon vertex")
        print("'c': Close polygon (fill area)")
        print("'r': Reset and start over")
        print("'s': Save mask and quit")
        print("'q': Quit without saving")
        print("="*60 + "\n")
        
        while True:
            display = self.image.copy()
            
            # Draw current points
            for i, pt in enumerate(self.points):
                cv2.circle(display, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(display, self.points[i-1], pt, (0, 255, 0), 2)
            
            # Draw closed polygon if we have one
            if len(self.points) >= 3 and self.mask is not None:
                overlay = display.copy()
                cv2.fillPoly(overlay, [np.array(self.points)], (255, 0, 0))
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
                # Draw outline
                cv2.polylines(display, [np.array(self.points)], True, (0, 255, 0), 2)
            
            # Status text
            status = f"Points: {len(self.points)} | 'c':Close 'r':Reset 's':Save 'q':Quit"
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):  # Close polygon
                if len(self.points) >= 3:
                    self.create_mask()
                    print(f"✓ Polygon closed with {len(self.points)} points")
                else:
                    print("⚠ Need at least 3 points")
            
            elif key == ord('r'):  # Reset
                self.points = []
                self.mask = None
                self.image = self.original.copy()
                print("↻ Reset")
            
            elif key == ord('s'):  # Save
                if self.mask is not None:
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("⚠ No mask to save. Press 'c' to close polygon first.")
            
            elif key == ord('q'):  # Quit
                cv2.destroyAllWindows()
                return False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"✓ Point {len(self.points)}: ({x}, {y})")
    
    def create_mask(self):
        """Create binary mask from polygon points"""
        h, w = self.original.shape[:2]
        self.mask = np.zeros((h, w), dtype=np.uint8)
        
        # Fill polygon (white = keep, black = exclude)
        # Inverse: white inside exclusion zone
        cv2.fillPoly(self.mask, [np.array(self.points)], 255)
        
    def get_mask(self):
        """Return the binary mask"""
        return self.mask
    
    def get_points(self):
        """Return polygon points"""
        return self.points


def draw_exclusion_mask(frames_dir):
    """
    Draw exclusion mask on first frame.
    
    Args:
        frames_dir: Directory containing extracted frames
        
    Returns:
        mask: Binary mask (255 = exclude, 0 = keep)
        points: List of polygon points
    """
    frames_dir = Path(frames_dir)
    
    # Load first frame
    frames = sorted(frames_dir.glob('frame_*.jpg'))
    if not frames:
        raise ValueError(f"No frames found in {frames_dir}")
    
    first_frame = cv2.imread(str(frames[0]))
    if first_frame is None:
        raise ValueError(f"Cannot load frame: {frames[0]}")
    
    print(f"Loaded frame: {first_frame.shape[1]}x{first_frame.shape[0]}")
    
    # Draw mask
    drawer = MaskDrawer(first_frame)
    success = drawer.draw_polygon()
    
    if not success:
        print("⚠ Mask drawing cancelled")
        return None, None
    
    mask = drawer.get_mask()
    points = drawer.get_points()
    
    # Save mask
    mask_path = frames_dir / 'exclusion_mask.png'
    cv2.imwrite(str(mask_path), mask)
    print(f"✓ Mask saved to {mask_path}")
    
    # Save polygon points
    points_path = frames_dir / 'exclusion_polygon.json'
    with open(points_path, 'w') as f:
        json.dump({'points': points}, f, indent=2)
    print(f"✓ Polygon saved to {points_path}")
    
    # Visualize result
    result = first_frame.copy()
    overlay = result.copy()
    cv2.fillPoly(overlay, [np.array(points)], (0, 0, 255))
    cv2.addWeighted(overlay, 0.4, result, 0.6, 0, result)
    cv2.polylines(result, [np.array(points)], True, (0, 255, 0), 3)
    
    result_path = frames_dir / 'mask_preview.jpg'
    cv2.imwrite(str(result_path), result)
    print(f"✓ Preview saved to {result_path}")
    
    return mask, points


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Draw exclusion mask on first frame')
    parser.add_argument('frames_dir', type=str, help='Directory containing frames')
    
    args = parser.parse_args()
    
    mask, points = draw_exclusion_mask(args.frames_dir)
    
    if mask is not None:
        print(f"\n✓ SUCCESS! Mask created with {len(points)} vertices")
        print(f"  Exclusion area covers {(mask > 0).sum()} pixels")
    else:
        print("\n✗ No mask created")
