"""
Camera Calibration Data Collection Script
Phase 1: Camera Calibration & Stereo Depth

This script provides an interactive interface for collecting calibration images
from a camera (webcam or connected camera). It helps users capture high-quality
checkerboard images from various angles and distances for accurate calibration.
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from typing import Tuple, Optional
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationDataCollector:
    """
    Interactive calibration data collection tool.
    
    Provides real-time feedback on checkerboard detection quality
    and guides users to capture diverse calibration images.
    """
    
    def __init__(self, 
                 camera_id: int = 0,
                 checkerboard_size: Tuple[int, int] = (9, 6),
                 output_dir: str = "calibration_images",
                 min_images: int = 20):
        """
        Initialize data collector.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            checkerboard_size: (corners_x, corners_y) in checkerboard
            output_dir: Directory to save captured images
            min_images: Minimum recommended number of images
        """
        self.camera_id = camera_id
        self.checkerboard_size = checkerboard_size
        self.output_dir = output_dir
        self.min_images = min_images
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize camera
        self.cap = None
        self.image_count = 0
        
        # Detection criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Quality metrics
        self.quality_scores = []
        
    def start_camera(self):
        """Initialize camera connection."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_id}")
            
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info(f"Camera {self.camera_id} initialized")
        
    def stop_camera(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
    def detect_checkerboard_quality(self, image: np.ndarray) -> Tuple[bool, float, Optional[np.ndarray]]:
        """
        Detect checkerboard and assess quality.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (detected, quality_score, corners)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if not ret:
            return False, 0.0, None
            
        # Refine corners
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        
        # Calculate quality metrics
        quality_score = self._calculate_quality(gray, corners)
        
        return True, quality_score, corners
        
    def _calculate_quality(self, gray: np.ndarray, corners: np.ndarray) -> float:
        """
        Calculate image quality score for calibration.
        
        Quality is based on:
        - Sharpness (gradient magnitude at corners)
        - Corner distribution (coverage of image area)
        - Perspective diversity
        """
        quality = 0.0
        
        # 1. Sharpness metric
        total_sharpness = 0.0
        for corner in corners:
            x, y = int(corner[0][0]), int(corner[0][1])
            
            # Calculate gradient magnitude in local region
            if 5 <= x < gray.shape[1] - 5 and 5 <= y < gray.shape[0] - 5:
                region = gray[y-5:y+5, x-5:x+5].astype(np.float32)
                grad_x = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                total_sharpness += np.mean(magnitude)
                
        sharpness_score = min(total_sharpness / len(corners) / 50.0, 1.0)  # Normalize
        
        # 2. Coverage metric (how well corners spread across image)
        corner_points = corners.reshape(-1, 2)
        min_x, max_x = np.min(corner_points[:, 0]), np.max(corner_points[:, 0])
        min_y, max_y = np.min(corner_points[:, 1]), np.max(corner_points[:, 1])
        
        coverage_x = (max_x - min_x) / gray.shape[1]
        coverage_y = (max_y - min_y) / gray.shape[0]
        coverage_score = min(coverage_x * coverage_y * 2.0, 1.0)  # Prefer larger coverage
        
        # 3. Perspective metric (check if board is at an angle)
        # Calculate aspect ratio of detected board
        width_top = np.linalg.norm(corner_points[self.checkerboard_size[0]-1] - corner_points[0])
        width_bottom = np.linalg.norm(corner_points[-1] - corner_points[-self.checkerboard_size[0]])
        height_left = np.linalg.norm(corner_points[-self.checkerboard_size[0]] - corner_points[0])
        height_right = np.linalg.norm(corner_points[-1] - corner_points[self.checkerboard_size[0]-1])
        
        width_ratio = min(width_top, width_bottom) / max(width_top, width_bottom)
        height_ratio = min(height_left, height_right) / max(height_left, height_right)
        
        # Moderate perspective is good (not too frontal, not too extreme)
        perspective_score = 1.0 - abs(0.8 - min(width_ratio, height_ratio)) / 0.8
        perspective_score = max(0.0, perspective_score)
        
        # Combine metrics
        quality = 0.4 * sharpness_score + 0.4 * coverage_score + 0.2 * perspective_score
        
        return quality
        
    def get_quality_color(self, quality: float) -> Tuple[int, int, int]:
        """Get color for quality visualization."""
        if quality > 0.7:
            return (0, 255, 0)    # Green - Excellent
        elif quality > 0.5:
            return (0, 255, 255)  # Yellow - Good
        elif quality > 0.3:
            return (0, 165, 255)  # Orange - Fair
        else:
            return (0, 0, 255)    # Red - Poor
            
    def draw_feedback(self, image: np.ndarray, detected: bool, quality: float, 
                     corners: Optional[np.ndarray]) -> np.ndarray:
        """
        Draw visual feedback on the image.
        
        Args:
            image: Input image
            detected: Whether checkerboard was detected
            quality: Quality score
            corners: Detected corners
            
        Returns:
            Image with feedback overlays
        """
        feedback_image = image.copy()
        
        # Draw corners if detected
        if detected and corners is not None:
            cv2.drawChessboardCorners(feedback_image, self.checkerboard_size, corners, detected)
            
            # Quality indicator
            color = self.get_quality_color(quality)
            cv2.rectangle(feedback_image, (10, 10), (200, 50), color, -1)
            cv2.putText(feedback_image, f"Quality: {quality:.2f}", (15, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                       
        # Status text
        status_text = "DETECTED" if detected else "NOT DETECTED"
        status_color = (0, 255, 0) if detected else (0, 0, 255)
        cv2.putText(feedback_image, status_text, (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Instructions
        instructions = [
            f"Images captured: {self.image_count}/{self.min_images}",
            "SPACE: Capture image",
            "Q: Quit",
            "",
            "Tips:",
            "- Move board to different positions",
            "- Vary angles and distances", 
            "- Ensure good lighting",
            "- Keep board flat and stable"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = feedback_image.shape[0] - (len(instructions) - i) * 25
            cv2.putText(feedback_image, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return feedback_image
        
    def capture_image(self, image: np.ndarray, quality: float) -> bool:
        """
        Save captured calibration image.
        
        Args:
            image: Image to save
            quality: Quality score of the image
            
        Returns:
            True if image was saved successfully
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"calibration_{timestamp}_{self.image_count:03d}_q{quality:.2f}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        success = cv2.imwrite(filepath, image)
        
        if success:
            self.image_count += 1
            self.quality_scores.append(quality)
            logger.info(f"Captured image {self.image_count}: {filename} (quality: {quality:.2f})")
            return True
        else:
            logger.error(f"Failed to save image: {filepath}")
            return False
            
    def collect_data(self):
        """Main data collection loop."""
        self.start_camera()
        
        try:
            logger.info("Starting calibration data collection...")
            logger.info("Move the checkerboard to different positions and press SPACE to capture")
            logger.info("Press Q to quit")
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logger.error("Failed to read from camera")
                    break
                    
                # Detect checkerboard and assess quality
                detected, quality, corners = self.detect_checkerboard_quality(frame)
                
                # Create feedback visualization
                display_image = self.draw_feedback(frame, detected, quality, corners)
                
                # Show image
                cv2.imshow('Calibration Data Collection', display_image)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # SPACE
                    if detected and quality > 0.3:  # Minimum quality threshold
                        self.capture_image(frame, quality)
                        
                        # Brief pause to show capture feedback
                        capture_feedback = display_image.copy()
                        cv2.rectangle(capture_feedback, (10, 100), (300, 140), (0, 255, 0), -1)
                        cv2.putText(capture_feedback, "IMAGE CAPTURED!", (15, 125), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        cv2.imshow('Calibration Data Collection', capture_feedback)
                        cv2.waitKey(500)  # Show for 500ms
                    else:
                        # Show error feedback
                        error_feedback = display_image.copy()
                        cv2.rectangle(error_feedback, (10, 100), (350, 140), (0, 0, 255), -1)
                        error_text = "Low quality or not detected!" if detected else "Checkerboard not detected!"
                        cv2.putText(error_feedback, error_text, (15, 125), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.imshow('Calibration Data Collection', error_feedback)
                        cv2.waitKey(500)
                        
        finally:
            self.stop_camera()
            
        # Print summary
        self._print_summary()
        
    def _print_summary(self):
        """Print collection summary."""
        print("\n" + "="*50)
        print("CALIBRATION DATA COLLECTION SUMMARY")
        print("="*50)
        print(f"Total images captured: {self.image_count}")
        print(f"Minimum recommended: {self.min_images}")
        print(f"Output directory: {self.output_dir}")
        
        if self.quality_scores:
            print(f"Average quality: {np.mean(self.quality_scores):.3f}")
            print(f"Quality range: {np.min(self.quality_scores):.3f} - {np.max(self.quality_scores):.3f}")
            
        if self.image_count >= self.min_images:
            print("✅ Sufficient images collected for calibration!")
        else:
            print(f"⚠️  Collect {self.min_images - self.image_count} more images for better results")
            
        print(f"\nNext steps:")
        print(f"1. Review images in {self.output_dir}")
        print(f"2. Remove blurry or poor quality images")
        print(f"3. Run calibration using: python calibrate_camera.py")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Collect calibration images for camera calibration")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--output", type=str, default="calibration_images", 
                       help="Output directory for images")
    parser.add_argument("--checkerboard", type=str, default="9x6", 
                       help="Checkerboard size as WxH (default: 9x6)")
    parser.add_argument("--min-images", type=int, default=20, 
                       help="Minimum number of images to collect")
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    try:
        w, h = map(int, args.checkerboard.split('x'))
        checkerboard_size = (w, h)
    except ValueError:
        print("Error: Checkerboard size must be in format WxH (e.g., 9x6)")
        return
    
    # Create collector and start
    collector = CalibrationDataCollector(
        camera_id=args.camera,
        checkerboard_size=checkerboard_size,
        output_dir=args.output,
        min_images=args.min_images
    )
    
    try:
        collector.collect_data()
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"Error during collection: {e}")


if __name__ == "__main__":
    main()