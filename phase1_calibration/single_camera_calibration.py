"""
Single Camera Calibration Module
Phase 1: Camera Calibration & Stereo Depth

This module implements comprehensive single camera calibration using OpenCV.
It estimates intrinsic parameters (focal length, principal point, distortion coefficients)
which are essential for accurate computer vision applications.
"""

import cv2
import numpy as np
import os
import glob
import json
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SingleCameraCalibrator:
    """
    Single camera calibration class using checkerboard patterns.
    
    This class handles the complete pipeline for camera calibration:
    1. Checkerboard detection in multiple images
    2. Camera parameter estimation
    3. Calibration quality assessment
    4. Results visualization and export
    """
    
    def __init__(self, 
                 checkerboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 25.0):
        """
        Initialize the calibrator.
        
        Args:
            checkerboard_size: (corners_x, corners_y) - internal corners in checkerboard
            square_size: Size of checkerboard squares in mm
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        
        # Calibration data
        self.object_points = []  # 3D points in world coordinate system
        self.image_points = []   # 2D points in image plane
        self.image_size = None
        
        # Calibration results
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.rvecs = None  # Rotation vectors
        self.tvecs = None  # Translation vectors
        self.reprojection_error = None
        
        # Prepare object points for checkerboard
        self._prepare_object_points()
        
    def _prepare_object_points(self):
        """Prepare 3D object points for the checkerboard pattern."""
        self.objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), 
                            np.float32)
        self.objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 
                                   0:self.checkerboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
    def detect_checkerboard(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Detect checkerboard corners in an image.
        
        Args:
            image: Input image (color or grayscale)
            
        Returns:
            Tuple of (success, corners) where corners are refined corner positions
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # Refine corner positions to sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners
        
        return False, None
    
    def add_calibration_image(self, image: np.ndarray) -> bool:
        """
        Add an image to the calibration dataset.
        
        Args:
            image: Calibration image containing checkerboard
            
        Returns:
            True if checkerboard was detected and added, False otherwise
        """
        ret, corners = self.detect_checkerboard(image)
        
        if ret:
            self.object_points.append(self.objp)
            self.image_points.append(corners)
            
            # Store image size (assuming all images have same size)
            if self.image_size is None:
                self.image_size = image.shape[:2][::-1]  # (width, height)
                
            logger.info(f"Added calibration image. Total images: {len(self.image_points)}")
            return True
        else:
            logger.warning("Could not detect checkerboard in image")
            return False
    
    def calibrate(self) -> Dict:
        """
        Perform camera calibration using collected images.
        
        Returns:
            Dictionary containing calibration results and statistics
        """
        if len(self.image_points) < 10:
            raise ValueError(f"Need at least 10 images for calibration, got {len(self.image_points)}")
        
        logger.info(f"Starting calibration with {len(self.image_points)} images...")
        
        # Perform calibration
        ret, self.camera_matrix, self.distortion_coeffs, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(self.object_points, self.image_points, 
                              self.image_size, None, None)
        
        self.reprojection_error = ret
        
        # Calculate additional statistics
        results = self._calculate_calibration_statistics()
        
        logger.info(f"Calibration completed. Reprojection error: {self.reprojection_error:.3f} pixels")
        
        return results
    
    def _calculate_calibration_statistics(self) -> Dict:
        """Calculate detailed calibration statistics."""
        # Per-image reprojection errors
        per_image_errors = []
        total_points = 0
        total_error = 0
        
        for i in range(len(self.object_points)):
            # Project 3D points to image plane
            projected_points, _ = cv2.projectPoints(
                self.object_points[i], self.rvecs[i], self.tvecs[i],
                self.camera_matrix, self.distortion_coeffs
            )
            
            # Calculate error for this image
            error = cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
            per_image_errors.append(error)
            
            total_points += len(self.object_points[i])
            total_error += cv2.norm(self.image_points[i], projected_points, cv2.NORM_L2) ** 2
        
        # Calculate focal lengths in pixels and mm
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        # Estimate pixel size (assuming square pixels)
        pixel_size_mm = self.square_size / np.sqrt(fx * fy) if fx > 0 and fy > 0 else None
        
        results = {
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.distortion_coeffs.tolist(),
            'reprojection_error': self.reprojection_error,
            'per_image_errors': per_image_errors,
            'mean_error': np.mean(per_image_errors),
            'std_error': np.std(per_image_errors),
            'focal_length_x': fx,
            'focal_length_y': fy,
            'principal_point': (cx, cy),
            'pixel_size_mm': pixel_size_mm,
            'image_size': self.image_size,
            'num_images': len(self.image_points),
            'checkerboard_size': self.checkerboard_size,
            'square_size': self.square_size
        }
        
        return results
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        Remove distortion from an image using calibration results.
        
        Args:
            image: Input distorted image
            
        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.distortion_coeffs is None:
            raise ValueError("Camera not calibrated yet")
            
        return cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
    
    def save_calibration(self, filepath: str):
        """
        Save calibration results to a JSON file.
        
        Args:
            filepath: Path to save calibration file
        """
        if self.camera_matrix is None:
            raise ValueError("No calibration data to save")
            
        results = self._calculate_calibration_statistics()
        
        # Add timestamp
        import datetime
        results['calibration_date'] = datetime.datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Calibration saved to {filepath}")
    
    def load_calibration(self, filepath: str):
        """
        Load calibration results from a JSON file.
        
        Args:
            filepath: Path to calibration file
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
            
        self.camera_matrix = np.array(results['camera_matrix'])
        self.distortion_coeffs = np.array(results['distortion_coefficients'])
        self.reprojection_error = results['reprojection_error']
        self.image_size = tuple(results['image_size'])
        
        logger.info(f"Calibration loaded from {filepath}")
    
    def visualize_calibration(self, sample_images: List[np.ndarray]) -> plt.Figure:
        """
        Create visualization of calibration results.
        
        Args:
            sample_images: List of sample images to show original vs undistorted
            
        Returns:
            Matplotlib figure with visualization
        """
        if self.camera_matrix is None:
            raise ValueError("Camera not calibrated yet")
            
        n_samples = min(len(sample_images), 3)
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 8))
        
        if n_samples == 1:
            axes = axes.reshape(2, 1)
            
        for i in range(n_samples):
            # Original image
            axes[0, i].imshow(cv2.cvtColor(sample_images[i], cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'Original Image {i+1}')
            axes[0, i].axis('off')
            
            # Undistorted image
            undistorted = self.undistort_image(sample_images[i])
            axes[1, i].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f'Undistorted Image {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        return fig


def calibrate_camera_from_images(image_dir: str, 
                                checkerboard_size: Tuple[int, int] = (9, 6),
                                square_size: float = 25.0,
                                save_path: Optional[str] = None) -> Dict:
    """
    Convenience function to calibrate camera from a directory of images.
    
    Args:
        image_dir: Directory containing calibration images
        checkerboard_size: (corners_x, corners_y) in checkerboard
        square_size: Size of checkerboard squares in mm
        save_path: Optional path to save calibration results
        
    Returns:
        Dictionary with calibration results
    """
    calibrator = SingleCameraCalibrator(checkerboard_size, square_size)
    
    # Load images
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    
    logger.info(f"Found {len(image_paths)} images in {image_dir}")
    
    # Process each image
    successful_images = 0
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is not None:
            if calibrator.add_calibration_image(image):
                successful_images += 1
    
    logger.info(f"Successfully processed {successful_images}/{len(image_paths)} images")
    
    if successful_images < 10:
        raise ValueError(f"Need at least 10 good images, got {successful_images}")
    
    # Perform calibration
    results = calibrator.calibrate()
    
    # Save results if requested
    if save_path:
        calibrator.save_calibration(save_path)
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Single Camera Calibration Module")
    print("This module provides tools for calibrating camera intrinsic parameters.")
    print("Use calibrate_camera_from_images() to calibrate from a directory of checkerboard images.")