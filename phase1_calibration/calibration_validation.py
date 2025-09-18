"""
Camera Calibration Validation and Visualization Tools
Phase 1: Camera Calibration & Stereo Depth

This module provides comprehensive tools for validating camera calibration results
and visualizing calibration quality through various metrics and plots.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import json
import os
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CalibrationValidator:
    """
    Comprehensive validation tool for camera calibration results.
    
    Provides various visualization and analysis tools to assess calibration quality
    and identify potential issues.
    """
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            calibration_file: Path to calibration JSON file (optional)
        """
        self.calibration_data = None
        self.camera_matrix = None
        self.distortion_coeffs = None
        
        if calibration_file:
            self.load_calibration(calibration_file)
            
    def load_calibration(self, calibration_file: str):
        """Load calibration data from JSON file."""
        with open(calibration_file, 'r') as f:
            self.calibration_data = json.load(f)
            
        self.camera_matrix = np.array(self.calibration_data['camera_matrix'])
        self.distortion_coeffs = np.array(self.calibration_data['distortion_coefficients'])
        
        logger.info(f"Loaded calibration from {calibration_file}")
        
    def create_calibration_report(self) -> plt.Figure:
        """
        Create comprehensive calibration quality report.
        
        Returns:
            Figure containing multiple validation plots
        """
        if self.calibration_data is None:
            raise ValueError("No calibration data loaded")
            
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Calibration summary
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_calibration_summary(ax1)
        
        # 2. Reprojection error distribution
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_error_distribution(ax2)
        
        # 3. Per-image error analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_per_image_errors(ax3)
        
        # 4. Distortion visualization
        ax4 = fig.add_subplot(gs[1, 2])
        self._plot_distortion_model(ax4)
        
        # 5. Camera parameters
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_camera_parameters(ax5)
        
        # 6. Quality assessment
        ax6 = fig.add_subplot(gs[2, 1])
        self._plot_quality_assessment(ax6)
        
        # 7. Recommendations
        ax7 = fig.add_subplot(gs[2, 2])
        self._plot_recommendations(ax7)
        
        plt.suptitle('Camera Calibration Validation Report', fontsize=16, fontweight='bold')
        
        return fig
        
    def _plot_calibration_summary(self, ax):
        """Plot calibration summary information."""
        ax.axis('off')
        
        # Create summary text
        summary_text = [
            f"📊 Calibration Summary",
            f"",
            f"📸 Images used: {self.calibration_data['num_images']}",
            f"📐 Checkerboard: {self.calibration_data['checkerboard_size'][0]}×{self.calibration_data['checkerboard_size'][1]}",
            f"📏 Square size: {self.calibration_data['square_size']:.1f} mm",
            f"🖼️ Image size: {self.calibration_data['image_size'][0]}×{self.calibration_data['image_size'][1]}",
            f"",
            f"🎯 Reprojection error: {self.calibration_data['reprojection_error']:.3f} pixels",
            f"📊 Mean error: {self.calibration_data['mean_error']:.3f} ± {self.calibration_data['std_error']:.3f} pixels",
            f"",
            f"🔍 Focal length: fx={self.calibration_data['focal_length_x']:.1f}, fy={self.calibration_data['focal_length_y']:.1f}",
            f"🎯 Principal point: ({self.calibration_data['principal_point'][0]:.1f}, {self.calibration_data['principal_point'][1]:.1f})",
        ]
        
        for i, line in enumerate(summary_text):
            ax.text(0.05, 0.95 - i * 0.08, line, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='top',
                   fontweight='bold' if line.startswith('📊') else 'normal')
                   
        # Add quality indicator
        error = self.calibration_data['reprojection_error']
        if error < 0.5:
            quality_text = "🟢 Excellent calibration"
            quality_color = 'green'
        elif error < 1.0:
            quality_text = "🟡 Good calibration"
            quality_color = 'orange'
        else:
            quality_text = "🔴 Poor calibration - recalibrate"
            quality_color = 'red'
            
        ax.text(0.05, 0.05, quality_text, transform=ax.transAxes,
               fontsize=12, fontweight='bold', color=quality_color)
        
    def _plot_error_distribution(self, ax):
        """Plot distribution of reprojection errors."""
        errors = self.calibration_data['per_image_errors']
        
        ax.hist(errors, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(errors), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(errors):.3f}')
        ax.axvline(np.median(errors), color='green', linestyle='--', 
                  label=f'Median: {np.median(errors):.3f}')
        
        ax.set_xlabel('Reprojection Error (pixels)')
        ax.set_ylabel('Number of Images')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_per_image_errors(self, ax):
        """Plot per-image error analysis."""
        errors = self.calibration_data['per_image_errors']
        image_indices = range(1, len(errors) + 1)
        
        ax.plot(image_indices, errors, 'bo-', markersize=4, linewidth=1)
        ax.axhline(np.mean(errors), color='red', linestyle='--', alpha=0.7)
        ax.axhline(np.mean(errors) + np.std(errors), color='orange', linestyle=':', alpha=0.7)
        ax.axhline(np.mean(errors) - np.std(errors), color='orange', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Image Number')
        ax.set_ylabel('Reprojection Error (pixels)')
        ax.set_title('Per-Image Errors')
        ax.grid(True, alpha=0.3)
        
        # Highlight outliers
        threshold = np.mean(errors) + 2 * np.std(errors)
        outliers = [(i+1, err) for i, err in enumerate(errors) if err > threshold]
        if outliers:
            outlier_x, outlier_y = zip(*outliers)
            ax.scatter(outlier_x, outlier_y, color='red', s=50, marker='x', 
                      label=f'{len(outliers)} outliers')
            ax.legend()
            
    def _plot_distortion_model(self, ax):
        """Visualize distortion coefficients."""
        dist_coeffs = self.distortion_coeffs.flatten()
        
        # Create distortion visualization
        labels = ['k1', 'k2', 'p1', 'p2', 'k3'][:len(dist_coeffs)]
        
        colors = ['red' if abs(x) > 0.1 else 'orange' if abs(x) > 0.01 else 'green' 
                 for x in dist_coeffs]
        
        bars = ax.bar(labels, dist_coeffs, color=colors, alpha=0.7)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_ylabel('Coefficient Value')
        ax.set_title('Distortion Coefficients')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, dist_coeffs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001 * np.sign(height),
                   f'{val:.4f}', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=9)
                   
    def _plot_camera_parameters(self, ax):
        """Plot camera parameter analysis."""
        ax.axis('off')
        
        fx = self.calibration_data['focal_length_x']
        fy = self.calibration_data['focal_length_y']
        cx, cy = self.calibration_data['principal_point']
        img_w, img_h = self.calibration_data['image_size']
        
        # Calculate field of view
        fov_x = 2 * np.arctan(img_w / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(img_h / (2 * fy)) * 180 / np.pi
        
        # Aspect ratio analysis
        aspect_ratio = fx / fy
        principal_offset_x = abs(cx - img_w/2) / img_w * 100
        principal_offset_y = abs(cy - img_h/2) / img_h * 100
        
        params_text = [
            "📐 Camera Parameters",
            "",
            f"🔍 Focal lengths:",
            f"   fx = {fx:.1f} pixels",
            f"   fy = {fy:.1f} pixels",
            f"   Aspect ratio = {aspect_ratio:.4f}",
            "",
            f"🎯 Principal point:",
            f"   cx = {cx:.1f} ({principal_offset_x:.1f}% offset)",
            f"   cy = {cy:.1f} ({principal_offset_y:.1f}% offset)",
            "",
            f"👁️ Field of view:",
            f"   Horizontal: {fov_x:.1f}°",
            f"   Vertical: {fov_y:.1f}°",
        ]
        
        for i, line in enumerate(params_text):
            ax.text(0.05, 0.95 - i * 0.07, line, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   fontweight='bold' if line.startswith('📐') else 'normal')
                   
    def _plot_quality_assessment(self, ax):
        """Plot overall quality assessment."""
        ax.axis('off')
        
        # Quality metrics
        metrics = []
        
        # 1. Reprojection error
        error = self.calibration_data['reprojection_error']
        if error < 0.5:
            error_score = 5
        elif error < 1.0:
            error_score = 4
        elif error < 1.5:
            error_score = 3
        elif error < 2.0:
            error_score = 2
        else:
            error_score = 1
            
        metrics.append(("Reprojection Error", error_score))
        
        # 2. Number of images
        num_images = self.calibration_data['num_images']
        if num_images >= 30:
            image_score = 5
        elif num_images >= 20:
            image_score = 4
        elif num_images >= 15:
            image_score = 3
        elif num_images >= 10:
            image_score = 2
        else:
            image_score = 1
            
        metrics.append(("Image Count", image_score))
        
        # 3. Error consistency
        error_std = self.calibration_data['std_error']
        if error_std < 0.2:
            consistency_score = 5
        elif error_std < 0.5:
            consistency_score = 4
        elif error_std < 1.0:
            consistency_score = 3
        elif error_std < 1.5:
            consistency_score = 2
        else:
            consistency_score = 1
            
        metrics.append(("Error Consistency", consistency_score))
        
        # Plot quality bars
        y_pos = 0.8
        for metric_name, score in metrics:
            # Draw metric name
            ax.text(0.05, y_pos, metric_name, transform=ax.transAxes, fontsize=11)
            
            # Draw score bars
            for i in range(5):
                color = 'green' if i < score else 'lightgray'
                rect = patches.Rectangle((0.6 + i * 0.06, y_pos - 0.02), 0.05, 0.04, 
                                       linewidth=1, edgecolor='black', facecolor=color,
                                       transform=ax.transAxes)
                ax.add_patch(rect)
                
            y_pos -= 0.15
            
        # Overall score
        overall_score = np.mean([score for _, score in metrics])
        ax.text(0.05, 0.2, f"Overall Quality: {overall_score:.1f}/5", 
               transform=ax.transAxes, fontsize=12, fontweight='bold')
        
    def _plot_recommendations(self, ax):
        """Plot calibration recommendations."""
        ax.axis('off')
        
        recommendations = ["💡 Recommendations:"]
        
        # Check various quality indicators
        error = self.calibration_data['reprojection_error']
        num_images = self.calibration_data['num_images']
        error_std = self.calibration_data['std_error']
        
        if error > 1.0:
            recommendations.append("• Recalibrate - high reprojection error")
            recommendations.append("• Check checkerboard detection quality")
            
        if num_images < 15:
            recommendations.append("• Collect more calibration images")
            recommendations.append("• Target 20-30 images minimum")
            
        if error_std > 0.5:
            recommendations.append("• Remove outlier images")
            recommendations.append("• Ensure consistent image quality")
            
        # Check distortion
        if np.max(np.abs(self.distortion_coeffs)) > 0.3:
            recommendations.append("• High distortion detected")
            recommendations.append("• Consider lens correction")
            
        # Check principal point offset
        cx, cy = self.calibration_data['principal_point']
        img_w, img_h = self.calibration_data['image_size']
        offset_x = abs(cx - img_w/2) / img_w * 100
        offset_y = abs(cy - img_h/2) / img_h * 100
        
        if offset_x > 5 or offset_y > 5:
            recommendations.append("• Large principal point offset")
            recommendations.append("• Check camera alignment")
            
        if len(recommendations) == 1:
            recommendations.append("• Calibration looks good!")
            recommendations.append("• Ready for stereo calibration")
            
        for i, rec in enumerate(recommendations):
            ax.text(0.05, 0.95 - i * 0.08, rec, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   fontweight='bold' if rec.startswith('💡') else 'normal')
                   
    def validate_with_test_images(self, test_image_dir: str) -> Dict:
        """
        Validate calibration using test images.
        
        Args:
            test_image_dir: Directory containing test checkerboard images
            
        Returns:
            Dictionary with validation results
        """
        if self.camera_matrix is None:
            raise ValueError("No calibration data loaded")
            
        import glob
        
        test_images = glob.glob(os.path.join(test_image_dir, "*.jpg")) + \
                     glob.glob(os.path.join(test_image_dir, "*.png"))
        
        if not test_images:
            raise ValueError(f"No test images found in {test_image_dir}")
            
        checkerboard_size = tuple(self.calibration_data['checkerboard_size'])
        square_size = self.calibration_data['square_size']
        
        # Prepare object points
        objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard_size[0], 
                              0:checkerboard_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        test_errors = []
        valid_images = 0
        
        for img_path in test_images:
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard
            ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
            
            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # Calculate reprojection error
                # First, solve PnP to get pose
                ret_pnp, rvec, tvec = cv2.solvePnP(objp, corners, 
                                                  self.camera_matrix, self.distortion_coeffs)
                
                if ret_pnp:
                    # Project points
                    projected_points, _ = cv2.projectPoints(objp, rvec, tvec,
                                                          self.camera_matrix, self.distortion_coeffs)
                    
                    # Calculate error
                    error = cv2.norm(corners, projected_points, cv2.NORM_L2) / len(projected_points)
                    test_errors.append(error)
                    valid_images += 1
                    
        if not test_errors:
            raise ValueError("No valid checkerboard patterns found in test images")
            
        results = {
            'test_images_processed': len(test_images),
            'valid_detections': valid_images,
            'test_errors': test_errors,
            'mean_test_error': np.mean(test_errors),
            'std_test_error': np.std(test_errors),
            'calibration_error': self.calibration_data['reprojection_error'],
            'error_difference': abs(np.mean(test_errors) - self.calibration_data['reprojection_error'])
        }
        
        return results
        
    def create_undistortion_comparison(self, image_paths: List[str]) -> plt.Figure:
        """
        Create before/after undistortion comparison.
        
        Args:
            image_paths: List of image paths to show undistortion effect
            
        Returns:
            Figure showing original vs undistorted images
        """
        if self.camera_matrix is None:
            raise ValueError("No calibration data loaded")
            
        n_images = min(len(image_paths), 4)
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
            
        for i in range(n_images):
            image = cv2.imread(image_paths[i])
            undistorted = cv2.undistort(image, self.camera_matrix, self.distortion_coeffs)
            
            # Original
            axes[0, i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0, i].set_title(f'Original {i+1}')
            axes[0, i].axis('off')
            
            # Undistorted
            axes[1, i].imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
            axes[1, i].set_title(f'Undistorted {i+1}')
            axes[1, i].axis('off')
            
        plt.tight_layout()
        return fig


def validate_calibration_cli(calibration_file: str, output_dir: str = "validation_results"):
    """
    Command-line interface for calibration validation.
    
    Args:
        calibration_file: Path to calibration JSON file
        output_dir: Directory to save validation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    validator = CalibrationValidator(calibration_file)
    
    # Create validation report
    report_fig = validator.create_calibration_report()
    report_path = os.path.join(output_dir, "calibration_report.png")
    report_fig.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close(report_fig)
    
    logger.info(f"Validation report saved to {report_path}")
    
    # Print summary to console
    data = validator.calibration_data
    print("\n" + "="*60)
    print("CAMERA CALIBRATION VALIDATION SUMMARY")
    print("="*60)
    print(f"Calibration file: {calibration_file}")
    print(f"Images used: {data['num_images']}")
    print(f"Reprojection error: {data['reprojection_error']:.4f} pixels")
    print(f"Mean ± std error: {data['mean_error']:.4f} ± {data['std_error']:.4f} pixels")
    
    error = data['reprojection_error']
    if error < 0.5:
        print("✅ EXCELLENT calibration quality")
    elif error < 1.0:
        print("✅ GOOD calibration quality")
    elif error < 2.0:
        print("⚠️  FAIR calibration quality - consider recalibrating")
    else:
        print("❌ POOR calibration quality - recalibration recommended")
        
    print(f"\nDetailed report saved to: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate camera calibration results")
    parser.add_argument("calibration_file", help="Path to calibration JSON file")
    parser.add_argument("--output", default="validation_results", 
                       help="Output directory for validation results")
    
    args = parser.parse_args()
    
    validate_calibration_cli(args.calibration_file, args.output)