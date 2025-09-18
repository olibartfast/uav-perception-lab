"""
Camera Calibration Tutorial
Phase 1: Camera Calibration & Stereo Depth

This interactive tutorial demonstrates camera calibration concepts and provides
hands-on examples for learning the calibration process.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import tempfile
from single_camera_calibration import SingleCameraCalibrator

class CalibrationTutorial:
    """Interactive camera calibration tutorial."""
    
    def __init__(self):
        self.calibrator = SingleCameraCalibrator()
        
    def lesson_1_camera_model(self):
        """Lesson 1: Understanding the pinhole camera model."""
        print("="*60)
        print("LESSON 1: PINHOLE CAMERA MODEL")
        print("="*60)
        print()
        print("The pinhole camera model describes how 3D world points project to 2D image points.")
        print()
        print("Key concepts:")
        print("• Focal length (f): Distance from lens to image sensor")
        print("• Principal point (cx, cy): Optical center of the image")
        print("• Image coordinates: Pixel positions in the image")
        print("• World coordinates: 3D positions in real world")
        print()
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pinhole camera diagram
        ax1.set_xlim(-4, 4)
        ax1.set_ylim(-3, 3)
        ax1.set_aspect('equal')
        
        # Draw camera
        camera = Rectangle((-0.5, -1), 1, 2, fill=False, linewidth=2)
        ax1.add_patch(camera)
        ax1.text(0, 0, 'Camera', ha='center', va='center', fontweight='bold')
        
        # Draw image plane
        ax1.plot([2, 2], [-2, 2], 'k-', linewidth=3, label='Image plane')
        ax1.text(2.2, 0, 'Image\nPlane', ha='left', va='center')
        
        # Draw 3D point
        ax1.plot(-3, 1.5, 'ro', markersize=10, label='3D point')
        ax1.text(-3, 1.8, '3D Point\n(X, Y, Z)', ha='center', va='bottom')
        
        # Draw projection
        ax1.plot([-3, 2], [1.5, 0.75], 'r--', alpha=0.7)
        ax1.plot(2, 0.75, 'ro', markersize=8)
        ax1.text(2.2, 0.75, '2D point\n(u, v)', ha='left', va='center')
        
        # Draw focal length
        ax1.plot([0, 2], [0, 0], 'b-', linewidth=2)
        ax1.text(1, -0.3, 'f (focal length)', ha='center', va='top', color='blue')
        
        ax1.set_title('Pinhole Camera Model')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Camera matrix visualization
        ax2.axis('off')
        ax2.text(0.1, 0.9, 'Camera Matrix (K):', fontsize=14, fontweight='bold', transform=ax2.transAxes)
        ax2.text(0.1, 0.7, 'K = [fx  0  cx]', fontsize=12, family='monospace', transform=ax2.transAxes)
        ax2.text(0.1, 0.6, '    [ 0 fy  cy]', fontsize=12, family='monospace', transform=ax2.transAxes)
        ax2.text(0.1, 0.5, '    [ 0  0   1]', fontsize=12, family='monospace', transform=ax2.transAxes)
        ax2.text(0.1, 0.3, 'Where:', fontsize=12, transform=ax2.transAxes)
        ax2.text(0.1, 0.2, '• fx, fy: focal lengths in pixels', fontsize=10, transform=ax2.transAxes)
        ax2.text(0.1, 0.15, '• cx, cy: principal point coordinates', fontsize=10, transform=ax2.transAxes)
        ax2.text(0.1, 0.05, 'Goal: Estimate these parameters from images!', 
                fontsize=12, fontweight='bold', color='red', transform=ax2.transAxes)
        
        plt.tight_layout()
        plt.show()
        
        input("\nPress Enter to continue to Lesson 2...")
        
    def lesson_2_distortion(self):
        """Lesson 2: Understanding lens distortion."""
        print("\n" + "="*60)
        print("LESSON 2: LENS DISTORTION")
        print("="*60)
        print()
        print("Real camera lenses introduce distortion that deviates from the ideal pinhole model.")
        print()
        print("Types of distortion:")
        print("• Radial distortion: Barrel/pincushion effects")
        print("• Tangential distortion: Lens manufacturing imperfections")
        print()
        
        # Create distortion examples
        self._demonstrate_distortion_effects()
        
        print("Distortion coefficients:")
        print("• k1, k2, k3: Radial distortion coefficients")
        print("• p1, p2: Tangential distortion coefficients")
        print()
        print("Calibration estimates these coefficients to undistort images.")
        
        input("\nPress Enter to continue to Lesson 3...")
        
    def _demonstrate_distortion_effects(self):
        """Demonstrate distortion effects visually."""
        # Create ideal grid
        grid_size = 200
        x = np.linspace(-1, 1, 21)
        y = np.linspace(-1, 1, 21)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original grid
        axes[0].set_title('Ideal (No Distortion)')
        for xi in x:
            axes[0].plot([xi, xi], [-1, 1], 'b-', alpha=0.7)
        for yi in y:
            axes[0].plot([-1, 1], [yi, yi], 'b-', alpha=0.7)
        axes[0].set_xlim(-1.2, 1.2)
        axes[0].set_ylim(-1.2, 1.2)
        axes[0].set_aspect('equal')
        
        # Barrel distortion
        axes[1].set_title('Barrel Distortion (k1 < 0)')
        xx, yy = np.meshgrid(x, y)
        r2 = xx**2 + yy**2
        k1 = -0.3
        distort_factor = 1 + k1 * r2
        xx_dist = xx * distort_factor
        yy_dist = yy * distort_factor
        
        for i in range(len(x)):
            axes[1].plot(xx_dist[i, :], yy_dist[i, :], 'r-', alpha=0.7)
            axes[1].plot(xx_dist[:, i], yy_dist[:, i], 'r-', alpha=0.7)
        axes[1].set_xlim(-1.2, 1.2)
        axes[1].set_ylim(-1.2, 1.2)
        axes[1].set_aspect('equal')
        
        # Pincushion distortion
        axes[2].set_title('Pincushion Distortion (k1 > 0)')
        k1 = 0.3
        distort_factor = 1 + k1 * r2
        xx_dist = xx * distort_factor
        yy_dist = yy * distort_factor
        
        for i in range(len(x)):
            axes[2].plot(xx_dist[i, :], yy_dist[i, :], 'g-', alpha=0.7)
            axes[2].plot(xx_dist[:, i], yy_dist[:, i], 'g-', alpha=0.7)
        axes[2].set_xlim(-1.2, 1.2)
        axes[2].set_ylim(-1.2, 1.2)
        axes[2].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
        
    def lesson_3_checkerboard_calibration(self):
        """Lesson 3: Checkerboard calibration process."""
        print("\n" + "="*60)
        print("LESSON 3: CHECKERBOARD CALIBRATION")
        print("="*60)
        print()
        print("Checkerboard patterns provide known 3D-2D point correspondences for calibration.")
        print()
        print("Why checkerboards?")
        print("• Easy to detect corners automatically")
        print("• Known 3D geometry (flat board with regular spacing)")
        print("• High contrast features for accurate corner detection")
        print("• Multiple points per image for robust estimation")
        print()
        
        # Demonstrate checkerboard detection
        self._demonstrate_checkerboard_detection()
        
        print("Calibration process:")
        print("1. Capture multiple images of checkerboard from different angles")
        print("2. Detect corner points in each image")
        print("3. Establish 3D-2D point correspondences")
        print("4. Solve for camera parameters using optimization")
        print("5. Validate results and assess quality")
        
        input("\nPress Enter to continue to Lesson 4...")
        
    def _demonstrate_checkerboard_detection(self):
        """Demonstrate checkerboard corner detection."""
        # Generate synthetic checkerboard
        board_size = (8, 6)
        square_size = 30
        
        # Create checkerboard pattern
        board_img = np.zeros((board_size[1] * square_size, board_size[0] * square_size), dtype=np.uint8)
        for i in range(board_size[1]):
            for j in range(board_size[0]):
                if (i + j) % 2 == 0:
                    start_y = i * square_size
                    end_y = (i + 1) * square_size
                    start_x = j * square_size
                    end_x = (j + 1) * square_size
                    board_img[start_y:end_y, start_x:end_x] = 255
        
        # Detect corners
        ret, corners = cv2.findChessboardCorners(board_img, (board_size[0]-1, board_size[1]-1), None)
        
        # Visualize
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original checkerboard
        ax1.imshow(board_img, cmap='gray')
        ax1.set_title('Checkerboard Pattern')
        ax1.axis('off')
        
        # With detected corners
        board_with_corners = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
        if ret:
            cv2.drawChessboardCorners(board_with_corners, (board_size[0]-1, board_size[1]-1), corners, ret)
        
        ax2.imshow(cv2.cvtColor(board_with_corners, cv2.COLOR_BGR2RGB))
        ax2.set_title(f'Detected Corners ({len(corners)} points)')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def lesson_4_quality_assessment(self):
        """Lesson 4: Assessing calibration quality."""
        print("\n" + "="*60)
        print("LESSON 4: CALIBRATION QUALITY ASSESSMENT")
        print("="*60)
        print()
        print("How to evaluate if your calibration is good:")
        print()
        print("1. REPROJECTION ERROR:")
        print("   • Measures how well the calibrated model fits the data")
        print("   • Lower is better (< 0.5 pixels = excellent, < 1.0 = good)")
        print("   • Calculated by projecting 3D points back to image plane")
        print()
        print("2. ERROR CONSISTENCY:")
        print("   • All images should have similar errors")
        print("   • Large variation indicates outlier images")
        print("   • Remove images with much higher errors")
        print()
        print("3. PARAMETER REASONABLENESS:")
        print("   • Focal lengths should be similar (fx ≈ fy for square pixels)")
        print("   • Principal point near image center")
        print("   • Distortion coefficients reasonable magnitude")
        print()
        print("4. VISUAL INSPECTION:")
        print("   • Undistorted images should look natural")
        print("   • Straight lines should remain straight after undistortion")
        print("   • No obvious warping artifacts")
        
        input("\nPress Enter to continue to hands-on exercise...")
        
    def hands_on_exercise(self):
        """Interactive hands-on calibration exercise."""
        print("\n" + "="*60)
        print("HANDS-ON EXERCISE: CAMERA CALIBRATION")
        print("="*60)
        print()
        print("Let's practice with a simulated calibration scenario!")
        print()
        
        # Generate synthetic calibration data
        self._generate_synthetic_calibration_data()
        
        print("In a real scenario, you would:")
        print("1. Run: python collect_calibration_data.py")
        print("2. Capture 20-30 checkerboard images from different angles")
        print("3. Run: python calibrate_camera.py --images calibration_images/")
        print("4. Run: python calibration_validation.py camera_calibration.json")
        print()
        print("This completes the single camera calibration tutorial!")
        print("Next: Proceed to stereo vision calibration")
        
    def _generate_synthetic_calibration_data(self):
        """Generate and demonstrate synthetic calibration."""
        print("Generating synthetic calibration data...")
        
        # Simulate camera parameters
        true_fx, true_fy = 800, 805
        true_cx, true_cy = 320, 240
        true_k1, true_k2 = -0.1, 0.02
        
        # Generate object points (checkerboard)
        board_size = (9, 6)
        square_size = 25.0
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Simulate multiple views
        object_points = []
        image_points = []
        
        for i in range(15):  # 15 synthetic images
            # Random camera pose
            rvec = np.random.randn(3) * 0.3
            tvec = np.array([0, 0, 500]) + np.random.randn(3) * 50
            
            # Project points
            camera_matrix = np.array([[true_fx, 0, true_cx],
                                    [0, true_fy, true_cy],
                                    [0, 0, 1]], dtype=np.float32)
            dist_coeffs = np.array([true_k1, true_k2, 0, 0, 0], dtype=np.float32)
            
            projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
            
            # Add noise
            projected += np.random.randn(*projected.shape) * 0.5
            
            object_points.append(objp)
            image_points.append(projected)
        
        # Perform calibration
        image_size = (640, 480)
        ret, camera_matrix_est, dist_coeffs_est, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None
        )
        
        # Show results
        print(f"\nSynthetic Calibration Results:")
        print(f"Reprojection error: {ret:.3f} pixels")
        print(f"True focal length: fx={true_fx}, fy={true_fy}")
        print(f"Estimated focal length: fx={camera_matrix_est[0,0]:.1f}, fy={camera_matrix_est[1,1]:.1f}")
        print(f"Error in fx: {abs(camera_matrix_est[0,0] - true_fx):.1f} pixels")
        print(f"Error in fy: {abs(camera_matrix_est[1,1] - true_fy):.1f} pixels")
        print(f"True distortion k1: {true_k1:.4f}")
        print(f"Estimated distortion k1: {dist_coeffs_est[0]:.4f}")


def run_tutorial():
    """Run the complete calibration tutorial."""
    tutorial = CalibrationTutorial()
    
    print("Welcome to the Camera Calibration Tutorial!")
    print("This tutorial will teach you the fundamentals of camera calibration.")
    print()
    input("Press Enter to start...")
    
    # Run lessons
    tutorial.lesson_1_camera_model()
    tutorial.lesson_2_distortion()
    tutorial.lesson_3_checkerboard_calibration()
    tutorial.lesson_4_quality_assessment()
    tutorial.hands_on_exercise()
    
    print("\n🎉 Tutorial completed! You now understand camera calibration fundamentals.")


if __name__ == "__main__":
    run_tutorial()