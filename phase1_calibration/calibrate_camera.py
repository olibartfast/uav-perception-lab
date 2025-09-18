#!/usr/bin/env python3
"""
Complete Camera Calibration Pipeline
Phase 1: Camera Calibration & Stereo Depth

This script demonstrates the complete workflow for camera calibration:
1. Load calibration images
2. Perform calibration
3. Validate results
4. Save calibration data

Usage:
    python calibrate_camera.py --images calibration_images/ --output camera_calibration.json
"""

import argparse
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from single_camera_calibration import calibrate_camera_from_images, SingleCameraCalibrator
from calibration_validation import CalibrationValidator

def main():
    parser = argparse.ArgumentParser(description="Perform camera calibration from checkerboard images")
    parser.add_argument("--images", required=True, 
                       help="Directory containing calibration images")
    parser.add_argument("--output", default="camera_calibration.json",
                       help="Output file for calibration results")
    parser.add_argument("--checkerboard", default="9x6",
                       help="Checkerboard size as WxH (default: 9x6)")
    parser.add_argument("--square-size", type=float, default=25.0,
                       help="Checkerboard square size in mm (default: 25.0)")
    parser.add_argument("--validate", action="store_true",
                       help="Create validation report")
    parser.add_argument("--show-undistortion", action="store_true",
                       help="Show undistortion examples")
    
    args = parser.parse_args()
    
    # Parse checkerboard size
    try:
        w, h = map(int, args.checkerboard.split('x'))
        checkerboard_size = (w, h)
    except ValueError:
        print("Error: Checkerboard size must be in format WxH (e.g., 9x6)")
        return 1
    
    # Check if images directory exists
    if not os.path.exists(args.images):
        print(f"Error: Images directory '{args.images}' does not exist")
        return 1
    
    print("="*60)
    print("CAMERA CALIBRATION PIPELINE")
    print("="*60)
    print(f"Images directory: {args.images}")
    print(f"Checkerboard size: {checkerboard_size[0]}×{checkerboard_size[1]}")
    print(f"Square size: {args.square_size} mm")
    print(f"Output file: {args.output}")
    print()
    
    try:
        # Perform calibration
        print("Starting calibration...")
        results = calibrate_camera_from_images(
            args.images,
            checkerboard_size=checkerboard_size,
            square_size=args.square_size,
            save_path=args.output
        )
        
        # Print results
        print("\n" + "="*40)
        print("CALIBRATION RESULTS")
        print("="*40)
        print(f"Images processed: {results['num_images']}")
        print(f"Reprojection error: {results['reprojection_error']:.4f} pixels")
        print(f"Mean error: {results['mean_error']:.4f} ± {results['std_error']:.4f} pixels")
        print(f"Focal length: fx={results['focal_length_x']:.1f}, fy={results['focal_length_y']:.1f}")
        print(f"Principal point: ({results['principal_point'][0]:.1f}, {results['principal_point'][1]:.1f})")
        
        # Quality assessment
        error = results['reprojection_error']
        if error < 0.5:
            print("✅ EXCELLENT calibration quality!")
        elif error < 1.0:
            print("✅ GOOD calibration quality")
        elif error < 2.0:
            print("⚠️  FAIR calibration quality - consider recalibrating with more/better images")
        else:
            print("❌ POOR calibration quality - recalibration strongly recommended")
        
        print(f"\nCalibration saved to: {args.output}")
        
        # Validation report
        if args.validate:
            print("\nGenerating validation report...")
            validator = CalibrationValidator(args.output)
            
            # Create validation report
            report_fig = validator.create_calibration_report()
            report_path = args.output.replace('.json', '_validation_report.png')
            report_fig.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close(report_fig)
            print(f"Validation report saved to: {report_path}")
        
        # Show undistortion examples
        if args.show_undistortion:
            print("\nCreating undistortion examples...")
            
            # Load some sample images
            import glob
            sample_images = glob.glob(os.path.join(args.images, "*.jpg"))[:3]
            if not sample_images:
                sample_images = glob.glob(os.path.join(args.images, "*.png"))[:3]
            
            if sample_images:
                validator = CalibrationValidator(args.output)
                undist_fig = validator.create_undistortion_comparison(sample_images)
                undist_path = args.output.replace('.json', '_undistortion_examples.png')
                undist_fig.savefig(undist_path, dpi=300, bbox_inches='tight')
                plt.close(undist_fig)
                print(f"Undistortion examples saved to: {undist_path}")
            else:
                print("No sample images found for undistortion demonstration")
        
        print("\n✅ Calibration completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Calibration failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())