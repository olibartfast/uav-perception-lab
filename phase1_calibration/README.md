# Phase 1: Camera Calibration & Stereo Depth

Foundation phase for all UAV perception capabilities, establishing accurate 3D vision systems through comprehensive camera calibration.

## 🎯 Learning Objectives

### Core Skills
- **Camera calibration**: Intrinsic and extrinsic parameter estimation using checkerboard patterns
- **Distortion correction**: Understanding and compensating for lens distortion effects
- **Quality assessment**: Evaluating calibration accuracy and identifying issues
- **3D vision foundations**: Preparing for stereo vision and depth estimation

### Practical Implementation
This phase includes complete implementation with:
- Interactive data collection tools
- Automated calibration pipeline
- Comprehensive validation and visualization
- Educational tutorials and examples

## 📁 Module Contents

### Core Implementation Files

#### `single_camera_calibration.py`
Comprehensive camera calibration implementation:
- `SingleCameraCalibrator` class for complete calibration workflow
- Checkerboard detection with sub-pixel accuracy
- Camera parameter estimation using OpenCV
- Quality metrics and error analysis
- Results saving/loading in JSON format

#### `collect_calibration_data.py`
Interactive data collection tool:
- Real-time checkerboard detection feedback
- Quality assessment during capture
- Visual guidance for optimal image diversity
- Automatic image naming and organization

#### `calibration_validation.py`
Advanced validation and visualization tools:
- Comprehensive calibration quality reports
- Error distribution analysis
- Visual distortion assessment
- Undistortion before/after comparisons

#### `calibrate_camera.py`
Complete calibration pipeline script:
- Command-line interface for batch processing
- Automated calibration from image directory
- Integrated validation and reporting
- Quality assessment and recommendations

#### `calibration_tutorial.py`
Interactive educational tutorial covering:
- Pinhole camera model fundamentals
- Lens distortion theory and effects
- Checkerboard calibration methodology
- Quality assessment techniques

## 🚀 Quick Start Guide

### 1. Data Collection
Collect calibration images using the interactive tool:
```bash
python collect_calibration_data.py --camera 0 --output calibration_images
```

**Tips for good calibration images:**
- Capture 20-30 images minimum
- Vary checkerboard positions and angles
- Include images at different distances
- Ensure good lighting and sharp focus
- Cover the entire image area
- Include some tilted/angled views

### 2. Perform Calibration
Run the complete calibration pipeline:
```bash
python calibrate_camera.py --images calibration_images/ --output camera_calibration.json --validate --show-undistortion
```

### 3. Validate Results
Generate comprehensive validation report:
```bash
python calibration_validation.py camera_calibration.json --output validation_results/
```

### 4. Learn the Theory
Run the interactive tutorial:
```bash
python calibration_tutorial.py
```

## 📊 Quality Assessment

### Reprojection Error Guidelines
- **< 0.5 pixels**: Excellent calibration quality
- **0.5 - 1.0 pixels**: Good calibration quality
- **1.0 - 2.0 pixels**: Fair quality, consider recalibrating
- **> 2.0 pixels**: Poor quality, recalibration required

### Key Quality Indicators
1. **Low reprojection error**: Overall accuracy measure
2. **Consistent per-image errors**: No significant outliers
3. **Reasonable parameters**: fx ≈ fy, principal point near center
4. **Visual validation**: Undistorted images look natural

## 🔧 Advanced Usage

### Custom Checkerboard Sizes
```bash
# For 7x5 checkerboard
python collect_calibration_data.py --checkerboard 7x5

# For different square size (30mm)
python calibrate_camera.py --images calibration_images/ --square-size 30.0
```

### Programmatic Usage
```python
from single_camera_calibration import SingleCameraCalibrator

# Create calibrator
calibrator = SingleCameraCalibrator(checkerboard_size=(9, 6), square_size=25.0)

# Add images
for image in calibration_images:
    calibrator.add_calibration_image(image)

# Perform calibration
results = calibrator.calibrate()

# Save results
calibrator.save_calibration("camera_calibration.json")

# Undistort new images
undistorted = calibrator.undistort_image(new_image)
```

## 🎓 Educational Resources

### Theory Background
- **Pinhole camera model**: Mathematical foundation of perspective projection
- **Lens distortion**: Radial and tangential distortion effects
- **Calibration mathematics**: Parameter estimation through optimization
- **Quality metrics**: Understanding reprojection error and validation

### Practical Skills
- **Data collection best practices**: How to capture effective calibration images
- **Quality assessment**: Interpreting calibration results and identifying issues
- **Troubleshooting**: Common problems and solutions
- **Integration preparation**: Setting up for stereo calibration

## 📈 Foundation for Air-to-Air Detection

### Critical Contributions to UAV Perception
- **Accurate distance estimation**: Essential for drone size/distance discrimination
- **3D target tracking**: Spatial tracking of detected aerial targets
- **Multi-camera setup**: Foundation for RGB-thermal sensor fusion
- **Coordinate transformations**: UAV-relative positioning for navigation

### Integration with Later Phases
- **Phase 2**: Calibrated cameras provide accurate detection geometry
- **Phase 4**: Multi-modal camera calibration (RGB-thermal fusion)
- **Phase 6**: Real UAV camera system deployment

## 🔍 Troubleshooting

### Common Issues

#### Poor Detection Quality
- **Cause**: Blurry images, poor lighting, wrong checkerboard size
- **Solution**: Improve lighting, ensure sharp focus, verify checkerboard dimensions

#### High Reprojection Error
- **Cause**: Insufficient images, poor image diversity, detection errors
- **Solution**: Collect more images, vary positions/angles, review detection quality

#### Unreasonable Parameters
- **Cause**: Incorrect checkerboard size, systematic detection errors
- **Solution**: Verify checkerboard specifications, check corner detection accuracy

#### Inconsistent Errors
- **Cause**: Outlier images with poor detection quality
- **Solution**: Remove outlier images, ensure consistent image quality

## 📋 Prerequisites
- Python 3.7+ with OpenCV, NumPy, Matplotlib
- Camera or webcam for data collection
- Printed checkerboard pattern (recommended: 9×6 or 8×6)
- Basic understanding of computer vision concepts

## ➡️ Next Phase
**Phase 2: Object Detection & Tracking** - Apply calibrated vision systems to detect and track objects, including drones and aerial targets.

## 📚 Additional Resources
- OpenCV Camera Calibration Documentation
- Multiple View Geometry (Hartley & Zisserman)
- Computer Vision: Algorithms and Applications (Szeliski)
- [Phase 1 Exercises and Datasets](exercises/)