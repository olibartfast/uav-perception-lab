# Phase 1: Single Camera Calibration - Setup Instructions

## 📋 Prerequisites

### System Requirements
- Python 3.7 or higher
- Camera or webcam
- Printed checkerboard calibration pattern

### Required Python Packages
```bash
pip install opencv-python numpy matplotlib seaborn
```

### Optional Packages (for enhanced functionality)
```bash
pip install scikit-image plotly
```

## 🖨️ Checkerboard Pattern Setup

### Download and Print Checkerboard
1. **Recommended pattern**: 9×6 or 8×6 internal corners
2. **Square size**: 25mm or 1 inch squares
3. **Print settings**: 
   - High quality, no scaling
   - Ensure squares are perfectly square
   - Mount on rigid, flat surface

### Pattern Sources
- [OpenCV Checkerboard Generator](https://docs.opencv.org/4.x/da/d0d/tutorial_camera_calibration_pattern.html)
- Generate custom patterns with: `python -c "import cv2; cv2.imwrite('checkerboard.png', cv2.imread('pattern.png'))"`

## 🔧 Hardware Setup

### Camera Requirements
- **Resolution**: Minimum 640×480, recommended 1280×720 or higher
- **Focus**: Manual focus preferred to avoid focus changes during calibration
- **Stability**: Stable mounting or tripod for consistent images
- **Lighting**: Even, diffuse lighting without shadows or reflections

### Optimal Setup
```
Camera → Checkerboard Distance: 0.5m - 2.0m
Lighting: Diffuse, even illumination
Background: Neutral, non-reflective
Stability: Tripod or stable surface mounting
```

## 📁 Directory Structure

After setup, your Phase 1 directory should look like:
```
phase1_calibration/
├── README.md                          # This documentation
├── requirements.txt                   # Python dependencies
├── single_camera_calibration.py       # Core calibration implementation
├── collect_calibration_data.py        # Interactive data collection
├── calibrate_camera.py               # Complete pipeline script
├── calibration_validation.py         # Validation and visualization
├── calibration_tutorial.py           # Educational tutorial
├── calibration_images/               # Your captured images (created during use)
├── results/                          # Calibration outputs (created during use)
└── examples/                         # Example data and scripts
    ├── sample_calibration.json       # Example calibration file
    └── demo_images/                  # Sample calibration images
```

## 🚀 Installation and Verification

### 1. Install Dependencies
```bash
cd phase1_calibration
pip install -r requirements.txt
```

### 2. Verify Camera Access
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

### 3. Test Checkerboard Detection
```bash
python collect_calibration_data.py --camera 0
# Should open camera view with checkerboard detection feedback
# Press 'q' to quit test
```

### 4. Run Tutorial (Optional)
```bash
python calibration_tutorial.py
```

## 🎯 Quick Validation

### Test with Sample Data
If you have sample calibration images:
```bash
python calibrate_camera.py --images examples/demo_images/ --output test_calibration.json --validate
```

Expected output:
- Reprojection error < 1.0 pixels
- Reasonable focal length values
- Principal point near image center
- Validation report generated

## ⚠️ Common Setup Issues

### Camera Not Detected
```bash
# Test different camera IDs
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not available')
"
```

### Permission Issues (Linux/Mac)
```bash
# Add user to video group (Linux)
sudo usermod -a -G video $USER
# Log out and back in

# Camera permissions (Mac)
# Grant camera access in System Preferences → Security & Privacy → Camera
```

### Dependencies Issues
```bash
# Update pip
pip install --upgrade pip

# Install with specific versions if needed
pip install opencv-python==4.8.1.78 numpy==1.24.3 matplotlib==3.7.1
```

## 📊 Verification Checklist

- [ ] Python 3.7+ installed
- [ ] Required packages installed successfully
- [ ] Camera accessible and working
- [ ] Checkerboard pattern printed and mounted
- [ ] Can run data collection script
- [ ] Can detect checkerboard pattern
- [ ] Tutorial runs without errors

## 🆘 Getting Help

### Documentation
- Check individual script help: `python script_name.py --help`
- Review error messages carefully
- Consult OpenCV documentation for camera issues

### Common Commands
```bash
# Test camera
python collect_calibration_data.py --help

# Check calibration
python calibrate_camera.py --help

# Validate results
python calibration_validation.py --help
```

## ➡️ Next Steps

Once setup is complete:
1. **Run Tutorial**: `python calibration_tutorial.py`
2. **Collect Data**: `python collect_calibration_data.py`
3. **Perform Calibration**: `python calibrate_camera.py --images calibration_images/`
4. **Validate Results**: Review output and validation report

Ready to proceed to actual calibration! 🎉