# Phase 1: Camera Calibration & Stereo Depth

Foundation phase for all UAV perception capabilities, establishing accurate 3D vision systems.

## Learning Objectives

### Core Skills
- **Camera calibration**: Intrinsic and extrinsic parameter estimation
- **Stereo vision**: Depth estimation from binocular camera systems
- **3D reconstruction**: Point cloud generation and processing
- **Coordinate transformations**: Camera, world, and UAV reference frames

### Hands-on Projects
- Single camera calibration with checkerboard patterns
- Stereo camera setup and calibration
- Real-time depth map generation
- 3D point cloud visualization and processing

## Foundation for Air-to-Air Detection

### Critical Contributions
- **Distance estimation**: Essential for drone size/distance discrimination
- **3D tracking**: Spatial tracking of detected targets
- **Multi-camera setup**: Preparation for RGB-thermal calibration
- **Coordinate systems**: UAV-relative positioning for navigation

### Key Measurements
- **Calibration accuracy**: <0.5 pixel reprojection error
- **Depth precision**: ±5% accuracy at 50m range
- **Processing speed**: Real-time depth at 30fps

## Integration Points
- **Phase 2**: Calibrated cameras for accurate detection
- **Phase 4**: Multi-modal camera calibration (RGB-thermal)
- **Phase 6**: Real UAV camera system setup

## Tools & Libraries
- OpenCV calibration functions
- Stereo vision algorithms
- Point cloud libraries (PCL, Open3D)
- ROS camera calibration tools

## Prerequisites
- Basic computer vision concepts
- Python programming
- Linear algebra fundamentals

## Next Phase
**Phase 2: Object Detection & Tracking** - Apply calibrated vision systems to object detection.