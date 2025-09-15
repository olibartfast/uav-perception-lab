# 📚 Learning Resources

A curated collection of educational materials for UAV perception, computer vision, and SLAM.

## 🎥 Video Tutorials

### SLAM & Localization
- **[Simultaneous Localization & Mapping: Which SLAM Is For You?](https://www.youtube.com/live/PDqbsQcUE7k)** by Ali Pahlevani
  - Comprehensive overview of different SLAM approaches
  - Helps choose the right SLAM method for your application
  - Relevant for Phase 4: Sensor Fusion (VIO/VINS, ORB-SLAM3)

### Computer Vision & Perception
- **[Air-to-Air Drone Detection System](air-to-air-detection.md)**: Comprehensive guide to detecting drones from UAV platforms
- **Small Object Detection**: Specialized techniques for detecting small targets in aerial imagery
- **Multi-Modal Sensor Fusion**: RGB and thermal camera integration for enhanced detection

### UAV & Robotics
- **Edge AI Deployment**: Optimizing deep learning models for onboard UAV computing
- **Real-time Processing**: Balancing accuracy and speed for flight-critical applications

## 📄 Research Papers

### Air-to-Air Drone Detection
- **[CF2Drone: Attention-Enhanced Cross-Feature Fusion for Small Object Detection](papers/2404.19276v1.pdf)**
  - Advanced techniques for drone detection in challenging conditions
  - Relevant for Phase 2: Object Detection & Tracking
  
- **[TransViDrone: Spatio-Temporal Transformer for Video-based Drone Detection](papers/2210.08423v2.pdf)**
  - Transformer-based approaches for video drone detection
  - Temporal consistency in detection across video frames
  - Relevant for Phase 2: Object Detection & Tracking

- **[DogFight: Detecting Drones from Drones Videos](papers/2103.17242v2.pdf)**
  - Air-to-air combat scenarios and detection strategies
  - Real-world drone-to-drone detection challenges
  - Relevant for Phase 5: Multi-Drone Scenarios

- **[Extended Air-to-Air Detection Methods](papers/2306.16175v3.pdf)**
  - Comparative studies and extended methodologies
  - Advanced fusion techniques for multi-modal detection

### SLAM & VIO
*Add foundational SLAM papers here*

### Object Detection & Tracking
- **General Object Detection**: YOLO series, R-CNN family
- **Small Object Detection**: Feature pyramid networks, attention mechanisms
- **Multi-Object Tracking**: BoTSORT, ByteTrack, DeepSORT

### Sensor Fusion
- **RGB-Thermal Fusion**: Early, feature-level, and decision-level fusion methods
- **Multi-Modal Detection**: Combining visible and infrared spectra
- **EKF & Kalman Filtering**: State estimation for tracking applications

## 📖 Books & Documentation

### Computer Vision
*Add OpenCV guides, CV textbooks*

### Robotics & UAVs
*Add robotics perception books*

## 🔗 Useful Datasets

### Air-to-Air Detection Datasets
- **Synthetic Drone Data**: Simulation-generated datasets with controlled conditions
- **Real-world UAV Data**: Field-collected RGB and thermal drone detection data
- **Multi-modal Datasets**: Combined RGB-thermal annotations for fusion research

### SLAM Datasets
*Add EuRoC, TUM datasets*

### Detection Datasets
- **VisDrone**: Large-scale drone detection benchmark
- **COCO**: General object detection with small object challenges
- **Custom Drone Datasets**: Domain-specific air-to-air detection collections

---
*💡 Tip: When adding new resources, include a brief description and relevance to specific project phases.*