# UAV Perception Lab 🚁👁️

A progressive hands-on training repo to build computer vision, sensor fusion, and real-time perception skills for UAVs.

## 📌 Roadmap
This repo follows a 6-phase curriculum with integrated air-to-air drone detection capabilities:
1. **Camera calibration & stereo depth** - Foundation for 3D perception
2. **Object detection & tracking** - From basic objects to drone detection to air-to-air scenarios
3. **Embedded AI acceleration** - Real-time optimization for UAV deployment
4. **Sensor fusion** - EKF/VIO basics + RGB-thermal fusion for enhanced detection
5. **Multi-drone scenarios** - Triangulation + air-to-air detection simulations
6. **Real drone perception tests** - Live validation of all capabilities

## 🧰 Tools
- OpenCV, ROS 2, PX4, AirSim
- PyTorch / YOLOv8, TensorRT / CUDA
- FilterPy (for Kalman/Extended KF)
- VINS-Mono / ORB-SLAM3
- **Air-to-Air Specific**: RGB-thermal fusion, BoTSORT/ByteTrack, edge optimization tools

## 📂 Structure
Each `phaseX_*/` folder contains progressive modules building toward complete UAV perception capabilities:

```
phase2_detection/
├── basic_object_detection/     # YOLO, R-CNN fundamentals
├── drone_detection/           # Drone-specific detection
└── air_to_air_detection/      # Advanced air-to-air scenarios

phase4_sensor_fusion/
├── ekf_vio/                  # Traditional sensor fusion
└── rgb_thermal_fusion/       # Multi-modal for enhanced detection

# ... and similar integrated structure across all phases
```

## 🎯 Progress Tracker

### Phase-wise Progression with Air-to-Air Integration

- [ ] **Phase 1: Calibration & Stereo**
  - [ ] Single camera calibration
  - [ ] Stereo vision setup
  - [ ] 3D reconstruction pipeline

- [ ] **Phase 2: Detection & Tracking**
  - [ ] Basic object detection (YOLO, R-CNN)
  - [ ] Drone-specific detection optimization
  - [ ] **Air-to-air detection pipeline**

- [ ] **Phase 3: Embedded Acceleration**
  - [ ] Model optimization (quantization, pruning)
  - [ ] **Air-to-air model optimization for edge**
  - [ ] Real-time performance validation

- [ ] **Phase 4: Sensor Fusion**
  - [ ] EKF/VIO implementation
  - [ ] **RGB-thermal fusion for enhanced detection**
  - [ ] Multi-modal calibration

- [ ] **Phase 5: Multi-Drone Scenarios**
  - [ ] Triangulation and cooperative detection
  - [ ] **Complex air-to-air scenarios**
  - [ ] Multi-target tracking (MOT)

- [ ] **Phase 6: Real Drone Tests**
  - [ ] General UAV perception deployment
  - [ ] **Live air-to-air validation**
  - [ ] Operational readiness assessment

## 🎯 Air-to-Air Detection Integration

This lab features **integrated air-to-air drone detection** as an advanced specialization woven throughout the curriculum:

- **Progressive complexity**: From basic detection to advanced air-to-air scenarios
- **Multi-modal sensing**: RGB-thermal fusion for enhanced capability  
- **Edge optimization**: Real-time performance on UAV hardware
- **Complete pipeline**: Data collection → training → deployment → validation

Each phase builds toward a complete air-to-air detection system. See individual `phaseX_*/air_*` modules and [docs/air-to-air-detection.md](docs/air-to-air-detection.md) for details.

## 📚 Resources
Check out [docs/resources.md](docs/resources.md) for curated learning materials including SLAM tutorials, research papers, and useful datasets.
