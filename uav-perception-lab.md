```
uav-perception-lab/
│
├── phase1_calibration_stereo/
│   ├── calibration/         # Camera calibration scripts + sample checkerboard images
│   ├── stereo_depth/        # Stereo vision & depth estimation
│   └── README.md
│
├── phase2_detection_tracking/
│   ├── detection/           # YOLOv8 training + inference scripts
│   ├── tracking/            # Kalman Filter, DeepSORT tracking
│   └── README.md
│
├── phase3_embedded_acceleration/
│   ├── yolov8_tensorrt/     # TensorRT export + benchmarks
│   ├── jetson_setup.md      # Notes on Jetson Nano/Xavier/Orin setup
│   └── README.md
│
├── phase4_sensor_fusion/
│   ├── ekf_tracking/        # Extended Kalman Filter for target tracking
│   ├── vio_vins/            # VINS-Mono or ORB-SLAM3 experiments
│   └── README.md
│
├── phase5_multi_drone_sim/
│   ├── airsim_px4/          # AirSim + PX4 multi-drone sim setup
│   ├── triangulation/       # Epipolar geometry, multi-drone localization
│   └── README.md
│
├── phase6_real_drone_tests/
│   ├── tello_tracking/      # DJI Tello or Parrot drone tracking pipeline
│   ├── ros2_px4/            # ROS2 integration with PX4
│   └── README.md
│
├── datasets/                # Links / scripts to download datasets (VisDrone, EuRoC, checkerboards)
├── docs/
│   ├── roadmap.md           # The 12-week roadmap
│   └── resources.md         # Books, papers, tutorials
│
└── README.md                # Project overview + progress tracker
```
