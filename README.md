# CVproject — Real-Time Drone Tracking & 3D Trajectory Reconstruction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-contrib-green)](https://opencv.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-yellow)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time computer vision system that detects drones, tracks them across frames, and reconstructs their 3D trajectory using an Intel RealSense depth camera. Built on YOLO for detection and a Kalman filter for smooth 3D state estimation.

---

## 🎥 Demo

![Dashboard](docs/demo.gif)

*Real-time dashboard: YOLO detection, 3D coordinates, and live trajectory projections.*

---

## ✨ Features

- **Real-time drone detection** using a fine-tuned YOLO model
- **Object tracking** with CSRT / KCF (via `opencv-contrib`) to survive short detection losses
- **Depth-based 3D localization** from Intel RealSense D435i (color + depth)
- **Kalman filter (3D)** for smooth position and velocity estimation
- **Live trajectory visualization** — top / front / side projections
- **3D trajectory export** to PNG (`matplotlib`)
- **Outlier rejection** — jump rejection + median smoothing
- **Dashboard UI** with FPS, tracking status, confidence, and 3D coordinates
- **Keyboard controls** for recording, pausing, zeroing, and exporting

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Intel RealSense D435i (or compatible) with `pyrealsense2` support
- GPU recommended (for real-time YOLO inference)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hlatar/CVproject.git
   cd CVproject
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt`:
   ```txt
   numpy
   opencv-contrib-python
   matplotlib
   ultralytics
   pyrealsense2
   huggingface_hub
   ```

4. **Download model weights** (automatic on first run, or manually):
   ```bash
   huggingface-cli download Hlatar/CVproject-drone-weights best.pt --local-dir .
   ```

5. **Run:**
   ```bash
   python main.py
   ```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `R` | Start / stop trajectory recording (saves plot on stop) |
| `S` | Save current trajectory as PNG |
| `C` | Clear trajectory |
| `P` | Pause / resume video stream |
| `O` | Set current position as origin (zero) |
| `B` | Reset origin |
| `ESC` | Exit |

---

## 🧠 How It Works

```
┌──────────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│  RealSense   │───▶│   YOLO   │───▶│  Depth →  │───▶│ Kalman   │
│  RGB+Depth   │    │ Detector │    │  3D point │    │ Filter3D │
└──────────────┘    └──────────┘    └───────────┘    └──────────┘
                          │                                  │
                          ▼                                  ▼
                    ┌──────────┐                     ┌──────────────┐
                    │ Tracker  │                     │  Trajectory  │
                    │ CSRT/KCF │                     │ + Smoothing  │
                    └──────────┘                     └──────────────┘
```

1. **Detection** — YOLO runs on the color frame and finds the best `drone` bounding box.
2. **Tracking** — a CSRT tracker maintains the object across frames when YOLO confidence drops. IoU gate suppresses false positives.
3. **Depth sampling** — the median depth in a small window around the bbox center gives a robust Z value.
4. **Projection to 3D** — pixel + depth are converted to camera coordinates via camera intrinsics.
5. **Kalman filtering** — a 6-state (x, y, z, vx, vy, vz) Kalman filter smooths position and predicts during missed detections.
6. **Trajectory** — smoothed points are filtered for outliers (`max_jump`) and median-smoothed over a sliding window.
7. **Visualization** — 3D coordinates, dashboard, and trajectory projections are rendered live.

---

## 📂 Project Structure

```
CVproject/
├── core/
│   ├── camera.py             # Stereo camera (cv.VideoCapture based)
│   ├── camera_realsense.py   # RealSense D435i wrapper (color + depth)
│   ├── calibration.py        # Stereo calibration loader
│   ├── detector.py           # YOLO + CSRT/KCF tracker
│   ├── geometry.py           # 3D math: pixel → 3D, camera → world
│   ├── kalman3d.py           # 3D Kalman filter (cv2.KalmanFilter)
│   ├── trajectory.py         # Trajectory buffer, smoothing, 3D plot export
│   └── visualization.py      # Overlays, dashboard, trajectory panel
├── config.py                 # Paths, camera IDs, model config
├── main.py                   # Entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
# Camera
REALSENSE_WIDTH  = 1280
REALSENSE_HEIGHT = 720
REALSENSE_FPS    = 30

# Model
YOLO_MODEL         = "best.pt"
YOLO_CONF          = 0.25
TARGET_CLASS_NAME  = "drone"

# Output
TRAJECTORY_PLOT_FILE = "trajectory_3d.png"
```

For stereo mode (`core/camera.py` + `core/calibration.py`), also set:

```python
CAM_ID_1   = 4
CAM_ID_2   = 2
CALIB_FILE = "stereo_params.npz"
```

---

## 🤖 Model

The detector uses a fine-tuned YOLO model trained specifically for drone detection.

| Property | Value |
|----------|-------|
| Base | YOLO11x |
| Classes | `drone` |
| Epochs | 50 |
| Image size | 640 |
| Batch | 8 |
| **mAP@50** | **0.964** |
| **mAP@50-95** | **0.592** |
| Precision | 0.934 |
| Recall | 0.929 |

Weights are hosted on Hugging Face:
👉 [`Hlatar/CVproject-drone-weights`](https://huggingface.co/Hlatar/Geoscan_pioneer_base_drone_detection_finetuned)

Downloaded automatically on first run via `huggingface_hub`.

---

## 📊 Output

The saved 3D trajectory plot (`trajectory_3d.png`) includes:

- Full path with start (`o`) and end (`x`) markers
- Fixed or auto-scaled axis limits
- Optional equal aspect ratio for a cubic view

Live dashboard shows:
- FPS, active tracks, stereo pairs
- Current 3D position (X, Y, Z), distance from origin
- Averaged confidence, saved points count
- Recording status (REC ON / OFF)

---

## 🛠️ Tech Stack

- **Python** — core language
- **OpenCV (contrib)** — video capture, tracking, Kalman filter, drawing
- **Ultralytics YOLO** — object detection
- **pyrealsense2** — Intel RealSense SDK
- **NumPy** — numerical operations
- **Matplotlib** — 3D trajectory plots
- **Hugging Face Hub** — model weight distribution

---

## ⚠️ Limitations

- Trained only on the `drone` class.
- Depth accuracy depends on the scene — reflective or transparent surfaces degrade Z.
- Multi-drone scenes may cause ID switching.
- Requires a RealSense depth camera for the main pipeline; stereo mode is available but not fully integrated into the dashboard.

---

## 🗺️ Roadmap

- [ ] Multi-object tracking with IDs
- [ ] Full stereo mode support in the dashboard
- [ ] Real-time 3D viewer (Open3D / pyvista)
- [ ] Export trajectory to CSV / ROS bag
- [ ] Docker image for reproducible setup

---

## 📄 License

This project is distributed under the **MIT License** — see `LICENSE` for details.

> ⚠️ The base YOLO11 model from Ultralytics is licensed under **AGPL-3.0**. If you plan to use this project commercially, review Ultralytics' licensing terms.

---

## 📬 Contact

**Hlatar** — [@Hlatar](https://github.com/Hlatar)

Project link: [https://github.com/Hlatar/CVproject](https://github.com/Hlatar/CVproject)