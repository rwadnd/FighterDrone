# Task2: YOLOv11 Multiprocess Balloon Tracker (ByteTrack)

Real-time webcam detection + tracking of a target class (default: **BALLOON_ID=1**) using Ultralytics YOLO and ByteTrack.  
Two processes keep latency low: **capture** → **inference** (YOLO+tracker) → **UI** with a compact HUD and per-target threat score.

---

## Features
- Multiprocess pipeline (separate capture + inference)
- ByteTrack ID persistence and per-ID state (centroids, areas, velocity)
- Threat scoring (size, center proximity, approach, velocity)
- Low-latency IPC via JPEG buffers
- Compact HUD (FPS, timings, ranked targets)

---

## Quick Start

### 1) Environment
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install ultralytics opencv-python numpy
# Torch: choose build for your system
pip install torch --index-url https://download.pytorch.org/whl/cpu    # CPU only
pip install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.x
```
---

## Performance
- Multiprocess pipeline (separate capture + inference)
## Tested Devices & Performance

| Device              | File         | CPU              | GPU             | Resolution | IMGSZ | Avg FPS   |
|---------------------|--------------|------------------|-----------------|------------|-------|-----------|
| Windows 11 Desktop  | task.py      | Ryzen 5 5600X    | RTX 4060TI 8GB  | 1280×720   | 640   | ~28–35 FPS |
| Windows 11 Desktop  | task_mp.py   | Ryzen 5 5600X    | RTX 4060TI 8GB  | 1280×720   | 640   | ~34–42 FPS |

---
