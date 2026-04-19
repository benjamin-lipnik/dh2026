# Input System

This folder contains two gesture-control input pipelines that output control data over UDP:

- `input_system.py`: webcam + MediaPipe pose
- `input_system_luxonis.py`: OAK camera + Luxonis YOLOv8 pose
- `input_system_dual_luxonis.py`: two OAK camera streams + MediaPipe pose on host

Both scripts provide:

- Calibration flow
- Optional calibration save/load
- Forward boost latch and stop gesture
- Punch detection (left/right direct, hook, uppercut)
- UDP output (including debug mode)

## 1. Environment Setup

From repository root (`C:\Users\blin\Documents\dh2026`):

```powershell
python -m venv .venv
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

If camera preview windows fail with OpenCV HighGUI errors, force GUI OpenCV in the active environment:

```powershell
pip uninstall -y opencv-python-headless
pip install --force-reinstall opencv-python==4.10.0.84
```

## 2. Baseline Script (Webcam + MediaPipe)

Run from `input_system` folder:

```powershell
python input_system.py --udp
```

Optional arguments:

- `--udp-host 192.168.31.217`
- `--udp-port 55555`
- `--run-mode udp-debug` (verbose UDP enqueue/send logging)
- `--headless` (no preview window)

### Save/Load Calibration

First run and save:

```powershell
python input_system.py --udp --save-calibration calibration_baseline.json
```

Next runs load and skip calibration:

```powershell
python input_system.py --udp --load-calibration calibration_baseline.json
```

## 3. Luxonis Script (OAK + YOLOv8 Pose)

Run from `input_system` folder:

```powershell
python input_system_luxonis.py --device 169.254.77.49 --udp --udp-host 192.168.31.217 --udp-port 55555
```

Optional arguments:

- `--model luxonis/yolov8-large-pose-estimation:coco-640x352`
- `--model-path <local_model_archive.tar.xz>` (offline model usage)
- `--run-mode udp-debug`
- `--headless`

### Save/Load Calibration

First run and save:

```powershell
python input_system_luxonis.py --device 169.254.77.49 --udp --save-calibration calibration_luxonis.json
```

Next runs load and skip calibration:

```powershell
python input_system_luxonis.py --device 169.254.77.49 --udp --load-calibration calibration_luxonis.json
```

## 4. UDP Output Payload

Key fields include:

- Motion: `move_x`, `move_y`, `move_z`, `turn`
- Boost: `boost_forward`, `boost_backward`, `boost_armed`, `boost_needs_guard`
- Punch (legacy): `punch_left`, `punch_right`
- Punch code: `punch_left_code`, `punch_right_code` (`N`, `D`, `H`, `U`)
- Punch channels (event-level booleans):
  - `punch_left_direct`, `punch_left_hook`, `punch_left_uppercut`
  - `punch_right_direct`, `punch_right_hook`, `punch_right_uppercut`
- Punch channels (latched booleans):
  - `punch_left_direct_latched`, `punch_left_hook_latched`, `punch_left_uppercut_latched`
  - `punch_right_direct_latched`, `punch_right_hook_latched`, `punch_right_uppercut_latched`

## 5. Dual Luxonis Input (Two Cameras, Host Processing)

`input_system_dual_luxonis.py` keeps the same control logic as baseline but uses two
Luxonis camera sockets as frame inputs and runs MediaPipe pose estimation on the laptop.

Typical topology:

- Controller camera on port 1 -> `CAM_A` (example)
- Controller camera on port 2 -> `CAM_B` (example)
- Laptop on controller port 3 (uplink)

Run with per-camera UDP ports:

```powershell
python input_system_dual_luxonis.py \
  --device 169.254.77.49 \
  --udp --udp-host 192.168.31.217 \
  --udp-port-cam1 55555 \
  --udp-port-cam2 55556 \
  --cam1-socket CAM_A \
  --cam2-socket CAM_B
```

Calibration save/load per camera:

```powershell
python input_system_dual_luxonis.py --device 169.254.77.49 --udp \
  --save-calibration-cam1 calibration_cam1.json \
  --save-calibration-cam2 calibration_cam2.json

python input_system_dual_luxonis.py --device 169.254.77.49 --udp \
  --load-calibration-cam1 calibration_cam1.json \
  --load-calibration-cam2 calibration_cam2.json
```

Each UDP payload includes `camera` with value `cam1` or `cam2` for downstream routing.

## 6. UDP Communication Test Utility

Standalone send/receive testing:

```powershell
python udp_comm_test.py --mode recv --bind-host 0.0.0.0 --port 55555
python udp_comm_test.py --mode send --host 192.168.31.217 --port 55555 --count 20 --interval 0.1
```

## 7. Troubleshooting

### No camera preview window

- Confirm you are in the intended virtual environment.
- Check OpenCV GUI support:

```powershell
python -c "import cv2,sys; print(sys.executable); print(cv2.__version__); print('GUI NONE' in cv2.getBuildInformation())"
```

Expected last value: `False`.

### Luxonis script starts but no detections

- Ensure OAK device is reachable and selected correctly with `--device`.
- Stand in view of the camera during initialization/calibration.
- Use `--run-mode udp-debug` for extra visibility.

### Offline model download failure

Use local archive:

```powershell
python input_system_luxonis.py --device 169.254.77.49 --model-path C:\path\to\model.tar.xz --udp
```
