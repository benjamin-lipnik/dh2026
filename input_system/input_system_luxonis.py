import argparse
import json
import queue
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import depthai as dai
    from depthai_nodes.node import ParsingNeuralNetwork
except Exception as exc:
    raise SystemExit(
        "Missing OAK dependencies. Install with: pip install depthai==3.0.0 depthai-nodes==0.3.4"
    ) from exc


DISPLAY_ENABLED = True


def show_frame(win_name, frame):
    global DISPLAY_ENABLED
    if not DISPLAY_ENABLED:
        return False
    try:
        cv2.imshow(win_name, frame)
        return True
    except cv2.error as exc:
        DISPLAY_ENABLED = False
        print(f"OpenCV display disabled: {exc}")
        print("Continuing without preview window (control/UDP still active).")
        return False


def read_key(default=-1):
    global DISPLAY_ENABLED
    if not DISPLAY_ENABLED:
        return default
    try:
        return cv2.waitKey(1) & 0xFF
    except cv2.error as exc:
        DISPLAY_ENABLED = False
        print(f"OpenCV keyboard input disabled: {exc}")
        return default


def ensure_display_available(require_display=True):
    global DISPLAY_ENABLED
    if not require_display:
        return
    try:
        cv2.namedWindow("Flight Control Demo (Luxonis)", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Flight Control Demo (Luxonis)")
    except cv2.error as exc:
        DISPLAY_ENABLED = False
        raise SystemExit(
            "Camera feed is required by default, but OpenCV GUI is unavailable. "
            "Install/activate GUI OpenCV (opencv-python) or run with --headless.\n"
            f"Original error: {exc}"
        ) from exc


# COCO keypoint indices used by YOLOv8 pose models.
KP_NOSE = 0
KP_LEFT_EYE = 1
KP_RIGHT_EYE = 2
KP_LEFT_SHOULDER = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP = 11
KP_RIGHT_HIP = 12

SKELETON_EDGES = [
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_SHOULDER, KP_LEFT_ELBOW),
    (KP_LEFT_ELBOW, KP_LEFT_WRIST),
    (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW),
    (KP_RIGHT_ELBOW, KP_RIGHT_WRIST),
    (KP_LEFT_SHOULDER, KP_LEFT_HIP),
    (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
]


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0


def v3(p):
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def map_axis(value, neutral, neg_extreme, pos_extreme, dead=0.12):
    if value >= neutral:
        den = max(1e-4, pos_extreme - neutral)
    else:
        den = max(1e-4, neutral - neg_extreme)
    out = (value - neutral) / den
    if abs(out) < dead:
        return 0.0
    return clamp(np.sign(out) * ((abs(out) - dead) / (1.0 - dead)))


def ewma(prev, curr, alpha=0.2):
    return prev * (1.0 - alpha) + curr * alpha


def parse_args():
    parser = argparse.ArgumentParser(
        description="Computer vision control input system using Luxonis OAK + YOLOv8 pose"
    )
    parser.add_argument(
        "--model",
        default="luxonis/yolov8-large-pose-estimation:coco-640x352",
        help="Luxonis model slug from HubAI.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional OAK device name, MXID, or IP.",
    )
    parser.add_argument(
        "--fps-limit",
        type=int,
        default=None,
        help="Optional NN FPS cap.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local .tar.xz NN archive path. Use this when offline.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without camera preview window")
    parser.add_argument("--udp", action="store_true", help="Enable UDP output")
    parser.add_argument("--udp-host", default="192.168.31.217", help="UDP target host")
    parser.add_argument("--udp-port", type=int, default=55555, help="UDP target port")
    parser.add_argument("--load-calibration", default=None, help="Load calibration JSON and skip calibration flow")
    parser.add_argument("--save-calibration", default=None, help="Save calibration JSON after successful calibration")
    parser.add_argument(
        "--run-mode",
        choices=["normal", "udp-debug"],
        default="normal",
        help="Runtime mode. udp-debug prints UDP payload diagnostics after calibration.",
    )
    return parser.parse_args()


def udp_sender(sock_obj, host, port, q, stop_evt, debug=False):
    sent_count = 0
    while not stop_evt.is_set():
        try:
            payload = q.get(timeout=0.1)
        except queue.Empty:
            continue
        try:
            sock_obj.sendto(payload, (host, port))
            sent_count += 1
            if debug:
                ts = time.strftime("%H:%M:%S")
                try:
                    text = payload.decode("utf-8")
                except UnicodeDecodeError:
                    text = str(payload)
                print(f"[{ts}] UDP SENT #{sent_count} -> {host}:{port} {text}")
        except OSError as exc:
            if debug:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] UDP SEND ERROR -> {host}:{port} {exc}")


def load_calibration_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_calibration_file(path, x_to_user, lean_to_user, neutral, extremes, gesture_cal):
    payload = {
        "x_to_user": float(x_to_user),
        "lean_to_user": float(lean_to_user),
        "neutral": {k: float(v) for k, v in neutral.items()},
        "extremes": {k: float(v) for k, v in extremes.items()},
        "gesture_cal": {k: float(v) for k, v in gesture_cal.items()},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_pipeline(args):
    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    platform = device.getPlatformAsString()
    print(f"Platform: {platform}")

    pipeline = dai.Pipeline(device)
    camera_node = pipeline.create(dai.node.Camera).build()

    if args.model_path:
        archive_path = Path(args.model_path)
        if not archive_path.exists():
            raise SystemExit(f"Local model archive not found: {archive_path}")
        nn_archive = dai.NNArchive(str(archive_path))
    else:
        model_description = dai.NNModelDescription(args.model, platform=platform)
        try:
            nn_archive = dai.NNArchive(dai.getModelFromZoo(model_description))
        except RuntimeError as exc:
            raise SystemExit(
                "Could not download model from Luxonis Hub. "
                "You appear to be offline. Provide a local archive with --model-path "
                "or run once with internet access to populate model cache.\n"
                f"Original error: {exc}"
            ) from exc

    nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
        camera_node,
        nn_archive,
        fps=args.fps_limit,
    )

    frame_queue = nn_with_parser.passthrough.createOutputQueue(maxSize=4, blocking=False)
    det_queue = nn_with_parser.out.createOutputQueue(maxSize=4, blocking=False)

    return device, pipeline, frame_queue, det_queue


def draw_pose(frame, keypoints):
    h, w = frame.shape[:2]
    for a, b in SKELETON_EDGES:
        if a < len(keypoints) and b < len(keypoints):
            pa = keypoints[a]
            pb = keypoints[b]
            xa, ya = int(pa.x * w), int(pa.y * h)
            xb, yb = int(pb.x * w), int(pb.y * h)
            cv2.line(frame, (xa, ya), (xb, yb), (90, 255, 90), 2)
    for kp in keypoints:
        x, y = int(kp.x * w), int(kp.y * h)
        cv2.circle(frame, (x, y), 4, (40, 200, 255), -1)


def extract_landmarks(det_msg, mirror=False):
    detections = getattr(det_msg, "detections", [])
    if not detections:
        return None, None

    best = max(detections, key=lambda d: float(getattr(d, "confidence", 0.0)))
    raw_keypoints = list(getattr(best, "keypoints", []))
    if len(raw_keypoints) <= KP_RIGHT_HIP:
        return None, None

    keypoints = []
    for kp in raw_keypoints:
        x = float(getattr(kp, "x", 0.0))
        y = float(getattr(kp, "y", 0.0))
        z = float(getattr(kp, "z", 0.0) or 0.0)
        if mirror:
            x = 1.0 - x
        keypoints.append(Landmark(x=x, y=y, z=z))

    out = {
        "nose": keypoints[KP_NOSE],
        "l_eye": keypoints[KP_LEFT_EYE],
        "r_eye": keypoints[KP_RIGHT_EYE],
        "l_sh": keypoints[KP_LEFT_SHOULDER],
        "r_sh": keypoints[KP_RIGHT_SHOULDER],
        "l_el": keypoints[KP_LEFT_ELBOW],
        "r_el": keypoints[KP_RIGHT_ELBOW],
        "l_wr": keypoints[KP_LEFT_WRIST],
        "r_wr": keypoints[KP_RIGHT_WRIST],
        "l_hip": keypoints[KP_LEFT_HIP],
        "r_hip": keypoints[KP_RIGHT_HIP],
    }
    return out, keypoints


def main():
    args = parse_args()
    ensure_display_available(require_display=not args.headless)

    run_mode = args.run_mode
    debug_udp = run_mode == "udp-debug"
    send_udp = args.udp or debug_udp

    sock = None
    udp_thread = None
    udp_stop = None
    udp_queue = None

    if send_udp:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        udp_stop = threading.Event()
        udp_queue = queue.Queue(maxsize=8)
        udp_thread = threading.Thread(
            target=udp_sender,
            args=(sock, args.udp_host, args.udp_port, udp_queue, udp_stop, debug_udp),
            daemon=True,
        )
        udp_thread.start()

    cal_steps = [
        (4.0, "Get ready: stand neutral, facing camera"),
        (3.0, "Neutral pose"),
        (2.0, "Head UP"),
        (2.0, "Head DOWN"),
        (2.0, "Torso LEAN LEFT"),
        (2.0, "Torso LEAN RIGHT"),
        (2.0, "Torso TURN LEFT"),
        (2.0, "Torso TURN RIGHT"),
        (2.0, "Hands UP (above eyes)"),
        (2.0, "Hands DOWN (below eyes)"),
        (2.0, "Hands CLOSE TOGETHER for forward boost"),
        (2.0, "Hands FAR APART for stop signal"),
    ]
    cal_prep = 2.0

    cal_idx = 0
    step_start = time.time()
    cal_data = {k: [] for k in [
        "orient",
        "head_neutral", "head_up", "head_down",
        "lean_neutral", "lean_left", "lean_right",
        "turn_neutral", "turn_left", "turn_right",
        "hands_neutral", "hands_up", "hands_down",
        "forward_wrist_span",
        "backward_wrist_span",
        "neutral_wrist_span", "neutral_elbow_span",
        "neutral_l_elbow_wrist", "neutral_r_elbow_wrist",
    ]}
    calibrated = False
    cal_msg = ""

    x_to_user = 1.0
    lean_to_user = 1.0

    neutral = {}
    extremes = {}
    gesture_cal = {
        "neutral_wrist_span": 0.0,
        "neutral_elbow_span": 0.0,
        "neutral_l_elbow_wrist": 0.0,
        "neutral_r_elbow_wrist": 0.0,
        "forward_wrist_span": 0.0,
        "backward_wrist_span": 0.0,
    }

    if args.load_calibration:
        try:
            cal = load_calibration_file(args.load_calibration)
            x_to_user = float(cal["x_to_user"])
            lean_to_user = float(cal["lean_to_user"])
            neutral = {k: float(v) for k, v in cal["neutral"].items()}
            extremes = {k: float(v) for k, v in cal["extremes"].items()}
            for k in gesture_cal:
                gesture_cal[k] = float(cal["gesture_cal"][k])
            calibrated = True
            print(f"Loaded calibration from {args.load_calibration}")
        except Exception as exc:
            raise SystemExit(f"Failed to load calibration file {args.load_calibration}: {exc}") from exc

    prev_t = time.time()
    smooth = {"x": 0.0, "y": 0.0, "z": 0.0, "turn": 0.0}

    prev = {
        "lw": None, "rw": None,
        "le": None, "re": None,
        "span_ratio": None,
        "ld": None, "rd": None,
        "lfh": None, "rfh": None,
        "lv": np.zeros(3), "rv": np.zeros(3),
    }
    cooldown = {"L": 0.0, "R": 0.0}
    cooldown_seconds = 0.28

    last_punch_code = {"L": "N", "R": "N"}
    punch_show_until = 0.0

    boost = None  # None | "F"
    boost_armed = True
    boost_needs_guard = False
    status_interval = 0.5
    next_status_at = 0.0
    next_no_pose_log_at = 0.0

    mirror_preview = True

    device = None
    pipeline = None
    try:
        device, pipeline, frame_queue, det_queue = build_pipeline(args)
        pipeline.start()

        while pipeline.isRunning():
            try:
                frame_msg = frame_queue.get()
                det_msg = det_queue.get()
            except Exception:
                continue

            frame = frame_msg.getCvFrame()
            if frame is None:
                continue

            if mirror_preview:
                frame = cv2.flip(frame, 1)

            now = time.time()
            dt = max(1e-3, now - prev_t)
            prev_t = now

            lm, pose_keypoints = extract_landmarks(det_msg, mirror=mirror_preview)
            if lm is None:
                if now >= next_no_pose_log_at:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] Waiting for pose detection (no person/keypoints found).")
                    next_no_pose_log_at = now + 2.0
                show_frame("Flight Control Demo (Luxonis)", frame)
                if read_key() == 27:
                    break
                continue

            nose = lm["nose"]
            l_eye, r_eye = lm["l_eye"], lm["r_eye"]
            l_sh, r_sh = lm["l_sh"], lm["r_sh"]
            l_el, r_el = lm["l_el"], lm["r_el"]
            l_wr, r_wr = lm["l_wr"], lm["r_wr"]
            l_hip, r_hip = lm["l_hip"], lm["r_hip"]

            eye_y = 0.5 * (l_eye.y + r_eye.y)
            wrist_y = 0.5 * (l_wr.y + r_wr.y)
            sh_cx = 0.5 * (l_sh.x + r_sh.x)
            sh_cy = 0.5 * (l_sh.y + r_sh.y)
            hip_cy = 0.5 * (l_hip.y + r_hip.y)
            hip_cx = 0.5 * (l_hip.x + r_hip.x)

            torso_right = np.array([r_sh.x - l_sh.x, r_sh.y - l_sh.y], dtype=np.float32)
            torso_right = torso_right / (np.linalg.norm(torso_right) + 1e-6)
            torso_up = np.array([-torso_right[1], torso_right[0]], dtype=np.float32)
            torso_len = max(1e-4, abs(hip_cy - sh_cy))
            nose_rel = np.array([nose.x - sh_cx, nose.y - sh_cy], dtype=np.float32)
            head_rel = float(np.dot(nose_rel, torso_up) / torso_len)

            orient = l_sh.x - r_sh.x
            lean_raw = sh_cx - hip_cx
            turn_raw = l_sh.z - r_sh.z
            lw, rw = v3(l_wr), v3(r_wr)
            le, re = v3(l_el), v3(r_el)
            hand_eye_raw = wrist_y - eye_y

            if not calibrated:
                duration, text = cal_steps[cal_idx]
                prep = 0.0 if cal_idx == 0 else cal_prep
                elapsed = now - step_start
                collecting = elapsed >= prep
                collect_elapsed = max(0.0, elapsed - prep)
                remain = max(0.0, duration - collect_elapsed)

                if collecting:
                    cal_msg = f"Calibrating [{cal_idx + 1}/{len(cal_steps)}] {text} ({remain:.1f}s)"
                    if cal_idx == 0:
                        cal_data["orient"].append(orient)
                    elif cal_idx == 1:
                        cal_data["head_neutral"].append(head_rel)
                        cal_data["lean_neutral"].append(lean_raw)
                        cal_data["turn_neutral"].append(turn_raw)
                        cal_data["hands_neutral"].append(hand_eye_raw)
                        cal_data["neutral_wrist_span"].append(abs(l_wr.x - r_wr.x))
                        cal_data["neutral_elbow_span"].append(abs(l_el.x - r_el.x))
                        cal_data["neutral_l_elbow_wrist"].append(np.linalg.norm(lw - le))
                        cal_data["neutral_r_elbow_wrist"].append(np.linalg.norm(rw - re))
                    elif cal_idx == 2:
                        cal_data["head_up"].append(head_rel)
                    elif cal_idx == 3:
                        cal_data["head_down"].append(head_rel)
                    elif cal_idx == 4:
                        cal_data["lean_left"].append(lean_raw)
                    elif cal_idx == 5:
                        cal_data["lean_right"].append(lean_raw)
                    elif cal_idx == 6:
                        cal_data["turn_left"].append(turn_raw)
                    elif cal_idx == 7:
                        cal_data["turn_right"].append(turn_raw)
                    elif cal_idx == 8:
                        cal_data["hands_up"].append(hand_eye_raw)
                    elif cal_idx == 9:
                        cal_data["hands_down"].append(hand_eye_raw)
                    elif cal_idx == 10:
                        cal_data["forward_wrist_span"].append(abs(l_wr.x - r_wr.x))
                    elif cal_idx == 11:
                        cal_data["backward_wrist_span"].append(abs(l_wr.x - r_wr.x))
                else:
                    cal_msg = (
                        f"Prepare [{cal_idx + 1}/{len(cal_steps)}] {text} "
                        f"({prep - elapsed:.1f}s)"
                    )

                if collect_elapsed >= duration:
                    cal_idx += 1
                    step_start = now
                    if cal_idx >= len(cal_steps):
                        x_to_user = 1.0 if np.median(cal_data["orient"]) < 0 else -1.0

                        lean_left_raw = float(np.median(cal_data["lean_left"]))
                        lean_right_raw = float(np.median(cal_data["lean_right"]))
                        lean_to_user = 1.0 if lean_right_raw >= lean_left_raw else -1.0

                        head_lo = min(
                            float(np.median(cal_data["head_up"])),
                            float(np.median(cal_data["head_down"])),
                        )
                        head_hi = max(
                            float(np.median(cal_data["head_up"])),
                            float(np.median(cal_data["head_down"])),
                        )
                        hand_lo = min(
                            float(np.median(cal_data["hands_up"])),
                            float(np.median(cal_data["hands_down"])),
                        )
                        hand_hi = max(
                            float(np.median(cal_data["hands_up"])),
                            float(np.median(cal_data["hands_down"])),
                        )

                        turn_left_raw = float(np.median(cal_data["turn_left"]))
                        turn_right_raw = float(np.median(cal_data["turn_right"]))

                        neutral = {
                            "head": float(np.median(cal_data["head_neutral"])),
                            "lean": float(np.median(cal_data["lean_neutral"])) * lean_to_user,
                            "turn": float(np.median(cal_data["turn_neutral"])),
                            "hands": float(np.median(cal_data["hands_neutral"])),
                        }
                        gesture_cal["neutral_wrist_span"] = float(np.median(cal_data["neutral_wrist_span"]))
                        gesture_cal["neutral_elbow_span"] = float(np.median(cal_data["neutral_elbow_span"]))
                        gesture_cal["neutral_l_elbow_wrist"] = float(np.median(cal_data["neutral_l_elbow_wrist"]))
                        gesture_cal["neutral_r_elbow_wrist"] = float(np.median(cal_data["neutral_r_elbow_wrist"]))
                        gesture_cal["forward_wrist_span"] = float(np.median(cal_data["forward_wrist_span"]))
                        gesture_cal["backward_wrist_span"] = float(np.median(cal_data["backward_wrist_span"]))

                        if head_hi - head_lo < 0.04:
                            head_lo = neutral["head"] - 0.05
                            head_hi = neutral["head"] + 0.05
                        if hand_hi - hand_lo < 0.06:
                            hand_lo = neutral["hands"] - 0.08
                            hand_hi = neutral["hands"] + 0.08

                        extremes = {
                            "head_up": head_lo,
                            "head_down": head_hi,
                            "lean_left": lean_left_raw * lean_to_user,
                            "lean_right": lean_right_raw * lean_to_user,
                            "turn_left": turn_left_raw,
                            "turn_right": turn_right_raw,
                            "hands_up": hand_lo,
                            "hands_down": hand_hi,
                        }
                        calibrated = True
                        side_text = "mirrored frame" if x_to_user > 0 else "non-mirrored frame"
                        print(f"\033[2J\033[HCalibration complete ({side_text}).")
                        if args.save_calibration:
                            try:
                                save_calibration_file(
                                    args.save_calibration,
                                    x_to_user,
                                    lean_to_user,
                                    neutral,
                                    extremes,
                                    gesture_cal,
                                )
                                print(f"Saved calibration to {args.save_calibration}")
                            except Exception as exc:
                                print(f"Failed to save calibration to {args.save_calibration}: {exc}")

                if now >= next_status_at:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] {cal_msg}")
                    next_status_at = now + status_interval

                draw_pose(frame, pose_keypoints)
                show_frame("Flight Control Demo (Luxonis)", frame)
                if read_key() == 27:
                    break
                continue

            lean_user = lean_raw * lean_to_user

            move_x = map_axis(lean_user, neutral["lean"], extremes["lean_left"], extremes["lean_right"], dead=0.18)
            move_y = map_axis(head_rel, neutral["head"], extremes["head_up"], extremes["head_down"], dead=0.16)
            l_hand_eye = l_wr.y - eye_y
            r_hand_eye = r_wr.y - eye_y
            hand_eye_avg = 0.5 * (l_hand_eye + r_hand_eye)
            hand_symmetry = clamp(1.0 - abs(l_hand_eye - r_hand_eye) / 0.18, 0.0, 1.0)
            hand_eye_stable = hand_symmetry * hand_eye_avg + (1.0 - hand_symmetry) * neutral["hands"]
            move_z = map_axis(hand_eye_stable, neutral["hands"], extremes["hands_up"], extremes["hands_down"], dead=0.20)
            turn = map_axis(turn_raw, neutral["turn"], extremes["turn_left"], extremes["turn_right"], dead=0.14)

            smooth["x"] = ewma(smooth["x"], move_x)
            smooth["y"] = ewma(smooth["y"], move_y)
            smooth["z"] = ewma(smooth["z"], move_z)
            smooth["turn"] = ewma(smooth["turn"], turn)

            guard = (l_el.y > l_wr.y + 0.035) and (r_el.y > r_wr.y + 0.035)
            l_ew = np.linalg.norm(lw - le)
            r_ew = np.linalg.norm(rw - re)
            l_compact = l_ew < max(0.10, gesture_cal["neutral_l_elbow_wrist"] * 0.72)
            r_compact = r_ew < max(0.10, gesture_cal["neutral_r_elbow_wrist"] * 0.72)
            side_span = abs(l_wr.x - r_wr.x)
            elbow_span = abs(l_el.x - r_el.x)
            shoulder_span = max(1e-4, abs(l_sh.x - r_sh.x))
            span_ratio = side_span / shoulder_span
            elbow_ratio = elbow_span / shoulder_span
            forward_span_ratio = (
                gesture_cal["forward_wrist_span"] / shoulder_span
                if gesture_cal["forward_wrist_span"] > 0 else 0.45
            )
            backward_span_ratio = (
                gesture_cal["backward_wrist_span"] / shoulder_span
                if gesture_cal["backward_wrist_span"] > 0 else 1.9
            )
            neutral_wrist_ratio = (
                gesture_cal["neutral_wrist_span"] / shoulder_span
                if gesture_cal["neutral_wrist_span"] > 0 else 1.0
            )
            neutral_elbow_ratio = (
                gesture_cal["neutral_elbow_span"] / shoulder_span
                if gesture_cal["neutral_elbow_span"] > 0 else 1.0
            )

            if prev["lw"] is None:
                span_jump = 0.0
            else:
                prev_span = span_ratio if prev["span_ratio"] is None else prev["span_ratio"]
                span_jump = span_ratio - prev_span

            forward_trigger = (
                not guard
                and span_ratio < max(0.28, forward_span_ratio * 1.25)
                and (l_wr.z < l_sh.z - 0.05)
                and (r_wr.z < r_sh.z - 0.05)
                and (l_el.z < l_sh.z - 0.02)
                and (r_el.z < r_sh.z - 0.02)
                and abs(l_wr.y - r_wr.y) < 0.16
                and abs(l_el.y - r_el.y) < 0.16
                and abs(l_sh.y - r_sh.y) < 0.16
                and abs(l_wr.y - l_el.y) < 0.09
                and abs(r_wr.y - r_el.y) < 0.09
                and abs(l_wr.y - l_sh.y) < 0.12
                and abs(r_wr.y - r_sh.y) < 0.12
                and abs(l_el.y - l_sh.y) < 0.12
                and abs(r_el.y - r_sh.y) < 0.12
            )

            stop_trigger = (
                span_ratio > max(neutral_wrist_ratio + 0.45, backward_span_ratio * 0.82)
                and span_jump > 0.08
                and elbow_ratio > max(1.10, neutral_elbow_ratio + 0.18)
                and abs(l_wr.y - r_wr.y) < 0.14
                and abs(l_wr.z - r_wr.z) < 0.25
                and not (l_compact and r_compact)
            )

            if boost_needs_guard and guard:
                boost_armed = True
                boost_needs_guard = False

            if boost == "F":
                if stop_trigger:
                    boost = None
                    boost_armed = False
                    boost_needs_guard = True
            else:
                if boost_armed and forward_trigger:
                    boost = "F"
                    boost_armed = False

            boosted_z = smooth["z"]
            if boost == "F":
                boosted_z = clamp(max(boosted_z, 0.95))

            cooldown["L"] = max(0.0, cooldown["L"] - dt)
            cooldown["R"] = max(0.0, cooldown["R"] - dt)

            punch = {"L": None, "R": None}
            punch_score = {"L": -1.0, "R": -1.0}

            if prev["lw"] is not None:
                lv = (lw - prev["lw"]) / dt
                rv = (rw - prev["rw"]) / dt
                la = (lv - prev["lv"]) / dt
                ra = (rv - prev["rv"]) / dt
                l_speed, r_speed = np.linalg.norm(lv), np.linalg.norm(rv)
                l_imp, r_imp = np.linalg.norm(la), np.linalg.norm(ra)

                speed_gate = 0.52
                accel_gate = 6.0

                if cooldown["L"] <= 0 and l_speed > speed_gate and l_imp > accel_gate:
                    fw = lw - le
                    p_fw = prev["lw"] - prev["le"]
                    fw_xy = fw[:2]
                    p_fw_xy = p_fw[:2]
                    h_ratio = abs(fw_xy[0]) / (abs(fw_xy[0]) + abs(fw_xy[1]) + 1e-6)
                    p_h_ratio = abs(p_fw_xy[0]) / (abs(p_fw_xy[0]) + abs(p_fw_xy[1]) + 1e-6)
                    elbow_v = (le - prev["le"]) / dt
                    vx, vy = lv[0], lv[1]

                    direct = (
                        h_ratio > 0.30 and h_ratio < 0.76
                        and abs(vx) > 0.26 and abs(vy) > 0.22
                        and abs(vx) / (abs(vy) + 1e-6) > 0.45
                        and abs(vx) / (abs(vy) + 1e-6) < 2.2
                        and abs(abs(elbow_v[0]) - abs(elbow_v[1])) < max(abs(elbow_v[0]), abs(elbow_v[1]), 1e-6) * 0.75
                        and np.sign(vx) == np.sign(elbow_v[0])
                        and np.sign(vy) == np.sign(elbow_v[1])
                        and vy < -0.18
                    )
                    upper = (
                        h_ratio < 0.40
                        and vy < -0.60 and elbow_v[1] < -0.34
                        and abs(vy) > abs(vx) * 1.25
                    )
                    hook = (
                        abs(vx) > 0.50
                        and abs(vx) > abs(vy) * 1.45
                        and abs(elbow_v[0]) > 0.20
                        and np.sign(vx) == np.sign(elbow_v[0])
                        and h_ratio > 0.58
                        and h_ratio > p_h_ratio + 0.04
                    )

                    if direct:
                        punch["L"] = "direct"
                    elif hook:
                        punch["L"] = "hook"
                    elif upper:
                        punch["L"] = "uppercut"
                    if punch["L"]:
                        punch_score["L"] = l_speed + 0.15 * l_imp

                if cooldown["R"] <= 0 and r_speed > speed_gate and r_imp > accel_gate:
                    fw = rw - re
                    p_fw = prev["rw"] - prev["re"]
                    fw_xy = fw[:2]
                    p_fw_xy = p_fw[:2]
                    h_ratio = abs(fw_xy[0]) / (abs(fw_xy[0]) + abs(fw_xy[1]) + 1e-6)
                    p_h_ratio = abs(p_fw_xy[0]) / (abs(p_fw_xy[0]) + abs(p_fw_xy[1]) + 1e-6)
                    elbow_v = (re - prev["re"]) / dt
                    vx, vy = rv[0], rv[1]

                    direct = (
                        h_ratio > 0.30 and h_ratio < 0.76
                        and abs(vx) > 0.26 and abs(vy) > 0.22
                        and abs(vx) / (abs(vy) + 1e-6) > 0.45
                        and abs(vx) / (abs(vy) + 1e-6) < 2.2
                        and abs(abs(elbow_v[0]) - abs(elbow_v[1])) < max(abs(elbow_v[0]), abs(elbow_v[1]), 1e-6) * 0.75
                        and np.sign(vx) == np.sign(elbow_v[0])
                        and np.sign(vy) == np.sign(elbow_v[1])
                        and vy < -0.18
                    )
                    upper = (
                        h_ratio < 0.40
                        and vy < -0.60 and elbow_v[1] < -0.34
                        and abs(vy) > abs(vx) * 1.25
                    )
                    hook = (
                        abs(vx) > 0.50
                        and abs(vx) > abs(vy) * 1.45
                        and abs(elbow_v[0]) > 0.20
                        and np.sign(vx) == np.sign(elbow_v[0])
                        and h_ratio > 0.58
                        and h_ratio > p_h_ratio + 0.04
                    )

                    if direct:
                        punch["R"] = "direct"
                    elif hook:
                        punch["R"] = "hook"
                    elif upper:
                        punch["R"] = "uppercut"
                    if punch["R"]:
                        punch_score["R"] = r_speed + 0.15 * r_imp

                if punch["L"] and punch["R"]:
                    if punch_score["L"] >= punch_score["R"]:
                        punch["R"] = None
                    else:
                        punch["L"] = None

                if punch["L"]:
                    cooldown["L"] = cooldown_seconds
                if punch["R"]:
                    cooldown["R"] = cooldown_seconds

                prev["lv"], prev["rv"] = lv, rv

            left_key = "L" if l_wr.x <= r_wr.x else "R"
            right_key = "R" if left_key == "L" else "L"
            punch_user_left = punch[left_key]
            punch_user_right = punch[right_key]

            if punch_user_left:
                last_punch_code["L"] = {"direct": "D", "uppercut": "U", "hook": "H"}[punch_user_left]
                last_punch_code["R"] = "N"
                punch_show_until = now + 0.9
            elif punch_user_right:
                last_punch_code["R"] = {"direct": "D", "uppercut": "U", "hook": "H"}[punch_user_right]
                last_punch_code["L"] = "N"
                punch_show_until = now + 0.9
            elif now > punch_show_until:
                last_punch_code["L"] = "N"
                last_punch_code["R"] = "N"

            prev["lw"], prev["rw"] = lw, rw
            prev["le"], prev["re"] = le, re
            prev["span_ratio"] = span_ratio
            prev["ld"], prev["rd"] = np.linalg.norm(lw - le), np.linalg.norm(rw - re)
            prev["lfh"] = abs((lw - le)[0]) / (abs((lw - le)[0]) + abs((lw - le)[1]) + 1e-6)
            prev["rfh"] = abs((rw - re)[0]) / (abs((rw - re)[0]) + abs((rw - re)[1]) + 1e-6)

            # Explicit punch channels for downstream controller mapping.
            punch_flags = {
                "punch_left_direct": punch_user_left == "direct",
                "punch_left_hook": punch_user_left == "hook",
                "punch_left_uppercut": punch_user_left == "uppercut",
                "punch_right_direct": punch_user_right == "direct",
                "punch_right_hook": punch_user_right == "hook",
                "punch_right_uppercut": punch_user_right == "uppercut",
            }
            punch_latched_flags = {
                "punch_left_direct_latched": last_punch_code["L"] == "D",
                "punch_left_hook_latched": last_punch_code["L"] == "H",
                "punch_left_uppercut_latched": last_punch_code["L"] == "U",
                "punch_right_direct_latched": last_punch_code["R"] == "D",
                "punch_right_hook_latched": last_punch_code["R"] == "H",
                "punch_right_uppercut_latched": last_punch_code["R"] == "U",
            }

            out = {
                "t": round(now, 3),
                "move_x": round(smooth["x"], 2),
                "move_y": round(smooth["y"], 2),
                "move_z": round(boosted_z, 2),
                "turn": round(smooth["turn"], 2),
                "boost_forward": boost == "F",
                "boost_backward": False,
                "boost_armed": boost_armed,
                "boost_needs_guard": boost_needs_guard,
                "punch_left": punch_user_left,
                "punch_right": punch_user_right,
                "punch_left_code": last_punch_code["L"],
                "punch_right_code": last_punch_code["R"],
                **punch_flags,
                **punch_latched_flags,
            }

            if send_udp and sock is not None:
                try:
                    payload = json.dumps(out).encode("utf-8")
                    try:
                        udp_queue.put_nowait(payload)
                        if debug_udp:
                            ts = time.strftime("%H:%M:%S")
                            print(f"[{ts}] UDP ENQUEUE q={udp_queue.qsize()}/8 {out}")
                    except queue.Full:
                        _ = udp_queue.get_nowait()
                        udp_queue.put_nowait(payload)
                        if debug_udp:
                            ts = time.strftime("%H:%M:%S")
                            print(f"[{ts}] UDP ENQUEUE DROPPED_OLDEST q={udp_queue.qsize()}/8 {out}")
                except (OSError, queue.Empty):
                    pass

            moves = []
            if abs(out["move_z"]) > 0.1:
                moves.append(f"{'F' if out['move_z'] > 0 else 'B'}:{abs(out['move_z']):.1f}")
            if abs(out["move_y"]) > 0.1:
                moves.append(f"{'U' if out['move_y'] > 0 else 'D'}:{abs(out['move_y']):.1f}")
            if abs(out["move_x"]) > 0.1:
                moves.append(f"{'R' if out['move_x'] > 0 else 'L'}:{abs(out['move_x']):.1f}")
            rot = "None" if abs(out["turn"]) <= 0.1 else f"{'R' if out['turn'] > 0 else 'L'}:{abs(out['turn']):.1f}"
            if now >= next_status_at:
                ts = time.strftime("%H:%M:%S")
                print(f"\n[{ts}] Control Snapshot")
                print(f"  Moves : {' '.join(moves) if moves else 'None'}")
                print(f"  Rot   : {rot}")
                print(f"  Boost : {boost or 'N'}")
                print(f"  Punch : L:{last_punch_code['L']} R:{last_punch_code['R']}")
                print(f"  Raw   : {out}")
                next_status_at = now + status_interval

            draw_pose(frame, pose_keypoints)
            show_frame("Flight Control Demo (Luxonis)", frame)

            if read_key() == 27:
                break

    finally:
        if DISPLAY_ENABLED:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if udp_stop is not None:
            udp_stop.set()
        if udp_thread is not None:
            udp_thread.join(timeout=0.5)
        if sock is not None:
            sock.close()
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception:
                pass
        if device is not None:
            try:
                device.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
