import argparse
import json
import queue
import socket
import threading
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    import depthai as dai
except Exception as exc:
    raise SystemExit("Missing depthai. Install with: pip install depthai==3.0.0") from exc


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
DISPLAY_ENABLED = True
STATUS_INTERVAL = 0.5
COOLDOWN = 0.28
CAL_PREP = 2.0
DEFAULT_CAL_PATH = str(Path(__file__).with_name("calibration_baseline.json"))
CAL_STEPS = [
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


def v3(p):
    return np.array([p.x, p.y, p.z], dtype=np.float32)


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
        cv2.namedWindow("Flight Control Demo Cam1", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Flight Control Demo Cam2", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Flight Control Demo Cam1")
        cv2.destroyWindow("Flight Control Demo Cam2")
    except cv2.error as exc:
        DISPLAY_ENABLED = False
        raise SystemExit(
            "Camera feed is required by default, but OpenCV GUI is unavailable. "
            "Install/activate GUI OpenCV (opencv-python) or run with --headless.\n"
            f"Original error: {exc}"
        ) from exc


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


class UdpChannel:
    def __init__(self, host, port, debug=False):
        self.host = host
        self.port = port
        self.debug = debug
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.stop_evt = threading.Event()
        self.q = queue.Queue(maxsize=8)
        self.thread = threading.Thread(target=self._sender, daemon=True)
        self.thread.start()

    def _sender(self):
        sent_count = 0
        while not self.stop_evt.is_set():
            try:
                payload = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.sock.sendto(payload, (self.host, self.port))
                sent_count += 1
                if self.debug:
                    ts = time.strftime("%H:%M:%S")
                    try:
                        text = payload.decode("utf-8")
                    except UnicodeDecodeError:
                        text = str(payload)
                    print(f"[{ts}] UDP SENT #{sent_count} -> {self.host}:{self.port} {text}")
            except OSError as exc:
                if self.debug:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] UDP SEND ERROR -> {self.host}:{self.port} {exc}")

    def send(self, out):
        payload = json.dumps(out).encode("utf-8")
        try:
            self.q.put_nowait(payload)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
            except queue.Empty:
                pass
            self.q.put_nowait(payload)

    def close(self):
        self.stop_evt.set()
        self.thread.join(timeout=0.5)
        self.sock.close()


class ControlState:
    def __init__(self, name, save_calibration_path=None):
        self.name = name
        self.save_calibration_path = save_calibration_path

        self.cal_idx = 0
        self.step_start = time.time()
        self.cal_data = {k: [] for k in [
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

        self.calibrated = False
        self.cal_msg = ""

        self.x_to_user = 1.0
        self.lean_to_user = 1.0

        self.neutral = {}
        self.extremes = {}

        self.prev_t = time.time()
        self.smooth = {"x": 0.0, "y": 0.0, "z": 0.0, "turn": 0.0}

        self.prev = {
            "lw": None, "rw": None,
            "le": None, "re": None,
            "span_ratio": None,
            "ld": None, "rd": None,
            "lfh": None, "rfh": None,
            "lv": np.zeros(3), "rv": np.zeros(3),
        }
        self.cooldown = {"L": 0.0, "R": 0.0}

        self.last_punch_code = {"L": "N", "R": "N"}
        self.punch_show_until = 0.0

        self.boost = None
        self.boost_armed = True
        self.boost_needs_guard = False

        self.next_status_at = 0.0

        self.gesture_cal = {
            "neutral_wrist_span": 0.0,
            "neutral_elbow_span": 0.0,
            "neutral_l_elbow_wrist": 0.0,
            "neutral_r_elbow_wrist": 0.0,
            "forward_wrist_span": 0.0,
            "backward_wrist_span": 0.0,
        }

    def load_calibration(self, path):
        cal = load_calibration_file(path)
        self.x_to_user = float(cal["x_to_user"])
        self.lean_to_user = float(cal["lean_to_user"])
        self.neutral = {k: float(v) for k, v in cal["neutral"].items()}
        self.extremes = {k: float(v) for k, v in cal["extremes"].items()}
        for k in self.gesture_cal:
            self.gesture_cal[k] = float(cal["gesture_cal"][k])
        self.calibrated = True
        print(f"[{self.name}] Loaded calibration from {path}")

    def _build_payload(self, now, boosted_z, turn, punch_user_left, punch_user_right):
        punch_flags = {
            "punch_left_direct": punch_user_left == "direct",
            "punch_left_hook": punch_user_left == "hook",
            "punch_left_uppercut": punch_user_left == "uppercut",
            "punch_right_direct": punch_user_right == "direct",
            "punch_right_hook": punch_user_right == "hook",
            "punch_right_uppercut": punch_user_right == "uppercut",
        }
        punch_latched_flags = {
            "punch_left_direct_latched": self.last_punch_code["L"] == "D",
            "punch_left_hook_latched": self.last_punch_code["L"] == "H",
            "punch_left_uppercut_latched": self.last_punch_code["L"] == "U",
            "punch_right_direct_latched": self.last_punch_code["R"] == "D",
            "punch_right_hook_latched": self.last_punch_code["R"] == "H",
            "punch_right_uppercut_latched": self.last_punch_code["R"] == "U",
        }
        return {
            "t": round(now, 3),
            "move_x": round(self.smooth["x"], 2),
            "move_y": round(self.smooth["y"], 2),
            "move_z": round(boosted_z, 2),
            "turn": round(turn, 2),
            "boost_forward": self.boost == "F",
            "boost_backward": False,
            "boost_armed": self.boost_armed,
            "boost_needs_guard": self.boost_needs_guard,
            "punch_left": punch_user_left,
            "punch_right": punch_user_right,
            "punch_left_code": self.last_punch_code["L"],
            "punch_right_code": self.last_punch_code["R"],
            **punch_flags,
            **punch_latched_flags,
        }

    def process_landmarks(self, lm):
        now = time.time()
        dt = max(1e-3, now - self.prev_t)
        self.prev_t = now

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

        if not self.calibrated:
            duration, text = CAL_STEPS[self.cal_idx]
            prep = 0.0 if self.cal_idx == 0 else CAL_PREP
            elapsed = now - self.step_start
            collecting = elapsed >= prep
            collect_elapsed = max(0.0, elapsed - prep)
            remain = max(0.0, duration - collect_elapsed)

            if collecting:
                self.cal_msg = f"Calibrating [{self.cal_idx + 1}/{len(CAL_STEPS)}] {text} ({remain:.1f}s)"
                if self.cal_idx == 0:
                    self.cal_data["orient"].append(orient)
                elif self.cal_idx == 1:
                    self.cal_data["head_neutral"].append(head_rel)
                    self.cal_data["lean_neutral"].append(lean_raw)
                    self.cal_data["turn_neutral"].append(turn_raw)
                    self.cal_data["hands_neutral"].append(hand_eye_raw)
                    self.cal_data["neutral_wrist_span"].append(abs(l_wr.x - r_wr.x))
                    self.cal_data["neutral_elbow_span"].append(abs(l_el.x - r_el.x))
                    self.cal_data["neutral_l_elbow_wrist"].append(np.linalg.norm(lw - le))
                    self.cal_data["neutral_r_elbow_wrist"].append(np.linalg.norm(rw - re))
                elif self.cal_idx == 2:
                    self.cal_data["head_up"].append(head_rel)
                elif self.cal_idx == 3:
                    self.cal_data["head_down"].append(head_rel)
                elif self.cal_idx == 4:
                    self.cal_data["lean_left"].append(lean_raw)
                elif self.cal_idx == 5:
                    self.cal_data["lean_right"].append(lean_raw)
                elif self.cal_idx == 6:
                    self.cal_data["turn_left"].append(turn_raw)
                elif self.cal_idx == 7:
                    self.cal_data["turn_right"].append(turn_raw)
                elif self.cal_idx == 8:
                    self.cal_data["hands_up"].append(hand_eye_raw)
                elif self.cal_idx == 9:
                    self.cal_data["hands_down"].append(hand_eye_raw)
                elif self.cal_idx == 10:
                    self.cal_data["forward_wrist_span"].append(abs(l_wr.x - r_wr.x))
                elif self.cal_idx == 11:
                    self.cal_data["backward_wrist_span"].append(abs(l_wr.x - r_wr.x))
            else:
                self.cal_msg = f"Prepare [{self.cal_idx + 1}/{len(CAL_STEPS)}] {text} ({prep - elapsed:.1f}s)"

            if collect_elapsed >= duration:
                self.cal_idx += 1
                self.step_start = now
                if self.cal_idx >= len(CAL_STEPS):
                    self.x_to_user = 1.0 if np.median(self.cal_data["orient"]) < 0 else -1.0

                    lean_left_raw = float(np.median(self.cal_data["lean_left"]))
                    lean_right_raw = float(np.median(self.cal_data["lean_right"]))
                    self.lean_to_user = 1.0 if lean_right_raw >= lean_left_raw else -1.0

                    head_lo = min(float(np.median(self.cal_data["head_up"])), float(np.median(self.cal_data["head_down"])))
                    head_hi = max(float(np.median(self.cal_data["head_up"])), float(np.median(self.cal_data["head_down"])))
                    hand_lo = min(float(np.median(self.cal_data["hands_up"])), float(np.median(self.cal_data["hands_down"])))
                    hand_hi = max(float(np.median(self.cal_data["hands_up"])), float(np.median(self.cal_data["hands_down"])))

                    turn_left_raw = float(np.median(self.cal_data["turn_left"]))
                    turn_right_raw = float(np.median(self.cal_data["turn_right"]))

                    self.neutral = {
                        "head": float(np.median(self.cal_data["head_neutral"])),
                        "lean": float(np.median(self.cal_data["lean_neutral"])) * self.lean_to_user,
                        "turn": float(np.median(self.cal_data["turn_neutral"])),
                        "hands": float(np.median(self.cal_data["hands_neutral"])),
                    }
                    self.gesture_cal["neutral_wrist_span"] = float(np.median(self.cal_data["neutral_wrist_span"]))
                    self.gesture_cal["neutral_elbow_span"] = float(np.median(self.cal_data["neutral_elbow_span"]))
                    self.gesture_cal["neutral_l_elbow_wrist"] = float(np.median(self.cal_data["neutral_l_elbow_wrist"]))
                    self.gesture_cal["neutral_r_elbow_wrist"] = float(np.median(self.cal_data["neutral_r_elbow_wrist"]))
                    self.gesture_cal["forward_wrist_span"] = float(np.median(self.cal_data["forward_wrist_span"]))
                    self.gesture_cal["backward_wrist_span"] = float(np.median(self.cal_data["backward_wrist_span"]))

                    if head_hi - head_lo < 0.04:
                        head_lo = self.neutral["head"] - 0.05
                        head_hi = self.neutral["head"] + 0.05
                    if hand_hi - hand_lo < 0.06:
                        hand_lo = self.neutral["hands"] - 0.08
                        hand_hi = self.neutral["hands"] + 0.08

                    self.extremes = {
                        "head_up": head_lo,
                        "head_down": head_hi,
                        "lean_left": lean_left_raw * self.lean_to_user,
                        "lean_right": lean_right_raw * self.lean_to_user,
                        "turn_left": turn_left_raw,
                        "turn_right": turn_right_raw,
                        "hands_up": hand_lo,
                        "hands_down": hand_hi,
                    }
                    self.calibrated = True
                    side_text = "mirrored frame" if self.x_to_user > 0 else "non-mirrored frame"
                    print(f"[{self.name}] Calibration complete ({side_text}).")
                    if self.save_calibration_path:
                        try:
                            save_calibration_file(
                                self.save_calibration_path,
                                self.x_to_user,
                                self.lean_to_user,
                                self.neutral,
                                self.extremes,
                                self.gesture_cal,
                            )
                            print(f"[{self.name}] Saved calibration to {self.save_calibration_path}")
                        except Exception as exc:
                            print(f"[{self.name}] Failed to save calibration to {self.save_calibration_path}: {exc}")

            if now >= self.next_status_at:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] [{self.name}] {self.cal_msg}")
                self.next_status_at = now + STATUS_INTERVAL
            return None

        lean_user = lean_raw * self.lean_to_user

        move_x = map_axis(lean_user, self.neutral["lean"], self.extremes["lean_left"], self.extremes["lean_right"], dead=0.18)
        move_y = map_axis(head_rel, self.neutral["head"], self.extremes["head_up"], self.extremes["head_down"], dead=0.16)
        l_hand_eye = l_wr.y - eye_y
        r_hand_eye = r_wr.y - eye_y
        hand_eye_avg = 0.5 * (l_hand_eye + r_hand_eye)
        hand_symmetry = clamp(1.0 - abs(l_hand_eye - r_hand_eye) / 0.18, 0.0, 1.0)
        hand_eye_stable = hand_symmetry * hand_eye_avg + (1.0 - hand_symmetry) * self.neutral["hands"]
        move_z = map_axis(hand_eye_stable, self.neutral["hands"], self.extremes["hands_up"], self.extremes["hands_down"], dead=0.20)
        turn = map_axis(turn_raw, self.neutral["turn"], self.extremes["turn_left"], self.extremes["turn_right"], dead=0.14)

        self.smooth["x"] = ewma(self.smooth["x"], move_x)
        self.smooth["y"] = ewma(self.smooth["y"], move_y)
        self.smooth["z"] = ewma(self.smooth["z"], move_z)
        self.smooth["turn"] = ewma(self.smooth["turn"], turn)

        guard = (l_el.y > l_wr.y + 0.035) and (r_el.y > r_wr.y + 0.035)
        l_ew = np.linalg.norm(lw - le)
        r_ew = np.linalg.norm(rw - re)
        l_compact = l_ew < max(0.10, self.gesture_cal["neutral_l_elbow_wrist"] * 0.72)
        r_compact = r_ew < max(0.10, self.gesture_cal["neutral_r_elbow_wrist"] * 0.72)
        side_span = abs(l_wr.x - r_wr.x)
        elbow_span = abs(l_el.x - r_el.x)
        shoulder_span = max(1e-4, abs(l_sh.x - r_sh.x))
        span_ratio = side_span / shoulder_span
        elbow_ratio = elbow_span / shoulder_span
        forward_span_ratio = self.gesture_cal["forward_wrist_span"] / shoulder_span if self.gesture_cal["forward_wrist_span"] > 0 else 0.45
        backward_span_ratio = self.gesture_cal["backward_wrist_span"] / shoulder_span if self.gesture_cal["backward_wrist_span"] > 0 else 1.9
        neutral_wrist_ratio = self.gesture_cal["neutral_wrist_span"] / shoulder_span if self.gesture_cal["neutral_wrist_span"] > 0 else 1.0
        neutral_elbow_ratio = self.gesture_cal["neutral_elbow_span"] / shoulder_span if self.gesture_cal["neutral_elbow_span"] > 0 else 1.0

        if self.prev["lw"] is None:
            span_jump = 0.0
        else:
            prev_span = span_ratio if self.prev["span_ratio"] is None else self.prev["span_ratio"]
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

        if self.boost_needs_guard and guard:
            self.boost_armed = True
            self.boost_needs_guard = False

        if self.boost == "F":
            if stop_trigger:
                self.boost = None
                self.boost_armed = False
                self.boost_needs_guard = True
        else:
            if self.boost_armed and forward_trigger:
                self.boost = "F"
                self.boost_armed = False

        boosted_z = self.smooth["z"]
        if self.boost == "F":
            boosted_z = clamp(max(boosted_z, 0.95))

        self.cooldown["L"] = max(0.0, self.cooldown["L"] - dt)
        self.cooldown["R"] = max(0.0, self.cooldown["R"] - dt)

        punch = {"L": None, "R": None}
        punch_score = {"L": -1.0, "R": -1.0}

        if self.prev["lw"] is not None:
            lv = (lw - self.prev["lw"]) / dt
            rv = (rw - self.prev["rw"]) / dt
            la = (lv - self.prev["lv"]) / dt
            ra = (rv - self.prev["rv"]) / dt
            l_speed, r_speed = np.linalg.norm(lv), np.linalg.norm(rv)
            l_imp, r_imp = np.linalg.norm(la), np.linalg.norm(ra)

            speed_gate = 0.52
            accel_gate = 6.0

            if self.cooldown["L"] <= 0 and l_speed > speed_gate and l_imp > accel_gate:
                fw = lw - le
                p_fw = self.prev["lw"] - self.prev["le"]
                fw_xy = fw[:2]
                p_fw_xy = p_fw[:2]
                h_ratio = abs(fw_xy[0]) / (abs(fw_xy[0]) + abs(fw_xy[1]) + 1e-6)
                p_h_ratio = abs(p_fw_xy[0]) / (abs(p_fw_xy[0]) + abs(p_fw_xy[1]) + 1e-6)
                elbow_v = (le - self.prev["le"]) / dt
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

            if self.cooldown["R"] <= 0 and r_speed > speed_gate and r_imp > accel_gate:
                fw = rw - re
                p_fw = self.prev["rw"] - self.prev["re"]
                fw_xy = fw[:2]
                p_fw_xy = p_fw[:2]
                h_ratio = abs(fw_xy[0]) / (abs(fw_xy[0]) + abs(fw_xy[1]) + 1e-6)
                p_h_ratio = abs(p_fw_xy[0]) / (abs(p_fw_xy[0]) + abs(p_fw_xy[1]) + 1e-6)
                elbow_v = (re - self.prev["re"]) / dt
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
                self.cooldown["L"] = COOLDOWN
            if punch["R"]:
                self.cooldown["R"] = COOLDOWN

            self.prev["lv"], self.prev["rv"] = lv, rv

        left_key = "L" if l_wr.x <= r_wr.x else "R"
        right_key = "R" if left_key == "L" else "L"
        punch_user_left = punch[left_key]
        punch_user_right = punch[right_key]

        if punch_user_left:
            self.last_punch_code["L"] = {"direct": "D", "uppercut": "U", "hook": "H"}[punch_user_left]
            self.last_punch_code["R"] = "N"
            self.punch_show_until = now + 0.9
        elif punch_user_right:
            self.last_punch_code["R"] = {"direct": "D", "uppercut": "U", "hook": "H"}[punch_user_right]
            self.last_punch_code["L"] = "N"
            self.punch_show_until = now + 0.9
        elif now > self.punch_show_until:
            self.last_punch_code["L"] = "N"
            self.last_punch_code["R"] = "N"

        self.prev["lw"], self.prev["rw"] = lw, rw
        self.prev["le"], self.prev["re"] = le, re
        self.prev["span_ratio"] = span_ratio
        self.prev["ld"], self.prev["rd"] = np.linalg.norm(lw - le), np.linalg.norm(rw - re)
        self.prev["lfh"] = abs((lw - le)[0]) / (abs((lw - le)[0]) + abs((lw - le)[1]) + 1e-6)
        self.prev["rfh"] = abs((rw - re)[0]) / (abs((rw - re)[0]) + abs((rw - re)[1]) + 1e-6)

        out = self._build_payload(now, boosted_z, self.smooth["turn"], punch_user_left, punch_user_right)
        out["camera"] = self.name

        if now >= self.next_status_at:
            ts = time.strftime("%H:%M:%S")
            moves = []
            if abs(out["move_z"]) > 0.1:
                moves.append(f"{'F' if out['move_z'] > 0 else 'B'}:{abs(out['move_z']):.1f}")
            if abs(out["move_y"]) > 0.1:
                moves.append(f"{'U' if out['move_y'] > 0 else 'D'}:{abs(out['move_y']):.1f}")
            if abs(out["move_x"]) > 0.1:
                moves.append(f"{'R' if out['move_x'] > 0 else 'L'}:{abs(out['move_x']):.1f}")
            rot = "None" if abs(out["turn"]) <= 0.1 else f"{'R' if out['turn'] > 0 else 'L'}:{abs(out['turn']):.1f}"
            print(f"\n[{ts}] [{self.name}] Control Snapshot")
            print(f"  Moves : {' '.join(moves) if moves else 'None'}")
            print(f"  Rot   : {rot}")
            print(f"  Boost : {self.boost or 'N'}")
            print(f"  Punch : L:{self.last_punch_code['L']} R:{self.last_punch_code['R']}")
            print(f"  Raw   : {out}")
            self.next_status_at = now + STATUS_INTERVAL

        return out


def parse_socket(name):
    key = name.strip().upper()
    if not hasattr(dai.CameraBoardSocket, key):
        raise SystemExit(f"Unknown socket '{name}'. Example values: CAM_A, CAM_B, CAM_C, RGB")
    return getattr(dai.CameraBoardSocket, key)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dual-camera Luxonis input system using MediaPipe on host"
    )
    parser.add_argument("--headless", action="store_true", help="Run without camera preview windows")
    parser.add_argument("--device-cam1", default=None, help="Optional device/IP for camera 1 stream")
    parser.add_argument("--device-cam2", default=None, help="Optional device/IP for camera 2 stream")
    parser.add_argument("--cam1-socket", default="CAM_A", help="Camera socket for channel 1")
    parser.add_argument("--cam2-socket", default="CAM_A", help="Camera socket for channel 2")
    parser.add_argument("--frame-width", type=int, default=640, help="Requested frame width")
    parser.add_argument("--frame-height", type=int, default=360, help="Requested frame height")
    parser.add_argument("--fps", type=float, default=30.0, help="Requested camera FPS")

    parser.add_argument("--udp", action="store_true", help="Enable UDP output")
    parser.add_argument("--udp-host", default="192.168.31.217", help="UDP recipient host")
    parser.add_argument("--udp-port-cam1", type=int, default=55555, help="UDP port for camera 1 controls")
    parser.add_argument("--udp-port-cam2", type=int, default=55556, help="UDP port for camera 2 controls")

    parser.add_argument("--load-calibration-cam1", default=DEFAULT_CAL_PATH, help="Load calibration JSON for camera 1")
    parser.add_argument("--load-calibration-cam2", default=DEFAULT_CAL_PATH, help="Load calibration JSON for camera 2")
    parser.add_argument("--save-calibration-cam1", default=None, help="Save calibration JSON for camera 1")
    parser.add_argument("--save-calibration-cam2", default=None, help="Save calibration JSON for camera 2")

    parser.add_argument(
        "--run-mode",
        choices=["normal", "udp-debug"],
        default="normal",
        help="Runtime mode. udp-debug prints UDP payload diagnostics.",
    )
    args, _ = parser.parse_known_args()
    return args


def choose_devices(args):
    infos = dai.Device.getAllAvailableDevices()
    if args.device_cam1 and args.device_cam2:
        return args.device_cam1, args.device_cam2
    if args.device_cam1 and not args.device_cam2:
        if len(infos) < 2:
            raise SystemExit("Only one OAK device detected; provide --device-cam2 explicitly.")
        alt = [str(i.name) for i in infos if str(i.name) != str(args.device_cam1)]
        if not alt:
            raise SystemExit("Could not auto-select a second device. Use --device-cam2.")
        return args.device_cam1, alt[0]
    if args.device_cam2 and not args.device_cam1:
        if len(infos) < 2:
            raise SystemExit("Only one OAK device detected; provide --device-cam1 explicitly.")
        alt = [str(i.name) for i in infos if str(i.name) != str(args.device_cam2)]
        if not alt:
            raise SystemExit("Could not auto-select a first device. Use --device-cam1.")
        return alt[0], args.device_cam2
    if len(infos) < 2:
        raise SystemExit(
            "Dual mode requires two detected OAK devices. "
            "Currently detected: {}".format(len(infos))
        )
    return str(infos[0].name), str(infos[1].name)


def build_single_stream(device_selector, socket_name, frame_width, frame_height, fps, label):
    info = dai.DeviceInfo(device_selector)
    device = dai.Device(info)
    print(f"[{label}] device={device_selector} platform={device.getPlatformAsString()}")

    pipeline = dai.Pipeline(device)
    socket_sel = parse_socket(socket_name)
    available = [f.socket for f in device.getConnectedCameraFeatures()]
    if socket_sel not in available:
        if not available:
            raise SystemExit(f"[{label}] No camera sensors detected on device {device_selector}.")
        fallback = available[0]
        print(f"[{label}] Requested socket {socket_sel} missing; using {fallback}.")
        socket_sel = fallback
    cam = pipeline.create(dai.node.Camera).build(socket_sel)
    out = cam.requestOutput(
        (frame_width, frame_height),
        dai.ImgFrame.Type.BGR888i,
        fps=fps,
    )
    q = out.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()
    return {"device": device, "pipeline": pipeline, "queue": q, "label": label}


def landmarks_from_mediapipe(result):
    if not result.pose_landmarks:
        return None
    lm = result.pose_landmarks.landmark
    return {
        "nose": lm[0],
        "l_eye": lm[2],
        "r_eye": lm[5],
        "l_sh": lm[11],
        "r_sh": lm[12],
        "l_el": lm[13],
        "r_el": lm[14],
        "l_wr": lm[15],
        "r_wr": lm[16],
        "l_hip": lm[23],
        "r_hip": lm[24],
    }


def process_camera_frame(name, frame, pose, state, mirror=True):
    if mirror:
        frame = cv2.flip(frame, 1)
    result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.pose_landmarks:
        mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    lm = landmarks_from_mediapipe(result)
    out = state.process_landmarks(lm) if lm else None
    show_frame(f"Flight Control Demo {name}", frame)
    return out


def main():
    args = parse_args()
    ensure_display_available(require_display=not args.headless)

    debug_udp = args.run_mode == "udp-debug"

    pose1 = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    pose2 = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    state1 = ControlState("cam1", save_calibration_path=args.save_calibration_cam1)
    state2 = ControlState("cam2", save_calibration_path=args.save_calibration_cam2)

    if args.load_calibration_cam1:
        state1.load_calibration(args.load_calibration_cam1)
    if args.load_calibration_cam2:
        state2.load_calibration(args.load_calibration_cam2)

    udp1 = None
    udp2 = None
    if args.udp or debug_udp:
        udp1 = UdpChannel(args.udp_host, args.udp_port_cam1, debug=debug_udp)
        udp2 = UdpChannel(args.udp_host, args.udp_port_cam2, debug=debug_udp)

    stream1 = None
    stream2 = None
    try:
        dev1, dev2 = choose_devices(args)
        stream1 = build_single_stream(dev1, args.cam1_socket, args.frame_width, args.frame_height, args.fps, "cam1")
        stream2 = build_single_stream(dev2, args.cam2_socket, args.frame_width, args.frame_height, args.fps, "cam2")

        while stream1["pipeline"].isRunning() or stream2["pipeline"].isRunning():
            got_any = False

            m1 = stream1["queue"].tryGet() if stream1 is not None else None
            if m1 is not None:
                got_any = True
                f1 = m1.getCvFrame()
                if f1 is not None:
                    out1 = process_camera_frame("Cam1", f1, pose1, state1)
                    if out1 is not None and udp1 is not None:
                        udp1.send(out1)

            m2 = stream2["queue"].tryGet() if stream2 is not None else None
            if m2 is not None:
                got_any = True
                f2 = m2.getCvFrame()
                if f2 is not None:
                    out2 = process_camera_frame("Cam2", f2, pose2, state2)
                    if out2 is not None and udp2 is not None:
                        udp2.send(out2)

            if read_key() == 27:
                break

            if not got_any:
                time.sleep(0.002)

    finally:
        pose1.close()
        pose2.close()

        if DISPLAY_ENABLED:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        if udp1 is not None:
            udp1.close()
        if udp2 is not None:
            udp2.close()

        if stream1 is not None and stream1["pipeline"] is not None:
            try:
                stream1["pipeline"].stop()
            except Exception:
                pass
        if stream1 is not None and stream1["device"] is not None:
            try:
                stream1["device"].close()
            except Exception:
                pass

        if stream2 is not None and stream2["pipeline"] is not None:
            try:
                stream2["pipeline"].stop()
            except Exception:
                pass
        if stream2 is not None and stream2["device"] is not None:
            try:
                stream2["device"].close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
