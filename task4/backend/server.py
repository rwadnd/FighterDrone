# server.py
import asyncio
import time
import struct
import cv2
import numpy as np
import websockets
import json
import torch
from ultralytics import YOLO
import math
from collections import deque
from datetime import datetime

# --- YOLO / app config (partly from task_mp) ---
HOST = "0.0.0.0"
PORT = 8765

TARGET_FPS = 100
JPEG_QUALITY = 70
TARGET_WIDTH = 1200
TARGET_HEIGHT = 720

MODELPATH = "task3/lastv6.pt"
IMGSZ = 640
CONF_T = 0.5
IOU_T = 0.45
BALLOON_ID = 1
TRACKER_CFG = "bytetrack.yaml"
MAX_DET = 3

# Sizes (server uses TARGET_WIDTH/HEIGHT; keep them in sync)
WIDTH = TARGET_WIDTH
HEIGHT = TARGET_HEIGHT
CENTER = (WIDTH // 2, HEIGHT // 2)

# Lock-on (two-phase)
LOCK_SHAPE = "circle"   # "circle" or "square"
LOCK_RADIUS = 120
LOCK_SIDE = 220
LOCK_ARM_SEC = 4.0
GAP_TOLERANCE_SEC = 0.5

# Threat / display
K_HISTORY = 5
V_MIN = 1.5
AREA_T_POS = 400.0
AREA_T_NEG = 400.0
CENTER_MARGIN_X = 0.10
CENTER_MARGIN_Y = 0.10
W_SIZE, W_CENTER, W_APPROACH, W_VEL = 0.45, 0.30, 0.20, 0.05
EXPECTED_MAX_AHAT = 0.08
W_LOCK_SIZE = 0.6
W_LOCK_CENTER = 0.4

# Logging paths (kept from task_mp but optional in server)
LOG_TXT_PATH = "task3/lock_log.txt"
LOG_JSON_PATH = "task3/lock_events.json"

# HUD
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICK = 1
LINE_SPACING = 20
PANEL_PAD_X = 10
PANEL_PAD_Y = 10

# Queues (not used for multiprocessing in server, but keep constants)
FRAME_QUEUE_MAX = 2
RESULT_QUEUE_MAX = 2
LOG_QUEUE_MAX = 100

# device selection
USE_GPU = torch.cuda.is_available()
DEVICE = 'cuda' if USE_GPU else 'cpu'
model = None


def pack_timestamp_ms():
    return struct.pack("<d", time.time() * 1000.0)


# ----------------- Utilities (from task_mp) -----------------
def clip01(x): return max(0.0, min(1.0, x))
def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

def size_index_from_Ahat(Ahat: float) -> str:
    if Ahat < 0.01: return "Small"
    if Ahat < 0.04: return "Medium"
    return "Large"

def sector_label(cx, cy, W, H, mx=CENTER_MARGIN_X, my=CENTER_MARGIN_Y):
    cx0, cy0 = W // 2, H // 2
    if abs(cx - cx0) <= mx * W and abs(cy - cy0) <= my * H:
        return "Center"
    top = cy < cy0
    left = cx < cy0
    if top and left: return "Top-Left"
    if top and not left: return "Top-Right"
    if not top and left: return "Bottom-Left"
    return "Bottom-Right"

def avg_velocity(centroids):
    if len(centroids) < 2: return 0.0, 0.0
    dx = sum(centroids[i+1][0] - centroids[i][0] for i in range(len(centroids)-1)) / (len(centroids)-1)
    dy = sum(centroids[i+1][1] - centroids[i][1] for i in range(len(centroids)-1)) / (len(centroids)-1)
    return dx, dy

def dir_label_from_v(vx, vy):
    vy = -vy
    speed = math.hypot(vx, vy)
    if speed < V_MIN:
        return "Stationary"
    ang = math.degrees(math.atan2(vy, vx))
    if -22.5 <= ang < 22.5: return "Moving Right"
    if 22.5 <= ang < 67.5: return "Moving Up-Right"
    if 67.5 <= ang < 112.5: return "Moving Up"
    if 112.5 <= ang < 157.5: return "Moving Up-Left"
    if ang >= 157.5 or ang < -157.5: return "Moving Left"
    if -157.5 <= ang < -112.5: return "Moving Down-Left"
    if -112.5 <= ang < -67.5: return "Moving Down"
    return "Moving Down-Right"

def median_delta(seq):
    if len(seq) < 2: return 0.0
    deltas = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    deltas.sort()
    m = len(deltas) // 2
    return deltas[m] if len(deltas) % 2 else 0.5 * (deltas[m-1] + deltas[m])

def center_score(cx, cy, W, H):
    dx = (cx - W/2) / (W/2)
    dy = (cy - H/2) / (H/2)
    d = min(1.0, math.hypot(dx, dy))
    return 1.0 - d

def point_in_lock_zone(pt):
    cx, cy = pt
    if LOCK_SHAPE == "circle":
        dx = cx - CENTER[0]
        dy = cy - CENTER[1]
        return (dx*dx + dy*dy) <= (LOCK_RADIUS * LOCK_RADIUS)
    else:
        half = LOCK_SIDE // 2
        return (CENTER[0]-half) <= cx <= (CENTER[0]+half) and (CENTER[1]-half) <= cy <= (CENTER[1]+half)

def draw_lock_zone(img):
    if LOCK_SHAPE == "circle":
        cv2.circle(img, CENTER, LOCK_RADIUS, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        half = LOCK_SIDE // 2
        cv2.rectangle(img, (CENTER[0]-half, CENTER[1]-half), (CENTER[0]+half, CENTER[1]+half), (255,255,255), 2, cv2.LINE_AA)

def draw_translucent_panel(img, x, y, w, h, color=(0,0,0), alpha=0.4):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_hud_bottom_left(img, lines, colors=None):
    max_text_w, text_h = 0, 0
    for ln in lines:
        (tw, th), _ = cv2.getTextSize(ln, FONT, FONT_SCALE, FONT_THICK)
        max_text_w = max(max_text_w, tw)
        text_h = max(text_h, th)
    panel_w = max_text_w + 2 * PANEL_PAD_X
    panel_h = PANEL_PAD_Y * 2 + len(lines) * LINE_SPACING
    x = 10
    y = img.shape[0] - 10 - panel_h
    draw_translucent_panel(img, x, y, panel_w, panel_h, color=(0,0,0), alpha=0.45)
    baseline_y = y + PANEL_PAD_Y + text_h
    for i, ln in enumerate(lines):
        col = (255,255,255)
        if colors is not None and i < len(colors) and colors[i] is not None:
            col = colors[i]
        cv2.putText(img, ln, (x + PANEL_PAD_X, baseline_y), FONT, FONT_SCALE, col, FONT_THICK, cv2.LINE_AA)
        baseline_y += LINE_SPACING


# Simple tracking state
class TrackState:
    def __init__(self):
        self.centroids = deque(maxlen=K_HISTORY)
        self.areas = deque(maxlen=K_HISTORY)


class LockFSM:
    def __init__(self):
        self.mode = "idle"
        self.id = None
        self.arming_start_perf = None
        self.arming_start_wall = None
        self.last_seen_perf = None
        self.lock_acquire_perf = None
        self.lock_start_wall = None

    def start_arming(self, tid):
        self.mode = "arming"
        self.id = tid
        self.arming_start_perf = time.perf_counter()
        self.arming_start_wall = time.time()
        self.last_seen_perf = self.arming_start_perf
        self.lock_acquire_perf = None
        self.lock_start_wall = None
        print(f"Balloon ID: {tid} entered lock zone (arming started)")

    def cancel_arming_as_failed(self, now_perf):
        if self.mode != "arming": return
        tid = self.id
        arm_dur = (now_perf - self.arming_start_perf) if self.arming_start_perf else 0.0
        print(f"Balloon ID: {tid} arming duration: {arm_dur:.1f}s -> LOCK FAILED")
        self.__init__()

    def acquire_lock(self, now_perf):
        if self.mode != "arming": return
        tid = self.id
        self.mode = "locked"
        self.lock_acquire_perf = now_perf
        self.lock_start_wall = time.time()
        print(f"Balloon ID: {tid} -> LOCK ACQUIRED")

    def end_locked_session(self, now_perf):
        if self.mode != "locked": return
        tid = self.id
        lock_dur = (now_perf - self.lock_acquire_perf) if self.lock_acquire_perf else 0.0
        print(f"Balloon ID: {tid} lock duration: {lock_dur:.1f}s -> LOCK ENDED")
        self.__init__()


def open_capture(source):
    """source = 0 for webcam, or a filename for video"""
    cap = cv2.VideoCapture(source)

    if isinstance(source, int):
        # Webcam settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 60)

    return cap


async def producer_handler(websocket):
    print("Client connected:", websocket.remote_address)

    # Default: webcam
    cap = open_capture(0)

    frame_interval = 1.0 / TARGET_FPS
    frame_times = []

    try:
        while True:
            loop_start = time.monotonic()

            # ---- LISTEN FOR COMMANDS FROM CLIENT (async) ----
            # Non-blocking check for messages
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=0.0001)
                data = json.loads(msg)

                # frontend sends { "cmd": "use_file", "path": "video.mp4" }
                if data.get("cmd") == "use_file":
                    new_path = data.get("path")
                    print("Switching to video:", new_path)
                    cap.release()
                    cap = open_capture(new_path)

                # frontend sends { "cmd": "use_camera" }
                elif data.get("cmd") == "use_camera":
                    print("Switching to webcam")
                    cap.release()
                    cap = open_capture(0)

            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print("Command error:", e)

            # ---- READ FRAME ----
            ret, frame = cap.read()
            if not ret:
                # If video file ended → loop it
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

            # ---------------- YOLO detection + tracking (migrated from task_mp) ----------------
            # per-client track DB and FSM (created on first iteration)
            if 'track_db' not in locals():
                track_db = {}
            if 'fsm' not in locals():
                fsm = LockFSM()

            annotated = frame.copy()
            draw_lock_zone(annotated)

            t0 = time.perf_counter()
            det_ms = 0.0
            outputs = []
            top_threat = None
            try:
                results = model.track(
                    frame,
                    persist=True,
                    tracker=TRACKER_CFG,
                    conf=CONF_T,
                    iou=IOU_T,
                    imgsz=IMGSZ,
                    verbose=False,
                    device=DEVICE,
                    classes=[BALLOON_ID],
                    max_det=MAX_DET,
                    half=USE_GPU,
                )
                r0 = results[0]
                boxes = getattr(r0, 'boxes', None)
                if boxes is not None and len(boxes) > 0:
                    xywh = boxes.xywh.detach().cpu().numpy()
                    confs = boxes.conf.detach().cpu().numpy()
                    ids = boxes.id.detach().cpu().numpy() if boxes.id is not None else np.array([None]*len(xywh))

                    for (x, y, w, h), s, tid in zip(xywh, confs, ids):
                        if tid is None:
                            continue
                        tid = int(tid)
                        cx, cy = int(x), int(y)
                        A = float(w) * float(h)
                        Ahat = A / float(WIDTH * HEIGHT)

                        st = track_db.get(tid) or TrackState()
                        track_db[tid] = st
                        st.centroids.append((cx, cy))
                        st.areas.append(A)

                        pos_label = sector_label(cx, cy, WIDTH, HEIGHT)
                        vx, vy = avg_velocity(st.centroids)
                        move_label = dir_label_from_v(vx, vy)
                        dA = median_delta(st.areas)
                        if dA > AREA_T_POS:
                            status = "Approaching"
                        elif dA < -AREA_T_NEG:
                            status = "Receding"
                        else:
                            status = "Stable Distance"

                        size_idx = size_index_from_Ahat(Ahat)
                        S_size = clip01(Ahat / EXPECTED_MAX_AHAT)
                        S_center = center_score(cx, cy, WIDTH, HEIGHT)
                        S_approach = 1.0 if status == "Approaching" else (0.5 if status == "Stable Distance" else 0.0)
                        speed = math.hypot(vx, vy)
                        S_vel = sigmoid(speed / 5.0)
                        threat = (W_SIZE * S_size + W_CENTER * S_center + W_APPROACH * S_approach + W_VEL * S_vel)

                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)

                        outputs.append({
                            "id": tid,
                            "conf": float(s),
                            "area": float(A),
                            "Ahat": float(Ahat),
                            "bbox": (x1, y1, x2, y2),
                            "centroid": (cx, cy),
                            "position_label": pos_label,
                            "status": status,
                            "size_index": size_idx,
                            "movement": move_label,
                            "speed": float(speed),
                            "S_size": float(S_size),
                            "S_center": float(S_center),
                            "S_approach": float(S_approach),
                            "S_vel": float(S_vel),
                            "threat_score": float(round(threat, 3)),
                            "lock_priority": float(W_LOCK_SIZE * S_size + W_LOCK_CENTER * S_center),
                            "in_lock_zone": bool(point_in_lock_zone((cx, cy)))
                        })

                    outputs.sort(key=lambda o: o["threat_score"], reverse=True)
                    top_threat = outputs[0] if outputs else None
            except Exception as e:
                print("YOLO/track error:", e)
            finally:
                t1 = time.perf_counter()
                det_ms = (t1 - t0) * 1000.0

            # ---------------- FSM update (same logic as task_mp) ----------------
            now_perf = time.perf_counter()
            if fsm.mode == "idle":
                in_zone = [o for o in outputs if o.get("in_lock_zone")]
                if in_zone:
                    in_zone.sort(key=lambda o: o["lock_priority"], reverse=True)
                    fsm.start_arming(in_zone[0]["id"])

            elif fsm.mode == "arming":
                aid = fsm.id
                cur = next((o for o in outputs if o["id"] == aid), None)
                if cur is not None and cur.get("in_lock_zone"):
                    fsm.last_seen_perf = now_perf
                    arm_elapsed = now_perf - fsm.arming_start_perf
                    if arm_elapsed >= LOCK_ARM_SEC:
                        fsm.acquire_lock(now_perf)
                else:
                    last_seen = fsm.last_seen_perf or fsm.arming_start_perf
                    if (now_perf - last_seen) > GAP_TOLERANCE_SEC:
                        fsm.cancel_arming_as_failed(now_perf)

            elif fsm.mode == "locked":
                aid = fsm.id
                cur = next((o for o in outputs if o["id"] == aid), None)
                if cur is not None and cur.get("in_lock_zone"):
                    fsm.last_seen_perf = now_perf
                else:
                    last_seen = fsm.last_seen_perf or fsm.lock_acquire_perf
                    if (now_perf - last_seen) > GAP_TOLERANCE_SEC:
                        fsm.end_locked_session(now_perf)

            # ---------------- Draw boxes & labels ----------------
            for o in outputs:
                x1, y1, x2, y2 = o["bbox"]
                color = (0, 255, 255)  # yellow default
                label = f"ID {o['id']} {o['size_index']}"

                if fsm.mode == "locked" and o["id"] == fsm.id:
                    color = (0, 200, 0)
                    label = f"LOCKED ID {o['id']}"
                elif fsm.mode == "arming" and o["id"] == fsm.id:
                    color = (0, 165, 255)
                    arm_elapsed = time.perf_counter() - (fsm.arming_start_perf or time.perf_counter())
                    remaining = max(0.0, LOCK_ARM_SEC - arm_elapsed)
                    label = f"ARMING ID {o['id']} ({remaining:.1f}s)"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
                cv2.putText(annotated, label, (x1, max(20, y1-8)), FONT, 0.65, color, 2, cv2.LINE_AA)

                if fsm.mode == "locked" and o["id"] == fsm.id:
                    dur = time.perf_counter() - (fsm.lock_acquire_perf or time.perf_counter())
                    cv2.putText(annotated, f"lock {dur:.1f}s", (x1, y2 + 20), FONT, 0.6, (0, 200, 0), 2, cv2.LINE_AA)

            # HUD & timing
            hud_lines = [
                f"FPS: {len([t for t in frame_times if time.monotonic() - t <= 1.0]):5d}",
                f"Times ms  det+track={det_ms:5.1f}",
                f"State: {fsm.mode}  ActiveID: {fsm.id}"
            ]
            if outputs:
                hud_lines.append(f"Detections: {len(outputs)}  TopThreat: {top_threat['threat_score']:.3f}" if top_threat else f"Detections: {len(outputs)}")
                hud_colors = [(255,255,255)] * len(hud_lines)
            else:
                hud_lines.append("No balloons detected")
                hud_colors = [(255,255,255)] * len(hud_lines)

            draw_hud_bottom_left(annotated, hud_lines, hud_colors)

            # Encode JPEG
            ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                continue

            jpg_bytes = jpg.tobytes()

            # ---- Compute SERVER FPS ----
            now = time.monotonic()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t <= 1.0]
            server_fps = len(frame_times)

            # ---- HEADER: timestamp + uint16 FPS ----
            header = pack_timestamp_ms() + struct.pack("<H", server_fps)
            payload = header + jpg_bytes

            await websocket.send(payload)

            # ---- FPS LIMIT ----
            elapsed = time.monotonic() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            else:
                await asyncio.sleep(0)

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        cap.release()


async def main():
    global model
    print(f"Starting server at ws://{HOST}:{PORT}")
    # Load YOLO model once
    try:
        print(f"Loading YOLO model from {MODELPATH} on device={DEVICE} ...")
        model = YOLO(MODELPATH)
        if DEVICE == 'cuda':
            model.to('cuda')
        print("YOLO model loaded")
    except Exception as e:
        print("Failed to load YOLO model:", e)

    async with websockets.serve(producer_handler, HOST, PORT, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
