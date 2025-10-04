import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow duplicate OpenMP libs (fix crash)

import cv2 as cv
from ultralytics import YOLO
import torch, math, time, json
from collections import deque
import numpy as np
from datetime import datetime

# ==== Video / Model Settings ====
WIDTH, HEIGHT = 1280, 720
IMGSZ = 640
BALLOON_ID = 1
TRACKER_CFG = "bytetrack.yaml"
CONF_T, IOU_T = 0.5, 0.45
MAX_DET = 3
CENTER = (WIDTH // 2, HEIGHT // 2)
MODELPATH = "task3/lastv6.pt" 

# ==== Lock-On Settings (Two-Phase) ====
LOCK_SHAPE = "circle"            # "circle" or "square"
LOCK_RADIUS = 120                # px (circle radius)
LOCK_SIDE = 220                  # px (square side)
LOCK_ARM_SEC = 4.0               # time to acquire the lock
GAP_TOLERANCE_SEC = 0.5          # tolerate short dropouts both in arming and locked
LOG_TXT_PATH = "task3/lock_log.txt"
LOG_JSON_PATH = "task3/lock_events.json"  # JSON Lines (one object per line)

# ==== Threat / Display ====
K_HISTORY = 5
V_MIN = 1.5
AREA_T_POS = 400.0
AREA_T_NEG = 400.0
CENTER_MARGIN_X = 0.10
CENTER_MARGIN_Y = 0.10
W_SIZE, W_CENTER, W_APPROACH, W_VEL = 0.45, 0.30, 0.20, 0.05
EXPECTED_MAX_AHAT = 0.08

# selection priority used ONLY for picking active target (size + center)
W_LOCK_SIZE = 0.6
W_LOCK_CENTER = 0.4

# ==== HUD ====
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICK = 1
LINE_SPACING = 20
PANEL_PAD_X = 10
PANEL_PAD_Y = 10

class TrackState:
    def __init__(self):
        self.centroids = deque(maxlen=K_HISTORY)
        self.areas = deque(maxlen=K_HISTORY)

track_db = {}

# ========= Utility Functions =========
def size_index_from_Ahat(Ahat: float) -> str:
    if Ahat < 0.01: return "Small"
    if Ahat < 0.04: return "Medium"
    return "Large"

def sector_label(cx, cy, W, H, mx=CENTER_MARGIN_X, my=CENTER_MARGIN_Y):
    cx0, cy0 = W // 2, H // 2
    if abs(cx - cx0) <= mx * W and abs(cy - cy0) <= my * H:
        return "Center"
    top = cy < cy0
    left = cx < cx0
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

def clip01(x): return max(0.0, min(1.0, x))
def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

def draw_translucent_panel(img, x, y, w, h, color=(0,0,0), alpha=0.4):
    overlay = img.copy()
    cv.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_hud_bottom_left(img, lines, colors=None):
    max_text_w, text_h = 0, 0
    for ln in lines:
        (tw, th), _ = cv.getTextSize(ln, FONT, FONT_SCALE, FONT_THICK)
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
        cv.putText(img, ln, (x + PANEL_PAD_X, baseline_y), FONT, FONT_SCALE, col, FONT_THICK, cv.LINE_AA)
        baseline_y += LINE_SPACING

# ---- Lock zone geometry
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
        cv.circle(img, CENTER, LOCK_RADIUS, (255, 255, 255), 2, cv.LINE_AA)
    else:
        half = LOCK_SIDE // 2
        cv.rectangle(img, (CENTER[0]-half, CENTER[1]-half), (CENTER[0]+half, CENTER[1]+half), (255,255,255), 2, cv.LINE_AA)

# ---- Logging / JSON
def _log_text(line: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] {line}"
    print(msg)
    try:
        with open(LOG_TXT_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def _write_json_event(obj: dict):
    try:
        with open(LOG_JSON_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ========= Two-Phase Lock State =========
# mode: "idle" -> "arming" -> "locked" -> back to "idle"
lock_state = {
    "mode": "idle",
    "id": None,
    "arming_start_perf": None,
    "arming_start_wall": None,
    "last_seen_perf": None,
    "lock_acquire_perf": None,
    "lock_start_wall": None
}

def start_arming(tid):
    lock_state["mode"] = "arming"
    lock_state["id"] = tid
    lock_state["arming_start_perf"] = time.perf_counter()
    lock_state["arming_start_wall"] = time.time()
    lock_state["last_seen_perf"] = lock_state["arming_start_perf"]
    lock_state["lock_acquire_perf"] = None
    lock_state["lock_start_wall"] = None
    _log_text(f"Balloon ID: {tid} entered lock zone (arming started, t={lock_state['arming_start_perf']:.1f}s)")

def cancel_arming_as_failed(end_perf):
    """Arming phase ends without acquisition."""
    if lock_state["mode"] != "arming":
        return
    tid = lock_state["id"]
    arm_start_perf = lock_state["arming_start_perf"]
    arm_start_wall = lock_state["arming_start_wall"]
    arm_dur = end_perf - arm_start_perf if arm_start_perf is not None else 0.0

    _log_text(f"Balloon ID: {tid} arming duration: {arm_dur:.1f}s -> LOCK FAILED (did not acquire)")
    # JSON event for a failed attempt
    event = {
        "BalloonID": int(tid),
        "LockStartTime": None,  # not acquired
        "LockEndTime": datetime.fromtimestamp(arm_start_wall + arm_dur).isoformat() if arm_start_wall else None,
        "LockDurationSec": 0.0,
        "Result": "Failed",
        "ArmingDurationSec": round(arm_dur, 3)
    }
    _write_json_event(event)
    # reset
    for k in list(lock_state.keys()):
        lock_state[k] = None
    lock_state["mode"] = "idle"

def acquire_lock(now_perf):
    """Transition from arming -> locked."""
    if lock_state["mode"] != "arming":
        return
    tid = lock_state["id"]
    lock_state["mode"] = "locked"
    lock_state["lock_acquire_perf"] = now_perf
    lock_state["lock_start_wall"] = time.time()
    _log_text(f"Balloon ID: {tid} -> LOCK ACQUIRED")

def end_locked_session(end_perf):
    """End a locked session (target left beyond gap)."""
    if lock_state["mode"] != "locked":
        return
    tid = lock_state["id"]
    lock_start_perf = lock_state["lock_acquire_perf"]
    lock_start_wall = lock_state["lock_start_wall"]
    lock_dur = end_perf - lock_start_perf if lock_start_perf is not None else 0.0

    _log_text(f"Balloon ID: {tid} lock duration: {lock_dur:.1f}s -> LOCK ENDED")
    # JSON event for a completed lock session
    event = {
        "BalloonID": int(tid),
        "LockStartTime": datetime.fromtimestamp(lock_start_wall).isoformat() if lock_start_wall else None,
        "LockEndTime": datetime.fromtimestamp((lock_start_wall or time.time()) + lock_dur).isoformat(),
        "LockDurationSec": round(lock_dur, 3),
        "Result": "Successful"  # session acquired and completed (ended)
    }
    _write_json_event(event)

    # reset
    for k in list(lock_state.keys()):
        lock_state[k] = None
    lock_state["mode"] = "idle"

# ========= Main detection =========
def detect():
    cv.namedWindow("YOLOv11 Detection", cv.WINDOW_NORMAL)
    cv.resizeWindow("YOLOv11 Detection", WIDTH, HEIGHT)

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 30)
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass

    model = YOLO(MODELPATH)

    USE_GPU = torch.cuda.is_available()
    device = 0 if USE_GPU else None
    if USE_GPU:
        torch.backends.cudnn.benchmark = True
        model.to('cuda')

    fps = 0.0
    alpha_fps = 0.1
    last_t = time.perf_counter()

    while True:
        loop_t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        # inference + tracking
        t0 = time.perf_counter()
        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=CONF_T,
            iou=IOU_T,
            imgsz=IMGSZ,
            verbose=False,
            device=device,
            classes=[BALLOON_ID],
            max_det=MAX_DET,
            half=USE_GPU,
        )
        t1 = time.perf_counter()

        r0 = results[0]
        annotated = frame
        draw_lock_zone(annotated)

        # build outputs (same as before)
        boxes = r0.boxes
        outputs, top_threat = [], None

        if boxes is not None and len(boxes) > 0:
            xywh = boxes.xywh.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            ids  = boxes.id.detach().cpu().numpy() if boxes.id is not None else np.array([None]*len(xywh))

            for (x, y, w, h), s, tid in zip(xywh, conf, ids):
                if tid is None: continue
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
                if dA > AREA_T_POS: status = "Approaching"
                elif dA < -AREA_T_NEG: status = "Receding"
                else: status = "Stable Distance"
                size_idx = size_index_from_Ahat(Ahat)

                S_size = clip01(Ahat / EXPECTED_MAX_AHAT)
                S_center = center_score(cx, cy, WIDTH, HEIGHT)
                S_approach = 1.0 if status == "Approaching" else (0.5 if status == "Stable Distance" else 0.0)
                speed = math.hypot(vx, vy)
                S_vel = sigmoid(speed / 5.0)
                threat = (W_SIZE * S_size + W_CENTER * S_center + W_APPROACH * S_approach + W_VEL * S_vel)

                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                bbox_xywh = (float(x), float(y), float(w), float(h))

                outputs.append({
                    "id": tid,
                    "conf": float(s),
                    "area": float(A),
                    "Ahat": float(Ahat),
                    "bbox": (x1, y1, x2, y2),
                    "bbox_xywh": bbox_xywh,
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
                    "in_lock_zone": point_in_lock_zone((cx, cy))
                })

            outputs.sort(key=lambda o: o["threat_score"], reverse=True)
            for idx, o in enumerate(outputs, start=1):
                o["rank"] = idx
            top_threat = outputs[0] if outputs else None

        # ===== Two-Phase Lock Logic =====
        now_perf = time.perf_counter()

        # Choose a candidate only if idle (no preemption to avoid flip-flops)
        if lock_state["mode"] == "idle":
            in_zone = [o for o in outputs if o.get("in_lock_zone")]
            if in_zone:
                in_zone.sort(key=lambda o: o["lock_priority"], reverse=True)
                start_arming(in_zone[0]["id"])

        elif lock_state["mode"] == "arming":
            # track current arming target presence
            aid = lock_state["id"]
            cur = next((o for o in outputs if o["id"] == aid), None)
            if cur is not None and cur.get("in_lock_zone"):
                lock_state["last_seen_perf"] = now_perf
                arm_elapsed = now_perf - lock_state["arming_start_perf"]
                # acquire once arming time reached
                if arm_elapsed >= LOCK_ARM_SEC:
                    acquire_lock(now_perf)
            else:
                # tolerate a short dropout
                last_seen = lock_state["last_seen_perf"] or lock_state["arming_start_perf"]
                if (now_perf - last_seen) > GAP_TOLERANCE_SEC:
                    cancel_arming_as_failed(now_perf)

        elif lock_state["mode"] == "locked":
            aid = lock_state["id"]
            cur = next((o for o in outputs if o["id"] == aid), None)
            if cur is not None and cur.get("in_lock_zone"):
                lock_state["last_seen_perf"] = now_perf
            else:
                last_seen = lock_state["last_seen_perf"] or lock_state["lock_acquire_perf"]
                if (now_perf - last_seen) > GAP_TOLERANCE_SEC:
                    end_locked_session(now_perf)

        # ===== Drawing / HUD =====
        # Draw boxes with status coloring:
        # - locked target: GREEN + "LOCKED" badge
        # - arming target: ORANGE + arming countdown
        # - others: YELLOW
        for o in outputs:
            x1, y1, x2, y2 = o["bbox"]
            color = (0, 255, 255)  # default: yellow
            label = f"ID {o['id']} {o['size_index']}"

            if lock_state["mode"] == "locked" and o["id"] == lock_state["id"]:
                color = (0, 200, 0)  # green
                label = f"LOCKED ID {o['id']}"
            elif lock_state["mode"] == "arming" and o["id"] == lock_state["id"]:
                color = (0, 165, 255)  # orange (BGR)
                arm_elapsed = time.perf_counter() - (lock_state["arming_start_perf"] or time.perf_counter())
                remaining = max(0.0, LOCK_ARM_SEC - arm_elapsed)
                label = f"ARMING ID {o['id']} ({remaining:.1f}s)"

            cv.rectangle(annotated, (x1, y1), (x2, y2), color, 3, cv.LINE_AA)
            cv.putText(annotated, label, (x1, max(20, y1-8)), FONT, 0.65, color, 2, cv.LINE_AA)

            # while locked, show duration on the target
            if lock_state["mode"] == "locked" and o["id"] == lock_state["id"]:
                dur = time.perf_counter() - (lock_state["lock_acquire_perf"] or time.perf_counter())
                cv.putText(annotated, f"lock {dur:.1f}s", (x1, y2 + 20), FONT, 0.6, (0, 200, 0), 2, cv.LINE_AA)

        # Draw lock zone and arming progress ring/bar
        draw_lock_zone(annotated)
        if lock_state["mode"] == "arming":
            arm_elapsed = time.perf_counter() - (lock_state["arming_start_perf"] or time.perf_counter())
            progress = clip01(arm_elapsed / LOCK_ARM_SEC)
            if LOCK_SHAPE == "circle":
                end_angle = int(360 * progress)
                cv.ellipse(annotated, CENTER, (LOCK_RADIUS+10, LOCK_RADIUS+10), 0, 0, end_angle, (0,165,255), 3, cv.LINE_AA)
            else:
                half = LOCK_SIDE // 2
                bar_w = int(LOCK_SIDE * progress)
                cv.rectangle(annotated, (CENTER[0]-half, CENTER[1]+half+8),
                             (CENTER[0]-half+bar_w, CENTER[1]+half+18), (0,165,255), -1, cv.LINE_AA)
            cv.putText(annotated, f"ARMING {arm_elapsed:.1f}/{LOCK_ARM_SEC:.0f}s",
                       (CENTER[0]-140, CENTER[1]-(LOCK_RADIUS if LOCK_SHAPE=='circle' else LOCK_SIDE//2)-20),
                       FONT, 0.7, (0,165,255), 2, cv.LINE_AA)

        elif lock_state["mode"] == "locked":
            dur = time.perf_counter() - (lock_state["lock_acquire_perf"] or time.perf_counter())
            if LOCK_SHAPE == "circle":
                cv.ellipse(annotated, CENTER, (LOCK_RADIUS+10, LOCK_RADIUS+10), 0, 0, 360, (0,200,0), 3, cv.LINE_AA)
            else:
                half = LOCK_SIDE // 2
                cv.rectangle(annotated, (CENTER[0]-half, CENTER[1]+half+8),
                             (CENTER[0]+half, CENTER[1]+half+18), (0,200,0), 2, cv.LINE_AA)
            cv.putText(annotated, f"LOCKED (duration {dur:.1f}s)",
                       (CENTER[0]-160, CENTER[1]-(LOCK_RADIUS if LOCK_SHAPE=='circle' else LOCK_SIDE//2)-20),
                       FONT, 0.7, (0,200,0), 2, cv.LINE_AA)

        # FPS + diagnostics
        t2 = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, (t2 - last_t))
        last_t = t2
        fps = (1 - alpha_fps) * fps + alpha_fps * inst_fps if fps > 0 else inst_fps

        hud_lines = [
            f"FPS: {fps:5.1f}",
            f"Times ms  det+track={(t1-t0)*1000:5.1f}  post={(t2-t1)*1000:5.1f}  loop={(t2-loop_t0)*1000:5.1f}",
            f"State: {lock_state['mode']}  ActiveID: {lock_state['id']}"
        ]
        if outputs:
            hud_lines.append(f"Detections: {len(outputs)}  TopThreat: {top_threat['threat_score']:.3f}")
            hud_colors = [(255,255,255)]*4
            for o in outputs:
                hud_lines.append(
                    f"R{o['rank']:d} ID{o['id']} thr={o['threat_score']:.3f} lockP={o['lock_priority']:.3f} "
                    f"conf={o['conf']:.2f} sz={o['size_index']} pos={o['position_label']} C{tuple(o['centroid'])} "
                    f"{'IN-LOCK' if o['in_lock_zone'] else ''}"
                )
                hud_colors.append((0,200,0) if (lock_state["mode"] == "locked" and lock_state["id"] == o["id"])
                                  else ((0,165,255) if (lock_state["mode"] == "arming" and lock_state["id"] == o["id"])
                                        else (255,255,255)))
        else:
            hud_lines.append("No balloons detected")
            hud_colors = [(255,255,255)]*3

        draw_hud_bottom_left(annotated, hud_lines, hud_colors)

        cv.imshow("YOLOv11 Detection", annotated)

        if (cv.waitKey(1) & 0xFF) == ord("q"):
            # finalize any ongoing attempt/session gracefully
            now = time.perf_counter()
            if lock_state["mode"] == "arming":
                cancel_arming_as_failed(now)
            elif lock_state["mode"] == "locked":
                end_locked_session(now)
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    detect()
