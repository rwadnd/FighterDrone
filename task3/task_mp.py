import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2 as cv
import torch, math, time, json, multiprocessing as mp
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# ================== Config ==================
WIDTH, HEIGHT = 1280, 720
IMGSZ = 640
BALLOON_ID = 1
TRACKER_CFG = "bytetrack.yaml"
CONF_T, IOU_T = 0.5, 0.45
MAX_DET = 3
CENTER = (WIDTH // 2, HEIGHT // 2)
MODELPATH = "task3/lastv6.pt"

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

# Logging
LOG_TXT_PATH = "task3/lock_log.txt"
LOG_JSON_PATH = "task3/lock_events.json"  # JSON Lines

# HUD
FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICK = 1
LINE_SPACING = 20
PANEL_PAD_X = 10
PANEL_PAD_Y = 10

# Queues
FRAME_QUEUE_MAX = 2         # small to minimize latency
RESULT_QUEUE_MAX = 2
LOG_QUEUE_MAX = 100


# ================== Utilities ==================
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
        cv.circle(img, CENTER, LOCK_RADIUS, (255, 255, 255), 2, cv.LINE_AA)
    else:
        half = LOCK_SIDE // 2
        cv.rectangle(img, (CENTER[0]-half, CENTER[1]-half), (CENTER[0]+half, CENTER[1]+half), (255,255,255), 2, cv.LINE_AA)

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


# ================== Logging Process ==================
def logger_process(log_q: mp.Queue, stop_ev: mp.Event):
    txt_path = LOG_TXT_PATH
    json_path = LOG_JSON_PATH
    txt_f = open(txt_path, "a", encoding="utf-8")
    json_f = open(json_path, "a", encoding="utf-8")
    try:
        while not stop_ev.is_set():
            try:
                item = log_q.get(timeout=0.1)
            except:
                continue
            t = item.get("type")
            if t == "text":
                txt_f.write(item["msg"] + "\n")
                txt_f.flush()
                print(item["msg"])
            elif t == "event":
                json_f.write(json.dumps(item["obj"], ensure_ascii=False) + "\n")
                json_f.flush()
            elif t == "flush":
                txt_f.flush(); json_f.flush()
    finally:
        txt_f.close()
        json_f.close()


def log_text(log_q, line: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] {line}"
    log_q.put({"type": "text", "msg": msg})


def write_json_event(log_q, obj: dict):
    log_q.put({"type": "event", "obj": obj})


# ================== Capture Process ==================
def capture_process(frame_q: mp.Queue, stop_ev: mp.Event, cam_index=0):
    cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 30)
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass

    try:
        while not stop_ev.is_set():
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv.flip(frame, 1)
            # encode to JPEG to reduce queue load
            ok, buf = cv.imencode(".jpg", frame, [cv.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            # drop oldest if full (non-blocking put)
            if frame_q.full():
                try:
                    frame_q.get_nowait()
                except:
                    pass
            try:
                frame_q.put_nowait(buf.tobytes())
            except:
                pass
    finally:
        cap.release()


# ================== Inference Process ==================
class TrackState:
    def __init__(self):
        self.centroids = deque(maxlen=K_HISTORY)
        self.areas = deque(maxlen=K_HISTORY)

def inference_process(frame_q: mp.Queue, result_q: mp.Queue, stop_ev: mp.Event):
    # Init model in this process (CUDA context local to proc)
    model = YOLO(MODELPATH)
    use_gpu = torch.cuda.is_available()
    device = 0 if use_gpu else None
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        model.to('cuda')

    track_db = {}
    last_time = time.perf_counter()

    while not stop_ev.is_set():
        try:
            jpg_bytes = frame_q.get(timeout=0.1)
        except:
            continue
        # decode
        npbuf = np.frombuffer(jpg_bytes, dtype=np.uint8)
        frame = cv.imdecode(npbuf, cv.IMREAD_COLOR)
        if frame is None:
            continue

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
            half=use_gpu,
        )
        r0 = results[0]

        boxes = r0.boxes
        outputs = []
        top_threat = None

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

        t1 = time.perf_counter()

        # send result + the original JPEG for display in main
        # drop oldest if queue full
        if result_q.full():
            try:
                result_q.get_nowait()
            except:
                pass
        try:
            result_q.put_nowait({
                "jpg": jpg_bytes,
                "outputs": outputs,
                "top_threat": top_threat,
                "det_ms": (t1 - t0) * 1000.0
            })
        except:
            pass


# ================== Main (UI + FSM) ==================
class LockFSM:
    # mode: "idle" -> "arming" -> "locked" -> "idle"
    def __init__(self, log_q):
        self.mode = "idle"
        self.id = None
        self.arming_start_perf = None
        self.arming_start_wall = None
        self.last_seen_perf = None
        self.lock_acquire_perf = None
        self.lock_start_wall = None
        self.log_q = log_q

    def _log(self, s): log_text(self.log_q, s)
    def _event(self, obj): write_json_event(self.log_q, obj)

    def start_arming(self, tid):
        self.mode = "arming"
        self.id = tid
        self.arming_start_perf = time.perf_counter()
        self.arming_start_wall = time.time()
        self.last_seen_perf = self.arming_start_perf
        self.lock_acquire_perf = None
        self.lock_start_wall = None
        self._log(f"Balloon ID: {tid} entered lock zone (arming started, t={self.arming_start_perf:.1f}s)")

    def cancel_arming_as_failed(self, now_perf):
        if self.mode != "arming": return
        tid = self.id
        arm_dur = (now_perf - self.arming_start_perf) if self.arming_start_perf else 0.0
        self._log(f"Balloon ID: {tid} arming duration: {arm_dur:.1f}s -> LOCK FAILED (did not acquire)")
        self._event({
            "BalloonID": int(tid),
            "LockStartTime": None,
            "LockEndTime": datetime.fromtimestamp(self.arming_start_wall + arm_dur).isoformat() if self.arming_start_wall else None,
            "LockDurationSec": 0.0,
            "Result": "Failed",
            "ArmingDurationSec": round(arm_dur, 3)
        })
        self.__init__(self.log_q)

    def acquire_lock(self, now_perf):
        if self.mode != "arming": return
        tid = self.id
        self.mode = "locked"
        self.lock_acquire_perf = now_perf
        self.lock_start_wall = time.time()
        self._log(f"Balloon ID: {tid} -> LOCK ACQUIRED")

    def end_locked_session(self, now_perf):
        if self.mode != "locked": return
        tid = self.id
        lock_dur = (now_perf - self.lock_acquire_perf) if self.lock_acquire_perf else 0.0
        self._log(f"Balloon ID: {tid} lock duration: {lock_dur:.1f}s -> LOCK ENDED")
        self._event({
            "BalloonID": int(tid),
            "LockStartTime": datetime.fromtimestamp(self.lock_start_wall).isoformat() if self.lock_start_wall else None,
            "LockEndTime": datetime.fromtimestamp((self.lock_start_wall or time.time()) + lock_dur).isoformat(),
            "LockDurationSec": round(lock_dur, 3),
            "Result": "Successful"
        })
        self.__init__(self.log_q)


def main():
    mp.set_start_method("spawn", force=True)

    frame_q = mp.Queue(maxsize=FRAME_QUEUE_MAX)
    result_q = mp.Queue(maxsize=RESULT_QUEUE_MAX)
    log_q = mp.Queue(maxsize=LOG_QUEUE_MAX)
    stop_ev = mp.Event()

    # Start processes
    log_p = mp.Process(target=logger_process, args=(log_q, stop_ev), daemon=True)
    cap_p = mp.Process(target=capture_process, args=(frame_q, stop_ev, 0), daemon=True)
    inf_p = mp.Process(target=inference_process, args=(frame_q, result_q, stop_ev), daemon=True)

    log_p.start()
    cap_p.start()
    inf_p.start()

    fsm = LockFSM(log_q)

    cv.namedWindow("YOLOv11 Detection (MP)", cv.WINDOW_NORMAL)
    cv.resizeWindow("YOLOv11 Detection (MP)", WIDTH, HEIGHT)

    fps = 0.0
    alpha_fps = 0.1
    last_t = time.perf_counter()

    try:
        while True:
            try:
                pkt = result_q.get(timeout=0.1)
            except:
                pkt = None

            if pkt is None:
                # still draw empty screen if needed
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # decode JPEG from inference for display
            npbuf = np.frombuffer(pkt["jpg"], dtype=np.uint8)
            frame = cv.imdecode(npbuf, cv.IMREAD_COLOR)
            if frame is None:
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            outputs = pkt["outputs"]
            top_threat = pkt["top_threat"]
            det_ms = pkt["det_ms"]

            annotated = frame
            draw_lock_zone(annotated)

            # FSM update
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

            # Draw boxes & labels
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

                cv.rectangle(annotated, (x1, y1), (x2, y2), color, 3, cv.LINE_AA)
                cv.putText(annotated, label, (x1, max(20, y1-8)), FONT, 0.65, color, 2, cv.LINE_AA)

                if fsm.mode == "locked" and o["id"] == fsm.id:
                    dur = time.perf_counter() - (fsm.lock_acquire_perf or time.perf_counter())
                    cv.putText(annotated, f"lock {dur:.1f}s", (x1, y2 + 20), FONT, 0.6, (0, 200, 0), 2, cv.LINE_AA)

            # HUD & timing
            t2 = time.perf_counter()
            inst_fps = 1.0 / max(1e-6, (t2 - last_t))
            last_t = t2
            fps = (1 - alpha_fps) * fps + alpha_fps * inst_fps if fps > 0 else inst_fps

            hud_lines = [
                f"FPS: {fps:5.1f}",
                f"Times ms  det+track={det_ms:5.1f}",
                f"State: {fsm.mode}  ActiveID: {fsm.id}"
            ]
            if outputs:
                hud_lines.append(f"Detections: {len(outputs)}  TopThreat: {top_threat['threat_score']:.3f}" if top_threat else f"Detections: {len(outputs)}")
                hud_colors = [(255,255,255)] * len(hud_lines)
            else:
                hud_lines.append("No balloons detected")
                hud_colors = [(255,255,255)] * len(hud_lines)

            # Panel bottom-left
            # (reuse draw_hud_bottom_left implemented above)
            # compute panel dims and draw
            def _draw_hud(img, lines):
                max_text_w, text_h = 0, 0
                for ln in lines:
                    (tw, th), _ = cv.getTextSize(ln, FONT, FONT_SCALE, FONT_THICK)
                    max_text_w = max(max_text_w, tw)
                    text_h = max(text_h, th)
                panel_w = max_text_w + 2 * PANEL_PAD_X
                panel_h = PANEL_PAD_Y * 2 + len(lines) * LINE_SPACING
                x, y = 10, img.shape[0] - 10 - panel_h
                draw_translucent_panel(img, x, y, panel_w, panel_h, color=(0,0,0), alpha=0.45)
                baseline_y = y + PANEL_PAD_Y + text_h
                for ln in lines:
                    cv.putText(img, ln, (x + PANEL_PAD_X, baseline_y), FONT, FONT_SCALE, (255,255,255), FONT_THICK, cv.LINE_AA)
                    baseline_y += LINE_SPACING

            _draw_hud(annotated, hud_lines)

            cv.imshow("YOLOv11 Detection (MP)", annotated)

            if (cv.waitKey(1) & 0xFF) == ord('q'):
                # finalize any ongoing attempt/session
                now = time.perf_counter()
                if fsm.mode == "arming":
                    fsm.cancel_arming_as_failed(now)
                elif fsm.mode == "locked":
                    fsm.end_locked_session(now)
                break

    finally:
        # signal children to stop and flush logs
        stop_ev.set()
        try: log_q.put({"type": "flush"}); time.sleep(0.1)
        except: pass

        for p in (cap_p, inf_p, log_p):
            if p.is_alive():
                p.join(timeout=1.0)
                if p.is_alive():
                    p.terminate()

        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
