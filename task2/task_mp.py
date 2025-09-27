import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # TEMP; fix env later

import cv2 as cv
import math, time, sys, signal
import numpy as np
from collections import deque
from multiprocessing import Process, Queue, Event, set_start_method

# -------------------- Tunables --------------------
WIDTH, HEIGHT = 1280, 720
IMGSZ = 640
BALLOON_ID = 1
TRACKER_CFG = "bytetrack.yaml"
CONF_T, IOU_T = 0.5, 0.45
MAX_DET = 3

CENTER = (WIDTH // 2, HEIGHT // 2)

K_HISTORY = 5
V_MIN = 1.5
AREA_T_POS = 400.0
AREA_T_NEG = 400.0
CENTER_MARGIN_X = 0.10
CENTER_MARGIN_Y = 0.10
W_SIZE, W_CENTER, W_APPROACH, W_VEL = 0.45, 0.30, 0.20, 0.05
EXPECTED_MAX_AHAT = 0.08

FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICK = 1
LINE_SPACING = 20
PANEL_PAD_X = 10
PANEL_PAD_Y = 10

# JPEG encoding quality for inter-process transfer
JPEG_QUALITY = 85

WINDOW_NAME = "YOLOv11 Multiprocess Detection"

# -------------------- Helpers (UI-side) --------------------
def draw_translucent_panel(img, x, y, w, h, color=(0,0,0), alpha=0.45):
    overlay = img.copy()
    cv.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_hud_bottom_left(img, lines):
    max_text_w = 0; text_h = 0
    for ln in lines:
        (tw, th), _ = cv.getTextSize(ln, FONT, FONT_SCALE, FONT_THICK)
        max_text_w = max(max_text_w, tw)
        text_h = max(text_h, th)
    panel_w = max_text_w + 2 * PANEL_PAD_X
    panel_h = PANEL_PAD_Y * 2 + len(lines) * LINE_SPACING
    x = 10
    y = img.shape[0] - 10 - panel_h
    draw_translucent_panel(img, x, y, panel_w, panel_h)
    baseline_y = y + PANEL_PAD_Y + text_h
    for ln in lines:
        cv.putText(img, ln, (x + PANEL_PAD_X, baseline_y), FONT, FONT_SCALE,
                   (255,255,255), FONT_THICK, cv.LINE_AA)
        baseline_y += LINE_SPACING

# -------------------- Capture Process --------------------
def capture_proc(frame_q: Queue, stop_ev: Event):
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 30)
    try:
        fourcc = cv.VideoWriter_fourcc(*'MJPG')
        cap.set(cv.CAP_PROP_FOURCC, fourcc)
    except Exception:
        pass

    while not stop_ev.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        # Keep latency low: drop oldest if full
        if frame_q.full():
            try:
                _ = frame_q.get_nowait()
            except Exception:
                pass
        frame_q.put(frame, block=False)
    cap.release()

# -------------------- Inference Process --------------------
def inference_proc(frame_q: Queue, result_q: Queue, stop_ev: Event):
    # Lazy import torch/ultralytics here, inside the process
    import torch
    from ultralytics import YOLO

    # Model
    model = YOLO("lastv6.pt")
    USE_GPU = torch.cuda.is_available()
    device = 0 if USE_GPU else None
    if USE_GPU:
        torch.backends.cudnn.benchmark = True
        model.to('cuda')

    # State
    track_db = {}  # id -> TrackState
    class TrackState:
        def __init__(self):
            self.centroids = deque(maxlen=K_HISTORY)
            self.areas = deque(maxlen=K_HISTORY)

    # Utility funcs (duplicated here to avoid cross-proc refs)
    def avg_velocity(centroids):
        if len(centroids) < 2: return 0.0, 0.0
        dx = sum(centroids[i+1][0] - centroids[i][0] for i in range(len(centroids)-1)) / (len(centroids)-1)
        dy = sum(centroids[i+1][1] - centroids[i][1] for i in range(len(centroids)-1)) / (len(centroids)-1)
        return dx, dy

    def dir_label_from_v(vx, vy):
        vy = -vy
        speed = math.hypot(vx, vy)
        if speed < V_MIN: return "Stationary"
        ang = math.degrees(math.atan2(vy, vx))
        if -22.5 <= ang < 22.5:    return "Moving Right"
        if 22.5 <= ang < 67.5:     return "Moving Up-Right"
        if 67.5 <= ang < 112.5:    return "Moving Up"
        if 112.5 <= ang < 157.5:   return "Moving Up-Left"
        if ang >= 157.5 or ang < -157.5: return "Moving Left"
        if -157.5 <= ang < -112.5: return "Moving Down-Left"
        if -112.5 <= ang < -67.5:  return "Moving Down"
        return "Moving Down-Right"

    def sector_label(cx, cy, W, H, mx=CENTER_MARGIN_X, my=CENTER_MARGIN_Y):
        cx0, cy0 = W // 2, H // 2
        if abs(cx - cx0) <= mx * W and abs(cy - cy0) <= my * H:
            return "Center"
        top = cy < cy0; left = cx < cx0
        if top and left:  return "Top-Left"
        if top and not left: return "Top-Right"
        if not top and left: return "Bottom-Left"
        return "Bottom-Right"

    def size_index_from_Ahat(Ahat: float) -> str:
        if Ahat < 0.01: return "Small"
        if Ahat < 0.04: return "Medium"
        return "Large"

    def center_score(cx, cy, W, H):
        dx = (cx - W/2) / (W/2)
        dy = (cy - H/2) / (H/2)
        d = min(1.0, math.hypot(dx, dy))
        return 1.0 - d

    def clip01(x): return max(0.0, min(1.0, x))
    def sigmoid(x): return 1.0 / (1.0 + math.exp(-x))
    def median_delta(seq):
        if len(seq) < 2: return 0.0
        deltas = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
        deltas.sort()
        m = len(deltas) // 2
        return deltas[m] if len(deltas) % 2 else 0.5 * (deltas[m-1] + deltas[m])

    # FPS/Timings inside inference
    last_t = time.perf_counter()

    while not stop_ev.is_set():
        try:
            frame = frame_q.get(timeout=0.05)
        except Exception:
            continue

        frame = cv.flip(frame, 1)
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
        annotated = frame.copy()
        cv.circle(annotated, CENTER, 3, (0, 255, 0), -1, cv.LINE_AA)

        boxes = r0.boxes
        outputs = []
        top_threat = None

        if boxes is not None and len(boxes) > 0:
            xywh = boxes.xywh.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            ids  = boxes.id.detach().cpu().numpy() if boxes.id is not None else np.array([None]*len(xywh))

            for (x, y, w, h), s, tid in zip(xywh, conf, ids):
                if tid is None:
                    continue
                tid = int(tid)
                cx, cy = int(x), int(y)
                A = float(w) * float(h)
                Ahat = A / float(WIDTH * HEIGHT)

                st = track_db.get(tid)
                if st is None:
                    st = TrackState()
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
                threat = (W_SIZE * S_size +
                          W_CENTER * S_center +
                          W_APPROACH * S_approach +
                          W_VEL * S_vel)

                outputs.append({
                    "id": tid,
                    "position_label": pos_label,
                    "centroid": (cx, cy),
                    "status": status,
                    "size_index": size_idx,
                    "movement": move_label,
                    "threat_score": round(threat, 3)
                })

                # Minimal drawing
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv.rectangle(annotated, (x1, y1), (x2, y2), (0,255,255), 2, cv.LINE_AA)
                cv.putText(annotated, f"ID {tid} {size_idx}", (x1, max(20, y1-6)),
                           FONT, 0.6, (0,255,255), 2, cv.LINE_AA)


            if outputs:
                top_threat = max(outputs, key=lambda o: o["threat_score"])
               

        t2 = time.perf_counter()
        det_ms = (t1 - t0) * 1000.0
        post_ms = (t2 - t1) * 1000.0

        # JPEG encode annotated frame to reduce IPC overhead
        ok, jpg = cv.imencode(".jpg", annotated, [int(cv.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue

        # Package result payload (lightweight metadata + jpg bytes)
        payload = {
            "jpg": jpg,                 # numpy 1D uint8
            "outputs": outputs,         # list of dicts
            "top_threat": top_threat,   # dict or None
            "timings": {"det+track": det_ms, "post": post_ms}
        }

        # If queue is full, drop the oldest to keep latency small
        if result_q.full():
            try:
                _ = result_q.get_nowait()
            except Exception:
                pass
        result_q.put(payload, block=False)

    # Cleanup
    # (nothing else to release here)

# -------------------- Main / UI Process --------------------
def main():
    # Windows needs 'spawn'
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # Queues with very small buffers to keep latency low
    frame_q = Queue(maxsize=2)
    result_q = Queue(maxsize=2)
    stop_ev = Event()

    cap_p = Process(target=capture_proc, args=(frame_q, stop_ev), daemon=True)
    inf_p = Process(target=inference_proc, args=(frame_q, result_q, stop_ev), daemon=True)

    cap_p.start()
    inf_p.start()

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    # FPS (display-side)
    fps = 0.0
    alpha_fps = 0.1
    last_t = time.perf_counter()

    def handle_sigint(sig, frame):
        stop_ev.set()
    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        # Pull latest result; if none quickly, just continue loop (keeps UI responsive)
        try:
            payload = result_q.get(timeout=0.03)
        except Exception:
            # still update FPS with idle time for smoother display number
            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - last_t)
            last_t = now
            fps = (1-alpha_fps)*fps + alpha_fps*inst if fps > 0 else inst
            # Show a placeholder if desired
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        jpg = payload["jpg"]
        outputs = payload["outputs"]
        top_threat = payload["top_threat"]
        timings = payload["timings"]

        # Decode JPEG back to image
        annotated = cv.imdecode(jpg, cv.IMREAD_COLOR)

        # FPS calc (UI loop)
        now = time.perf_counter()
        inst = 1.0 / max(1e-6, (now - last_t))
        last_t = now
        fps = (1 - alpha_fps) * fps + alpha_fps * inst if fps > 0 else inst

        # HUD
        hud_lines = [
            f"FPS: {fps:5.1f}",
            f"Times ms  det+track={timings['det+track']:5.1f}  post={timings['post']:5.1f}"
        ]
        if outputs:
            for o in outputs:
                hud_lines.append(f"ID {o['id']} | {o['position_label']} | C{tuple(o['centroid'])}")
                hud_lines.append(f"  {o['status']} | {o['size_index']} | {o['movement']}")
                hud_lines.append(f"  Threat: {o['threat_score']:.3f}")
            if top_threat:
                hud_lines.append("--- Highest Priority Threat ---")
                hud_lines.append(f"Target ID {top_threat['id']} "
                                 f"({top_threat['size_index']}, {top_threat['status']}, {top_threat['position_label']})")
        else:
            hud_lines.append("No balloons detected")

        draw_hud_bottom_left(annotated, hud_lines)

        cv.imshow(WINDOW_NAME, annotated)
        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    # Shutdown
    stop_ev.set()
    try:
        cap_p.join(timeout=1.0)
        inf_p.join(timeout=1.0)
    except Exception:
        pass
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
