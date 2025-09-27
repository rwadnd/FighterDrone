import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow duplicate OpenMP libs if present

import cv2 as cv
import math, time, sys, signal
import numpy as np
from collections import deque
from multiprocessing import Process, Queue, Event, set_start_method
import queue as pyqueue

# -------------------- Tunables (quick to tweak) --------------------
WIDTH, HEIGHT = 1280, 720        # camera resolution
IMGSZ = 640                      # YOLO input size
BALLOON_ID = 1                   # target class id
TRACKER_CFG = "bytetrack.yaml"   # tracker config file
CONF_T, IOU_T = 0.5, 0.45        # detection thresholds
MAX_DET = 3                      # max detections per frame

CENTER = (WIDTH // 2, HEIGHT // 2)  # screen center for overlays

K_HISTORY = 5                   # tracking history length
V_MIN = 1.5                     # min speed to call it “moving”
AREA_T_POS = 400.0             # area delta -> approaching
AREA_T_NEG = 400.0             # area delta -> receding
CENTER_MARGIN_X = 0.10         # central box (x) for “Center”
CENTER_MARGIN_Y = 0.10         # central box (y) for “Center”
W_SIZE, W_CENTER, W_APPROACH, W_VEL = 0.45, 0.30, 0.20, 0.05  # threat weights
EXPECTED_MAX_AHAT = 0.08       # area fraction that maps to “large”

FONT = cv.FONT_HERSHEY_SIMPLEX  # HUD font
FONT_SCALE = 0.6
FONT_THICK = 1
LINE_SPACING = 20
PANEL_PAD_X = 10
PANEL_PAD_Y = 10

JPEG_QUALITY = 85              # IPC: JPEG quality for result frames
WINDOW_NAME = "YOLOv11 Multiprocess Detection"

# -------------------- Helpers (UI-side) --------------------
def draw_translucent_panel(img, x, y, w, h, color=(0,0,0), alpha=0.45):
    overlay = img.copy()
    cv.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_hud_bottom_left(img, lines, colors=None):
    # draw a compact text panel bottom-left
    max_text_w, text_h = 0, 0
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
    for i, ln in enumerate(lines):
        col = (255, 255, 255)
        if colors is not None and i < len(colors) and colors[i] is not None:
            col = colors[i]
        cv.putText(img, ln, (x + PANEL_PAD_X, baseline_y), FONT, FONT_SCALE, col, FONT_THICK, cv.LINE_AA)
        baseline_y += LINE_SPACING

# -------------------- Capture Process --------------------
def capture_proc(frame_q: Queue, stop_ev: Event):
    # grab frames as fast as possible; keep queue tiny for low latency
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 30)
    try:
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass

    while not stop_ev.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        if frame_q.full():
            # drop oldest to keep freshest frame
            try: _ = frame_q.get_nowait()
            except Exception: pass
        try:
            frame_q.put_nowait(frame)
        except pyqueue.Full:
            try: _ = frame_q.get_nowait()
            except pyqueue.Empty: pass
            try: frame_q.put_nowait(frame)
            except pyqueue.Full: pass

    cap.release()

# -------------------- Inference Process --------------------
def inference_proc(frame_q: Queue, result_q: Queue, stop_ev: Event):
    # import heavy libs inside the process
    import torch
    from ultralytics import YOLO

    # model/device
    model = YOLO("lastv6.pt")
    USE_GPU = torch.cuda.is_available()
    device = 0 if USE_GPU else None
    if USE_GPU:
        torch.backends.cudnn.benchmark = True
        model.to('cuda')

    # per-id state
    track_db = {}

    class TrackState:
        def __init__(self):
            self.centroids = deque(maxlen=K_HISTORY)
            self.areas = deque(maxlen=K_HISTORY)

    # small utilities (local to this process)
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
        if -22.5 <= ang < 22.5: return "Moving Right"
        if 22.5 <= ang < 67.5: return "Moving Up-Right"
        if 67.5 <= ang < 112.5: return "Moving Up"
        if 112.5 <= ang < 157.5: return "Moving Up-Left"
        if ang >= 157.5 or ang < -157.5: return "Moving Left"
        if -157.5 <= ang < -112.5: return "Moving Down-Left"
        if -112.5 <= ang < -67.5: return "Moving Down"
        return "Moving Down-Right"

    def sector_label(cx, cy, W, H, mx=CENTER_MARGIN_X, my=CENTER_MARGIN_Y):
        cx0, cy0 = W // 2, H // 2
        if abs(cx - cx0) <= mx * W and abs(cy - cy0) <= my * H: return "Center"
        top = cy < cy0; left = cx < cx0
        if top and left: return "Top-Left"
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

    last_t = time.perf_counter()

    while not stop_ev.is_set():
        try:
            frame = frame_q.get(timeout=0.05)
        except Exception:
            continue

        frame = cv.flip(frame, 1)

        # run detector+tracker
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
        outputs, top_threat = [], None

        if boxes is not None and len(boxes) > 0:
            xywh = boxes.xywh.detach().cpu().numpy()
            conf = boxes.conf.detach().cpu().numpy()
            ids = boxes.id.detach().cpu().numpy() if boxes.id is not None else np.array([None] * len(xywh))

            for (x, y, w, h), s, tid in zip(xywh, conf, ids):
                if tid is None: continue
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
                    "threat_score": float(round(threat, 3))
                })

                cv.putText(annotated, f"ID {tid} {size_idx}", (x1, max(20, y1 - 6)),
                           FONT, 0.6, (0, 255, 255), 2, cv.LINE_AA)

            # rank by threat and mark top
            outputs.sort(key=lambda o: o["threat_score"], reverse=True)
            for idx, o in enumerate(outputs, start=1):
                o["rank"] = idx
            top_threat = outputs[0] if outputs else None

            # draw boxes after ranking
            if outputs:
                top_id = top_threat["id"] if top_threat is not None else None
                for o in outputs:
                    bx1, by1, bx2, by2 = o["bbox"]
                    color = (0, 0, 255) if o["id"] == top_id else (0, 255, 255)
                    thickness = 3 if o["id"] == top_id else 2
                    cv.rectangle(annotated, (bx1, by1), (bx2, by2), color, thickness, cv.LINE_AA)
                    if o["id"] == top_id:
                        cv.putText(annotated, f"TOP ID {o['id']}", (bx1, max(20, by1 - 26)),
                                   FONT, 0.6, (0, 0, 255), 2, cv.LINE_AA)

        t2 = time.perf_counter()
        det_ms = (t1 - t0) * 1000.0
        post_ms = (t2 - t1) * 1000.0

        # encode to JPEG for light IPC
        ok, jpg = cv.imencode(".jpg", annotated, [int(cv.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue

        payload = {
            "jpg": jpg,                                # np.uint8 buffer
            "outputs": outputs,                        # sorted, contains rank
            "top_threat": top_threat,                  # dict or None
            "timings": {"det+track": det_ms, "post": post_ms}
        }

        # push latest; drop old if needed
        if result_q.full():
            try: _ = result_q.get_nowait()
            except Exception: pass
        try:
            result_q.put_nowait(payload)
        except pyqueue.Full:
            try: _ = result_q.get_nowait()
            except pyqueue.Empty: pass
            try: result_q.put_nowait(payload)
            except pyqueue.Full: pass

# -------------------- Main / UI Process --------------------
def main():
    # ensure Windows-safe start method
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # small queues keep latency low
    frame_q = Queue(maxsize=2)
    result_q = Queue(maxsize=2)
    stop_ev = Event()

    cap_p = Process(target=capture_proc, args=(frame_q, stop_ev), daemon=True)
    inf_p = Process(target=inference_proc, args=(frame_q, result_q, stop_ev), daemon=True)

    cap_p.start()
    inf_p.start()

    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
    cv.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

    # UI-side FPS
    fps = 0.0
    alpha_fps = 0.1
    last_t = time.perf_counter()

    def handle_sigint(sig, frame):
        stop_ev.set()
    signal.signal(signal.SIGINT, handle_sigint)

    while True:
        # fetch latest result quickly; keep UI responsive
        try:
            payload = result_q.get(timeout=0.03)
        except Exception:
            now = time.perf_counter()
            inst = 1.0 / max(1e-6, now - last_t)
            last_t = now
            fps = (1 - alpha_fps) * fps + alpha_fps * inst if fps > 0 else inst
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        jpg = payload["jpg"]
        outputs = payload["outputs"]
        top_threat = payload["top_threat"]
        timings = payload["timings"]

        annotated = cv.imdecode(jpg, cv.IMREAD_COLOR)

        # UI FPS update
        now = time.perf_counter()
        inst = 1.0 / max(1e-6, (now - last_t))
        last_t = now
        fps = (1 - alpha_fps) * fps + alpha_fps * inst if fps > 0 else inst

        # build HUD lines
        hud_lines = [
            f"FPS: {fps:5.1f}",
            f"Times ms  det+track={timings['det+track']:5.1f}  post={timings['post']:5.1f}"
        ]
        if outputs:
            hud_lines.append(f"Detections: {len(outputs)}  TopThreat: {top_threat['threat_score']:.3f}")
            hud_colors = [(255,255,255), (255,255,255), (255,255,255)]
            for o in outputs:
                hud_lines.append(
                    f"R{o['rank']:d} ID{o['id']} thr={o['threat_score']:.3f} conf={o['conf']:.2f} "
                    f"sz={o['size_index']} st={o['status']} pos={o['position_label']} C{tuple(o['centroid'])}"
                )
                hud_colors.append((0,0,255) if o.get("rank", 999) == 1 else (255,255,255))
        else:
            hud_lines.append("No balloons detected")
            hud_colors = [(255,255,255), (255,255,255)]

        draw_hud_bottom_left(annotated, hud_lines, hud_colors)
        cv.imshow(WINDOW_NAME, annotated)

        if (cv.waitKey(1) & 0xFF) == ord('q'):
            break

    # shutdown children and UI
    stop_ev.set()
    try:
        cap_p.join(timeout=1.0)
        inf_p.join(timeout=1.0)
    except Exception:
        pass
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
