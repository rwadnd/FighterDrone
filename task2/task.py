import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow duplicate OpenMP libs (fix crash)

import cv2 as cv
from ultralytics import YOLO
import torch, math, time
from collections import deque
import numpy as np

# ==== Settings ====
WIDTH, HEIGHT = 1280, 720        # webcam resolution
IMGSZ = 640                      # YOLO input size
BALLOON_ID = 1                   # target class ID
TRACKER_CFG = "bytetrack.yaml"   # tracker config
CONF_T, IOU_T = 0.5, 0.45        # detection thresholds
MAX_DET = 3                      # max detections/frame
CENTER = (WIDTH // 2, HEIGHT // 2)

# ==== Tracking params ====
K_HISTORY = 5                   # tracking history length
V_MIN = 1.5                     # min speed to call it “moving”
AREA_T_POS = 400.0             # area delta -> approaching
AREA_T_NEG = 400.0             # area delta -> receding
CENTER_MARGIN_X = 0.10         # central box (x) for “Center”
CENTER_MARGIN_Y = 0.10         # central box (y) for “Center”
W_SIZE, W_CENTER, W_APPROACH, W_VEL = 0.45, 0.30, 0.20, 0.05  # threat weights
EXPECTED_MAX_AHAT = 0.08       # area fraction that maps to “large”

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


def size_index_from_Ahat(Ahat: float) -> str:
    if Ahat < 0.01: return "Small"
    if Ahat < 0.04: return "Medium"
    return "Large"


def sector_label(cx, cy, W, H, mx=CENTER_MARGIN_X, my=CENTER_MARGIN_Y):
    # classify position relative to screen center
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
    # mean velocity from centroid history
    if len(centroids) < 2: return 0.0, 0.0
    dx = sum(centroids[i+1][0] - centroids[i][0] for i in range(len(centroids)-1)) / (len(centroids)-1)
    dy = sum(centroids[i+1][1] - centroids[i][1] for i in range(len(centroids)-1)) / (len(centroids)-1)
    return dx, dy


def dir_label_from_v(vx, vy):
    # convert velocity vector to direction label
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
    # median difference between values in sequence
    if len(seq) < 2: return 0.0
    deltas = [seq[i+1] - seq[i] for i in range(len(seq)-1)]
    deltas.sort()
    m = len(deltas) // 2
    return deltas[m] if len(deltas) % 2 else 0.5 * (deltas[m-1] + deltas[m])


def center_score(cx, cy, W, H):
    # score closeness to center
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
    # draw info panel bottom-left
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


def detect():
    # main detection loop
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

    model = YOLO("lastv6.pt")

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

        # run YOLO + tracker
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
        cv.circle(annotated, CENTER, 3, (0, 255, 0), -1, cv.LINE_AA)

        # process detections
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
                    "threat_score": float(round(threat, 3))
                })

                cv.putText(annotated, f"ID {tid} {size_idx}", (x1, max(20, y1-6)),
                           FONT, 0.6, (0,255,255), 2, cv.LINE_AA)

            outputs.sort(key=lambda o: o["threat_score"], reverse=True)
            for idx, o in enumerate(outputs, start=1):
                o["rank"] = idx
            top_threat = outputs[0] if outputs else None

            if outputs:
                top_id = top_threat["id"]
                for o in outputs:
                    bx1, by1, bx2, by2 = o["bbox"]
                    color = (0,0,255) if o["id"] == top_id else (0,255,255)
                    thickness = 3 if o["id"] == top_id else 2
                    cv.rectangle(annotated, (bx1, by1), (bx2, by2), color, thickness, cv.LINE_AA)
                    if o["id"] == top_id:
                        cv.putText(annotated, f"TOP ID {o['id']}", (bx1, max(20, by1-26)),
                                   FONT, 0.6, (0,0,255), 2, cv.LINE_AA)

        # FPS + HUD
        t2 = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, (t2 - last_t))
        last_t = t2
        fps = (1 - alpha_fps) * fps + alpha_fps * inst_fps if fps > 0 else inst_fps

        hud_lines = [f"FPS: {fps:5.1f}",
                     f"Times ms  det+track={(t1-t0)*1000:5.1f}  post={(t2-t1)*1000:5.1f}  loop={(t2-loop_t0)*1000:5.1f}"]
        if outputs:
            hud_lines.append(f"Detections: {len(outputs)}  TopThreat: {top_threat['threat_score']:.3f}")
            hud_colors = [(255,255,255)]*3
            for o in outputs:
                hud_lines.append(
                    f"R{o['rank']:d} ID{o['id']} thr={o['threat_score']:.3f} conf={o['conf']:.2f} "
                    f"sz={o['size_index']} st={o['status']} pos={o['position_label']} C{tuple(o['centroid'])}"
                )
                hud_colors.append((0,0,255) if o.get("rank", 999) == 1 else (255,255,255))
        else:
            hud_lines.append("No balloons detected")
            hud_colors = [(255,255,255)]*2

        draw_hud_bottom_left(annotated, hud_lines, hud_colors)
        cv.imshow("YOLOv11 Detection", annotated)

        if (cv.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    detect()
