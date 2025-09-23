import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2 as cv
from ultralytics import YOLO
import torch, math, time
from collections import deque


WIDTH, HEIGHT = 1920, 1080
CENTER = (WIDTH // 2, HEIGHT // 2)

TRACKER_CFG = "bytetrack.yaml"   # or "botsort.yaml"
BALLOON_ID = 1                   

K_HISTORY = 5
V_MIN = 1.5
AREA_T_POS = 400.0
AREA_T_NEG = 400.0
CENTER_MARGIN_X = 0.15
CENTER_MARGIN_Y = 0.15
W_SIZE, W_CENTER, W_APPROACH, W_VEL = 0.45, 0.30, 0.20, 0.05
EXPECTED_MAX_AHAT = 0.08

FONT = cv.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICK = 1
LINE_SPACING = 20  # px between lines
PANEL_PAD_X = 10
PANEL_PAD_Y = 10


class TrackState:
    def __init__(self):
        self.centroids = deque(maxlen=K_HISTORY)
        self.areas = deque(maxlen=K_HISTORY)

track_db = {}  # track_id -> TrackState


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
    if top and left:  return "Top-Left"
    if top and not left: return "Top-Right"
    if not top and left: return "Bottom-Left"
    return "Bottom-Right"

def avg_velocity(centroids):
    if len(centroids) < 2: return 0.0, 0.0
    dx = sum(centroids[i+1][0] - centroids[i][0] for i in range(len(centroids)-1)) / (len(centroids)-1)
    dy = sum(centroids[i+1][1] - centroids[i][1] for i in range(len(centroids)-1)) / (len(centroids)-1)
    return dx, dy

def dir_label_from_v(vx, vy):
    vy = -vy  # invert for display intuition
    speed = math.hypot(vx, vy)
    if speed < V_MIN:
        return "Stationary"
    ang = math.degrees(math.atan2(vy, vx))
    if -22.5 <= ang < 22.5:    return "Moving Right"
    if 22.5 <= ang < 67.5:     return "Moving Up-Right"
    if 67.5 <= ang < 112.5:    return "Moving Up"
    if 112.5 <= ang < 157.5:   return "Moving Up-Left"
    if ang >= 157.5 or ang < -157.5: return "Moving Left"
    if -157.5 <= ang < -112.5: return "Moving Down-Left"
    if -112.5 <= ang < -67.5:  return "Moving Down"
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
    """Draw a translucent rectangle at (x,y) with size (w,h)."""
    overlay = img.copy()
    cv.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_hud_bottom_left(img, lines):
    """Draws a multi-line HUD block in the bottom-left."""
    # Measure panel width by the longest line
    max_text_w = 0
    text_h = 0
    for ln in lines:
        (tw, th), _ = cv.getTextSize(ln, FONT, FONT_SCALE, FONT_THICK)
        max_text_w = max(max_text_w, tw)
        text_h = max(text_h, th)
    panel_w = max_text_w + 2 * PANEL_PAD_X
    panel_h = PANEL_PAD_Y * 2 + len(lines) * LINE_SPACING

    # Bottom-left corner
    x = 10
    y = img.shape[0] - 10 - panel_h  # top-left y of the panel

    # Panel
    draw_translucent_panel(img, x, y, panel_w, panel_h, color=(0,0,0), alpha=0.45)

    # Text lines
    baseline_y = y + PANEL_PAD_Y + text_h
    for ln in lines:
        cv.putText(img, ln, (x + PANEL_PAD_X, baseline_y), FONT, FONT_SCALE, (255,255,255), FONT_THICK, cv.LINE_AA)
        baseline_y += LINE_SPACING



def detect():
    cv.namedWindow("YOLOv11 Detection", cv.WINDOW_NORMAL)
    cv.resizeWindow("YOLOv11 Detection", WIDTH, HEIGHT)

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv.CAP_PROP_FPS, 30)

    model = YOLO("lastv6.pt")
    device = 0 if torch.cuda.is_available() else None

    # FPS smoothing (EMA)
    fps = 0.0
    alpha_fps = 0.1
    last_t = time.perf_counter()

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=0.5,
            iou=0.45,
            imgsz=960,
            verbose=False,
            device=device,
            classes=[BALLOON_ID],
        )

        r0 = results[0]
        annotated = r0.plot()
        cv.circle(annotated, CENTER, 3, (0, 255, 0), -1, cv.LINE_AA)

        boxes = r0.boxes
        outputs = []
        top_threat = None

        if boxes is not None and len(boxes) > 0:
            xywh = boxes.xywh.detach().cpu().tolist()
            conf = boxes.conf.detach().cpu().tolist()
            ids  = boxes.id.detach().cpu().tolist() if boxes.id is not None else [None]*len(xywh)

            for (x, y, w, h), s, tid in zip(xywh, conf, ids):
                cx, cy = int(x), int(y)
                A = w * h
                Ahat = A / float(WIDTH * HEIGHT)

                if tid is None:
                    continue
                tid = int(tid)
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

                # Draw centroid + quick label
                cv.circle(annotated, (cx, cy), 3, (0, 255, 255), -1, cv.LINE_AA)
                cv.putText(annotated, f"ID {tid} | {size_idx}", (cx + 8, cy - 8),
                           FONT, 0.6, (50, 255, 255), 2, cv.LINE_AA)

            if outputs:
                top_threat = max(outputs, key=lambda o: o["threat_score"])
                cx, cy = top_threat["centroid"]
                ver = "top" if cy <= CENTER[1] else "bottom"
                hor = "left" if cx <= CENTER[0] else "right"
                cv.putText(annotated, f"Center is: {ver} {hor}",
                           (10, 30), FONT, 1, (0, 255, 0), 2, cv.LINE_AA)


        # Update FPS (EMA)
        t1 = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, (t1 - last_t))
        last_t = t1
        fps = (1 - alpha_fps) * fps + alpha_fps * inst_fps if fps > 0 else inst_fps

        hud_lines = [f"FPS: {fps:5.1f}"]
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

        # Display
        cv.imshow("YOLOv11 Detection", annotated)
        if (cv.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    detect()
