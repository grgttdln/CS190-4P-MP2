# ===============================================================
# VEHICLE SPEED ESTIMATION (YOLOv8 + Kalman Filter + Hungarian Matching)
# with Predefined Calibration
# ===============================================================

import cv2
import numpy as np
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import time

# =====================
# CONFIGURATION
# =====================
VIDEO_PATH = "cars.mp4"
IOU_THRESHOLD = 0.3
MAX_AGE = 10
paused = False
elapsed_time = 0.0  # stores total elapsed time excluding pauses
last_time = time.time()

cap = cv2.VideoCapture(VIDEO_PATH)

# =====================
# --- PREDEFINED CALIBRATION ---
# =====================
pixels_per_meter = 8.8
FPS = 18
print(f"Using predefined calibration: {pixels_per_meter:.2f} pixels/meter, {FPS} FPS")

# =====================
# COLOR BY SPEED
# =====================
def get_speed_color(speed_kmph):
    if speed_kmph <= 40:
        return (0, 255, 0)        # Green
    elif speed_kmph <= 80:
        return (0, 165, 255)      # Orange
    else:
        return (0, 0, 255)        # Red

# =====================
# NON-MAXIMUM SUPPRESSION
# =====================
def non_max_suppression_fast(boxes, overlap_thresh=0.4):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype=float)
    pick = []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        if last > 0:
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(
                idxs,
                np.concatenate(([last], np.where(overlap > overlap_thresh)[0]))
            )
        else:
            idxs = np.delete(idxs, last)

    return boxes[pick].astype("int")

# =====================
# TRACK CLASS
# =====================
class Track:
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])
        self.kf.R *= 10.
        self.kf.P[4:,4:] *= 1000.
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.x[:4] = np.array(bbox).reshape((4,1))

        self.last_center = None
        self.speed_kmh = 0.0
        self.no_losses = 0

    def predict(self):
        self.kf.predict()
        return self.kf.x[:4].reshape(-1)

    def update(self, bbox):
        self.kf.update(np.array(bbox))
        self.no_losses = 0

        x, y, w, h = self.kf.x[:4].reshape(-1)
        center = np.array([x, y])
        if self.last_center is not None:
            dist_px = np.linalg.norm(center - self.last_center)
            dist_m = dist_px / pixels_per_meter
            speed_mps = dist_m * FPS
            self.speed_kmh = speed_mps * 3.6  # Conversion: 1 m/s = 3.6 km/h
        self.last_center = center

# =====================
# IOU FUNCTION
# =====================
def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[0] + bb_test[2], bb_gt[0] + bb_gt[2])
    yy2 = min(bb_test[1] + bb_test[3], bb_gt[1] + bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    union = (bb_test[2]*bb_test[3]) + (bb_gt[2]*bb_gt[3]) - inter + 1e-6
    return inter / union

# =====================
# INITIALIZE YOLO MODEL
# =====================
model = YOLO("yolov8n.pt")
tracks = []

# =====================
# MAIN LOOP
# =====================
while True:
    current_time = time.time()
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time += current_time - last_time
        last_time = current_time

        # Display elapsed time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        cv2.putText(frame, f"Time: {time_str}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        # --- DETECTIONS ---
        results = model(frame, verbose=False)[0]
        detections = [box for box in results.boxes.xywh.cpu().numpy()]

        # --- REMOVE OVERLAPPING DETECTIONS ---
        detections = non_max_suppression_fast(detections, overlap_thresh=0.4)
        predictions = [t.predict() for t in tracks]

        # --- ASSIGN DETECTIONS TO TRACKS ---
        if len(predictions) > 0 and len(detections) > 0:
            cost_matrix = np.zeros((len(predictions), len(detections)))
            for i, pred in enumerate(predictions):
                for j, det in enumerate(detections):
                    cost_matrix[i,j] = 1 - iou(pred, det)
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = np.array([]), np.array([])

        assigned_tracks = set()
        assigned_detections = set()

        for i, j in zip(row_ind, col_ind):
            if 1 - cost_matrix[i,j] < IOU_THRESHOLD:
                continue
            tracks[i].update(detections[j])
            assigned_tracks.add(i)
            assigned_detections.add(j)

        # --- HANDLE UNASSIGNED TRACKS ---
        for i, t in enumerate(tracks):
            if i not in assigned_tracks:
                t.no_losses += 1

        # --- CREATE NEW TRACKS ---
        for j, det in enumerate(detections):
            if j not in assigned_detections:
                tracks.append(Track(det))

        # --- REMOVE OLD TRACKS ---
        tracks = [t for t in tracks if t.no_losses <= MAX_AGE]

        # --- DRAW TRACKS ---
        for t in tracks:
            x, y, w, h = t.kf.x[:4].reshape(-1)
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            color = get_speed_color(t.speed_kmh)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{t.speed_kmh:.1f} km/h", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # --- DISPLAY FRAME ---
    cv2.imshow("Vehicle Speed Estimation (YOLOv8)", frame)

    # --- CONTROLS ---
    key = cv2.waitKey(30) & 0xFF
    if key == 32:  # SPACE to pause/resume
        paused = not paused
        last_time = time.time()
    elif key == ord('q') or key == 27:  # quit
        break

cap.release()
cv2.destroyAllWindows()
