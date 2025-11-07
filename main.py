import cv2
import numpy as np
from collections import deque
import time

# --- CONFIGURATION --- 
VIDEO_PATH = "cars.mp4"
CASCADE_PATH = "cars_haar.xml" 

MIN_CAR_SIZE = (20, 20)
SMOOTHING_WINDOW = 5
DETECTION_INTERVAL = 15
MIN_VISIBLE_SIZE = 20
MIN_SPEED_KMPH = 2.0
SIZE_SMOOTHING_WINDOW = 10
CENTER_SMOOTHING_WINDOW = 5
DISTANCE_THRESHOLD = 50
DETECTION_SCALE = 0.9
BOX_PAD = 0

# --- PREDEFINED CALIBRATION ---
pixels_per_meter = 8.8
FPS = 18
print(f"Using predefined calibration: {pixels_per_meter:.2f} pixels/meter, {FPS} FPS")

# --- LOAD VIDEO ---
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

# --- LOAD HAAR CASCADE ---
car_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if car_cascade.empty():
    raise RuntimeError("Failed to load Haar cascade XML")

# --- TRACKING STRUCTURES ---
trackers = {}
positions = {}
speed_history = {}
size_history = {}
center_history = {}
car_counter = 0
frame_count = 0
paused = False
start_time = time.time()

# --- HELPER FUNCTIONS ---
def get_speed_color(speed_kmph):
    if speed_kmph <= 40:
        return (0, 255, 0)
    elif speed_kmph <= 80:
        return (0, 165, 255)
    else:
        return (0, 0, 255)

def create_tracker():
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    elif hasattr(cv2, "legacy"):
        return cv2.legacy.TrackerCSRT_create()
    else:
        raise RuntimeError("CSRT tracker not available in your OpenCV build.")

# --- MAIN LOOP ---
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        remove_ids = []
        tracked_boxes = {}

        # --- UPDATE EXISTING TRACKERS ---
        for car_id, tracker in list(trackers.items()):
            success, box = tracker.update(frame)
            if not success:
                remove_ids.append(car_id)
                continue

            x, y, w, h = map(int, box)

            # Limit expansion to prevent oversized boxes
            if car_id in size_history and len(size_history[car_id]) > 0:
                avg_w = int(np.mean([s[0] for s in size_history[car_id]]))
                avg_h = int(np.mean([s[1] for s in size_history[car_id]]))
                if w > 1.5 * avg_w or h > 1.5 * avg_h:
                    w = int(1.2 * avg_w)
                    h = int(1.2 * avg_h)

            if w < MIN_VISIBLE_SIZE or h < MIN_VISIBLE_SIZE:
                remove_ids.append(car_id)
                continue

            # Smooth size and center
            size_history[car_id].append((w, h))
            avg_w = int(np.mean([s[0] for s in size_history[car_id]]))
            avg_h = int(np.mean([s[1] for s in size_history[car_id]]))

            cx, cy = int(x + w/2), int(y + h/2)
            center_history[car_id].append((cx, cy))
            smoothed_center = np.mean(center_history[car_id], axis=0).astype(int)
            cx, cy = smoothed_center
            x = int(cx - avg_w / 2)
            y = int(cy - avg_h / 2)
            w, h = avg_w, avg_h

            tracked_boxes[car_id] = (x, y, w, h)

            # Compute speed
            smoothed_speed = 0
            if car_id in positions:
                dx = cx - positions[car_id][0]
                dy = cy - positions[car_id][1]
                pixel_distance = np.sqrt(dx**2 + dy**2)
                distance_m = pixel_distance / pixels_per_meter
                speed_mps = distance_m * FPS
                speed_kmph = speed_mps * 3.6
                speed_history[car_id].append(speed_kmph)
                smoothed_speed = np.mean(speed_history[car_id])
            else:
                speed_history[car_id].append(0)

            if smoothed_speed < MIN_SPEED_KMPH:
                remove_ids.append(car_id)
                continue

            color = get_speed_color(smoothed_speed)
            label = f"{smoothed_speed:.1f} km/h"
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            positions[car_id] = (cx, cy)

        # --- REMOVE OLD TRACKERS ---
        for car_id in remove_ids:
            for d in [trackers, positions, speed_history, size_history, center_history]:
                d.pop(car_id, None)

        # --- PERIODIC DETECTION (HAAR CASCADE + NMS) ---
        if frame_count % DETECTION_INTERVAL == 0:
            small_gray = cv2.resize(gray, (0,0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
            detections = car_cascade.detectMultiScale(
                small_gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(int(MIN_CAR_SIZE[0]*DETECTION_SCALE), int(MIN_CAR_SIZE[1]*DETECTION_SCALE))
            )

            detections = [
                (int(x/DETECTION_SCALE), int(y/DETECTION_SCALE),
                 int(w/DETECTION_SCALE), int(h/DETECTION_SCALE))
                for (x, y, w, h) in detections
            ]

            boxes, confidences = [], []
            for (x, y, w, h) in detections:
                aspect = w / max(h, 1)
                if aspect > 3.5 or aspect < 0.8:
                    continue

                shrink_factor_w = 0.85 if aspect > 1.5 else 0.9
                shrink_factor_h = 0.9 if aspect > 1.5 else 0.95
                new_w = int(w * shrink_factor_w)
                new_h = int(h * shrink_factor_h)
                x += (w - new_w) // 2
                y += (h - new_h) // 2
                w, h = new_w, new_h

                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                if w > frame.shape[1] * 0.25 or h > frame.shape[0] * 0.25:
                    continue
                if w * h < 100:
                    continue

                confidence = float(w * h / (1 + abs(aspect - 1.6)))
                boxes.append([x, y, x+w, y+h])
                confidences.append(confidence)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.0, nms_threshold=0.3)
            nms_detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x1, y1, x2, y2 = boxes[i]
                    nms_detections.append([x1, y1, x2-x1, y2-y1])

            # --- ADD NEW TRACKERS ---
            for (x, y, w, h) in nms_detections:
                cx, cy = int(x+w/2), int(y+h/2)
                duplicate = False
                for tbox in tracked_boxes.values():
                    tx, ty, tw, th = tbox
                    tcx, tcy = tx+tw//2, ty+th//2
                    if np.linalg.norm([cx-tcx, cy-tcy]) < DISTANCE_THRESHOLD:
                        duplicate = True
                        break
                if duplicate:
                    continue

                car_counter += 1
                tracker = create_tracker()
                tracker.init(frame, (x, y, w, h))
                trackers[car_counter] = tracker
                positions[car_counter] = (cx, cy)
                speed_history[car_counter] = deque(maxlen=SMOOTHING_WINDOW)
                size_history[car_counter] = deque(maxlen=SIZE_SMOOTHING_WINDOW)
                center_history[car_counter] = deque(maxlen=CENTER_SMOOTHING_WINDOW)
                size_history[car_counter].append((w, h))
                center_history[car_counter].append((cx, cy))

        # --- DISPLAY TIMESTAMP ---
        elapsed_sec = frame_count / FPS
        minutes = int(elapsed_sec // 60)
        seconds = int(elapsed_sec % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        cv2.putText(frame, f"Time: {timestamp}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- SHOW FRAME ---
        cv2.imshow("Vehicle Speed Tracker", frame)

    # --- CONTROL KEYS ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == 32:  # Spacebar to pause/resume
        paused = not paused
        print("⏸️ Paused" if paused else "▶️ Resumed")

cap.release()
cv2.destroyAllWindows()
