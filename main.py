from ultralytics import YOLO
import cv2
import numpy as np
import math

model = YOLO("yolov8n.pt")

video_path = "videos/challenge_color_848x480.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Eroare la deschiderea videoclipului.")
    exit()

previous_sizes = {}
previous_distances = {}

FOCAL_LENGTH = 650
AVERAGE_PERSON_HEIGHT = 1.7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, verbose=False)[0]
    ids = results.boxes.id.int().cpu().tolist() if results.boxes.id is not None else []

    for i, box in enumerate(results.boxes):
        cls = int(box.cls[0])
        label = model.names[cls]
        if label != "person":
            continue

        obj_id = ids[i] if i < len(ids) else -1
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        size = w * h

        if h > 0:
            distance = (FOCAL_LENGTH * AVERAGE_PERSON_HEIGHT) / h
        else:
            distance = 0

        distance = np.clip(distance, 0.5, 15)

        if obj_id in previous_sizes:
            delta = size - previous_sizes[obj_id]
            if delta > 2000:
                status = "Apropiere"
                color = (0, 0, 255)
            elif delta < -2000:
                status = "Depărtare"
                color = (0, 255, 0)
            else:
                status = "Stabil"
                color = (255, 255, 0)
        else:
            status = "Detectat"
            color = (255, 255, 255)

        previous_sizes[obj_id] = size
        previous_distances[obj_id] = distance

        text = f"ID {obj_id} | {status} | {distance:.2f} m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        bar_x = x1
        bar_y = y2 + 10
        bar_w = int(200 * (1 - min(distance / 10, 1)))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), color, -1)
        cv2.putText(frame, f"{distance:.1f} m", (bar_x, bar_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Approach / Move Away Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

