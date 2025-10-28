import cv2
import numpy as np

video_color = cv2.VideoCapture("videos/challenge_color_848x480.mp4")
video_depth = cv2.VideoCapture("videos/challenge_depth_848x480.mp4")

while True:
    ret_c, frame_c = video_color.read()
    ret_d, frame_d = video_depth.read()
    if not ret_c or not ret_d:

        break

    depth_gray = cv2.cvtColor(frame_d, cv2.COLOR_BGR2GRAY)
    depth_norm = cv2.normalize(depth_gray, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = np.uint8(depth_norm)


    _, mask_close = cv2.threshold(depth_norm, 60, 255, cv2.THRESH_BINARY_INV)


    contours, _ = cv2.findContours(mask_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 500:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame_c, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame_c, "Obstacol", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Color", frame_c)
    cv2.imshow("Adancime", depth_norm)
    cv2.imshow("Obstacole", mask_close)

    if cv2.waitKey(30) & 0xFF == 27:
        break

video_color.release()
video_depth.release()
cv2.destroyAllWindows()
