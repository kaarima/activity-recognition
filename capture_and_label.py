import cv2
import numpy as np
import os

# Create directory to save labeled data
if not os.path.exists('labeled_data'):
    os.makedirs('labeled_data')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)
prev_frame = None
activity = input("Enter the activity you are recording (e.g., walking): ").strip().lower()
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_frame is None:
        prev_frame = gray
        continue

    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the frame with activity label
            frame_filename = f'labeled_data/{activity}_frame_{frame_count}.jpg'
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

    cv2.imshow('Activity Recording', frame)
    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
