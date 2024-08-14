import cv2
import numpy as np

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Initialize variables
prev_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Initialize the previous frame
    if prev_frame is None:
        prev_frame = gray
        continue

    # Compute the absolute difference between the current and previous frame
    frame_delta = cv2.absdiff(prev_frame, gray)

    # Apply a threshold to the delta image
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the threshold image
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected movements
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small contours
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Motion Detection', frame)

    # Update the previous frame
    prev_frame = gray

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
