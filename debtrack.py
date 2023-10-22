import cv2
import numpy as np

# Initialize the video capture
cap = cv2.VideoCapture(0)  # Replace 0 with the video file path if you have recorded video

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

# Variables for tracking
previous_center = None
trajectory = []

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = fgbg.apply(frame)

    # Perform morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours of moving objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Reset previous center
    previous_center = None

    for contour in contours:
        # Ignore small contours
        if cv2.contourArea(contour) < 500:
            continue

        # Find the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        center = (x + w // 2, y + h // 2)

        # Draw a rectangle around the moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw a circle at the center of the moving object
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        # Store the current center for tracking
        if previous_center is not None:
            trajectory.append((previous_center, center))
        previous_center = center

    # Show the resulting frame
    cv2.imshow('Frame', frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
