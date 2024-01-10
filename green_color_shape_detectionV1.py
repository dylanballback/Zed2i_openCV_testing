import cv2
import numpy as np

def detect_green_shapes(frame):
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for green
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    result = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Convert to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find edges using Canny
    edges = cv2.Canny(blurred, 30, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Approximate the contour
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Draw contours on the original frame
        cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

    return frame

# Start capturing from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect green shapes in the frame
    result_frame = detect_green_shapes(frame)

    # Display the frame
    cv2.imshow('Green Shapes Detection', result_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()