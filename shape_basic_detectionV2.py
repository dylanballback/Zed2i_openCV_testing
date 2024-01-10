import cv2
import numpy as np

def detect_shapes_and_colors(frame):
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for green and blue
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Create masks for green and blue colors
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find and draw shapes for each color
    for color_mask, shape_name in zip([green_mask, blue_mask], ['triangle', 'circle']):
        # Apply mask
        result = cv2.bitwise_and(frame, frame, mask=color_mask)

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

            # Determine shape based on the number of vertices
            if shape_name == "triangle" and len(approx) == 3:
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
            elif shape_name == "circle" and len(approx) >= 5:
                area = cv2.contourArea(cnt)
                if area > 300:  # Filter out small circles
                    cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

    return frame

# Start capturing from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect shapes and colors in the frame
    result_frame = detect_shapes_and_colors(frame)

    cv2.imshow('Frame', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
