import cv2
import numpy as np

def detect_shapes_in_blue_and_green(frame):
    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Adjusted HSV range for green and blue
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([85, 255, 255])
    lower_blue = np.array([90, 75, 35])  # You may need to adjust these values
    upper_blue = np.array([140, 255, 255])

    # Create masks for green and blue colors
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Detect and draw green shapes
    green_result = cv2.bitwise_and(frame, frame, mask=green_mask)
    detect_and_draw_contours(green_result, frame, (0, 255, 0))

    # Detect and draw blue shapes
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)
    detect_and_draw_contours(blue_result, frame, (255, 0, 0))

    return frame

def detect_and_draw_contours(color_result, original_frame, contour_color):
    gray = cv2.cvtColor(color_result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(original_frame, [approx], 0, contour_color, 3)

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = detect_shapes_in_blue_and_green(frame)
    cv2.imshow('Blue and Green Shapes Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()