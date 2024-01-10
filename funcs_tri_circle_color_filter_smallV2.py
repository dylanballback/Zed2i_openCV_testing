import cv2
import numpy as np

def preprocess_mask(mask):
    """
    Apply morphological operations to reduce noise and smooth the mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def detect_triangles(frame, color_name, min_area=500):
    """
    Detect triangles of a specific color and minimum area in the frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color_name == "green":
        lower_color = np.array([35, 35, 35])
        upper_color = np.array([85, 255, 255])
    else:
        raise ValueError("Color not supported for triangles")
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    color_mask = preprocess_mask(color_mask)
    masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_triangles = []
    for cnt in contours:
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 3:
            area = cv2.contourArea(cnt)
            if area > min_area:
                detected_triangles.append(approx)
    return detected_triangles

def detect_circles(frame, color_name, min_radius=10):
    """
    Detect circles of a specific color and minimum radius in the frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if color_name == "blue":
        lower_color = np.array([90, 75, 35])
        upper_color = np.array([140, 255, 255])
    else:
        raise ValueError("Color not supported for circles")
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    color_mask = preprocess_mask(color_mask)
    masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=10, maxRadius=100)
    detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if i[2] > min_radius:
                detected_circles.append(i)
    return detected_circles

# Example usage
cap = cv2.VideoCapture(0)
previous_triangles = []
previous_circles = []
smoothing_frames = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_triangles = detect_triangles(frame, "green")
    current_circles = detect_circles(frame, "blue")

    # Debugging information
    print(f"Detected {len(current_triangles)} triangles, {len(current_circles)} circles")

    # Temporal Smoothing (simple version)
    if len(previous_triangles) >= smoothing_frames:
        previous_triangles.pop(0)
    if len(previous_circles) >= smoothing_frames:
        previous_circles.pop(0)

    previous_triangles.append(current_triangles)
    previous_circles.append(current_circles)

    # Draw detected shapes
    for triangle_set in previous_triangles:
        for triangle in triangle_set:
            cv2.drawContours(frame, [triangle], 0, (0, 255, 0), 3)
    
    for circle_set in previous_circles:
        for circle in circle_set:
            cv2.circle(frame, (circle[0], circle[1]), circle[2], (255, 0, 0), 3)

    cv2.imshow('Shape Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()