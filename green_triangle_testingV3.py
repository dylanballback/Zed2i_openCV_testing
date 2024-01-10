import cv2
import numpy as np

def preprocess_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def detect_triangles(frame, color_name, min_area=500):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([35, 35, 35])  # Adjust as needed
    upper_color = np.array([100, 255, 255])  # Adjusted as per your input
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    color_mask = preprocess_mask(color_mask)

    mask_display = cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)
    masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_triangles = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            epsilon = 0.05 * cv2.arcLength(cnt, True)  # Adjust epsilon value as needed
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 3:
                detected_triangles.append(approx)

    # Only draw the final, approximated triangles
    for triangle in detected_triangles:
        cv2.drawContours(frame, [triangle], 0, (0, 255, 0), 2)  # Draw detected triangle

    return frame, mask_display, detected_triangles

# Example usage
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, mask_display, current_triangles = detect_triangles(frame, "green")
    print(f"Detected {len(current_triangles)} triangles")  # Debugging information

    cv2.imshow('Triangle Detection', frame)
    cv2.imshow('Color Mask', mask_display)  # Display the mask

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()