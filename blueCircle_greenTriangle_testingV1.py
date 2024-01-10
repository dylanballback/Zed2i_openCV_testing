import cv2
import numpy as np

def preprocess_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    #mask = cv2.GaussianBlur(mask, (9, 9), 0) #Gaussian blur to reduce noise
    return mask


def detect_circles(frame, mask, min_radius=10, max_radius=100):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=50, minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Check if the detected circle is within the color mask
            if mask[i[1], i[0]] != 0:  # Mask value is not zero at the circle's center
                cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 2)
                cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
    return frame, circles

def detect_shapes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 35, 35])
    upper_green = np.array([100, 255, 255])
    lower_blue = np.array([90, 75, 35])
    upper_blue = np.array([140, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = preprocess_mask(mask_green)
    mask_blue = preprocess_mask(mask_blue)

    # Detect green triangles
    masked_frame_green = cv2.bitwise_and(frame, frame, mask=mask_green)
    gray_green = cv2.cvtColor(masked_frame_green, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            if len(approx) == 3:
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)

    # Detect blue circles
    masked_frame_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
    gray_blue = cv2.cvtColor(masked_frame_blue, cv2.COLOR_BGR2GRAY)
    gray_blue = cv2.medianBlur(gray_blue, 5)
    circles = cv2.HoughCircles(gray_blue, cv2.HOUGH_GRADIENT, dp=1.3, minDist=40,
                               param1=100, param2=35, minRadius=40, maxRadius=60)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 2)
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    return frame

# Set up video capture from the default camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the capture
    ret, frame = cap.read()
    if not ret:
        break

    # Detect shapes in the frame
    frame_with_shapes = detect_shapes(frame)

    # Display the frame with the detected shapes
    cv2.imshow('Shape Detection', frame_with_shapes)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
