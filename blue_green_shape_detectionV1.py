import cv2
import numpy as np

def detect_shapes_for_colors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Adjusted HSV range for green and blue
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    


    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    def find_and_draw_shapes(mask, contour_color, shape):
        result = cv2.bitwise_and(frame, frame, mask=mask)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            if shape == "triangle" and len(approx) == 3:
                cv2.drawContours(frame, [approx], 0, contour_color, 3)
            elif shape == "circle":
                area = cv2.contourArea(cnt)
                if area > 500:  # Filter out small circles
                    if len(approx) > 5:  # More vertices likely means a smoother shape
                        cv2.drawContours(frame, [approx], 0, contour_color, 3)

    find_and_draw_shapes(green_mask, (0, 255, 0), "triangle")
    find_and_draw_shapes(blue_mask, (255, 0, 0), "circle")

    return frame

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result_frame = detect_shapes_for_colors(frame)
    cv2.imshow('Color Shapes Detection', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()