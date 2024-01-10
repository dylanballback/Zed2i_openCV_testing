import cv2
import numpy as np

def detect_specific_shape_and_color(frame, color_name, shape_name):
    # Convert frame to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for specific colors
    if color_name == "green":
        lower_color = np.array([35, 40, 40])
        upper_color = np.array([85, 255, 255])
    elif color_name == "blue":
        lower_color = np.array([100, 150, 0])
        upper_color = np.array([140, 255, 255])
    else:
        raise ValueError("Color not supported")
    
    # Create a mask for the color and apply it
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    #Convert to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    #Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #Find edges using Canny
    edges = cv2.Canny(blurred, 30, 200)

    #Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        #Approximate the contour
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        #Determine shape based on the number of vertices
        if shape_name == "triangle" and len(approx) == 3:
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
        elif shape_name == "circle" and len(approx) >= 5:
            area = cv2.contourArea(cnt)
            if area > 300:  # Filter out small circles
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
    return frame

#Start capturing from webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    #Detect specific shape and color in the frame
    #result_frame = detect_specific_shape_and_color(frame, 'green', 'triangle')
    result_frame = detect_specific_shape_and_color(frame, 'blue', 'circle')

    cv2.imshow('Frame', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()