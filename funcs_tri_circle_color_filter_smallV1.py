import cv2
import numpy as np

def detect_triangles(frame, color_name, min_area=500):
    # Convert to HSV color space for better color segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for green color detection
    if color_name == "green":
        lower_color = np.array([35, 35, 35])
        upper_color = np.array([85, 255, 255])
    else:
        raise ValueError("Color not supported for triangles")

    # Create a mask to isolate the specified color
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply the mask to filter out everything but the specified color
    masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)

    # Convert masked frame to grayscale for contour detection
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    # Find contours from the grayscale image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour found
    for cnt in contours:
        # Approximate the contour to a polygon
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the polygon has 3 sides (triangle)
        if len(approx) == 3:
            # Calculate area of the triangle
            area = cv2.contourArea(cnt)
            # Draw the triangle only if its area is above the threshold
            if area > min_area:
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)

    return frame

def detect_circles(frame, color_name, min_radius=10):
    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV range for blue color detection
    if color_name == "blue":
        lower_color = np.array([90, 75, 35])
        upper_color = np.array([140, 255, 255])
    else:
        raise ValueError("Color not supported for circles")

    # Create a mask to isolate the specified color
    color_mask = cv2.inRange(hsv, lower_color, upper_color)

    # Apply the mask to filter out everything but the specified color
    masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)

    # Convert masked frame to grayscale for circle detection
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # Use Hough Transform to detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=10, maxRadius=100)

    # If circles are detected, draw them
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the circle only if its radius is above the threshold
            if i[2] > min_radius:
                cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 3)

    return frame

# Example usage
cap = cv2.VideoCapture(0)  # Start capturing video from the default camera
while True:
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break

    # Detect green triangles
    frame_with_triangles = detect_triangles(frame, "green")

    # Detect blue circles
    frame_with_circles_and_triangles = detect_circles(frame_with_triangles, "blue")

    # Display the frame with detected shapes
    cv2.imshow('Green Triangles and Blue Circles Detection', frame_with_circles_and_triangles)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the camera resource
cv2.destroyAllWindows()  # Close all OpenCV windows