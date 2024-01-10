import cv2
import numpy as np

def preprocess_mask(mask):
    # Apply morphological operations to reduce noise in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def detect_circles(frame, color_name, min_radius=40, max_radius=60):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds for the blue color in HSV
    if color_name == "blue":
        lower_color = np.array([90, 75, 35])
        upper_color = np.array([140, 255, 255])
    else:
        raise ValueError("Color not supported for circles")
    
    # Create a mask to isolate the color of interest
    color_mask = cv2.inRange(hsv, lower_color, upper_color)
    color_mask = preprocess_mask(color_mask)

    # Apply the mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=color_mask)
    
    # Convert the masked frame to grayscale
    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)  # Apply a median blur to reduce noise

    # Use the Hough Transform to detect circles in the grayscale image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=40,
                               param1=100, param2=30, minRadius=min_radius, maxRadius=max_radius)

    # If circles are detected
    if circles is not None:
        # Convert the circle parameters to integers
        circles = np.uint16(np.around(circles))
        
        # Iterate through each circle and draw it on the frame
        for i in circles[0, :]:
            # Draw the outer circle in blue
            cv2.circle(frame, (i[0], i[1]), i[2], (255, 0, 0), 2)
            # Draw the center of the circle in red
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Return the frame with the detected circles drawn on it and the circles themselves
    return frame, circles

# Set up video capture from the default camera
cap = cv2.VideoCapture(0)

# Loop to continuously get frames from the camera
while True:
    # Read a frame from the capture
    ret, frame = cap.read()
    
    # If the frame was successfully retrieved
    if not ret:
        break

    # Detect circles in the frame
    frame_with_circles, circles = detect_circles(frame, "blue")
    
    # If circles are detected, print the count for debugging
    if circles is not None:
        print(f"Detected {len(circles[0])} circles")

    # Display the frame with the detected circles
    cv2.imshow('Circle Detection', frame_with_circles)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()