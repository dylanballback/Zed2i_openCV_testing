import cv2
import numpy as np

def detect_colors(frame):
    # Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for different colors
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # Create masks for each color
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    # Combine masks
    combined_mask = cv2.bitwise_or(yellow_mask, green_mask)
    combined_mask = cv2.bitwise_or(combined_mask, blue_mask)

    # Apply the combined mask to the frame
    result = cv2.bitwise_and(frame, frame, mask=combined_mask)
    return result

# Start capturing from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Call the color detection function for multiple colors
    color_detected_frame = detect_colors(frame)

    # Display the resulting frame
    cv2.imshow('Frame', color_detected_frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()