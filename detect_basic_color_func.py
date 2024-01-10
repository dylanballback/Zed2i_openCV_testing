import cv2
import numpy as np

def detect_color(frame, color_name):
    # Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for different colors
    if color_name == "red":
        lower = np.array([0, 120, 70])
        upper = np.array([10, 255, 255])
    elif color_name == "green":
        lower = np.array([35, 40, 40])
        upper = np.array([85, 255, 255])
        standard_lower = np.array([40, 40, 40])
        standard_upper = np.array([70, 255, 255])
    elif color_name == "blue":
        lower = np.array([100, 150, 0])
        upper = np.array([140, 255, 255])
    elif color_name == "yellow":
        lower = np.array([20, 100, 100])
        upper = np.array([30, 255, 255])
    else:
        raise ValueError("Color not supported")

    # Create a mask and apply it to the frame
    mask = cv2.inRange(hsv_frame, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    return result

# Start capturing from webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Call the color detection function
    # Replace 'red' with your desired color
    color_detected_frame = detect_color(frame, 'yellow')

    # Display the resulting frame
    cv2.imshow('Frame', color_detected_frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()