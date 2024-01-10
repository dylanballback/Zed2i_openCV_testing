import cv2 
import numpy as np

#Get video stream from USB Camera
cap = cv2.VideoCapture(0)

#Example for color red
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

while True:
    #Capture frame by frame 
    ret, frame = cap.read()
    if not ret:
        break

    #Convert frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Create  mask for color detection 
    color_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    color_result = cv2.bitwise_and(frame, frame, mask=color_mask)


    #Display resulting frame
    cv2.imshow('Frame', color_result)

    #Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()