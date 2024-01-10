import cv2

#Get video stream from USB Camera
cap = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    #If frame is read correctly, ret is true
    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Raw Video Feed', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()