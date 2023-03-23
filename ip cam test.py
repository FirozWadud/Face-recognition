import cv2

# RTSP URL for Dahua camera
rtsp_url = "rtsp://admin:admin123@192.168.0.150:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera stream is open
if not cap.isOpened():
    print("Error opening stream")
    exit()

# Loop through the frames in the video stream
while True:
    # Read a frame from the stream
    ret, frame = cap.read()

    # Check if a frame was successfully read
    if not ret:
        break

    # Display the frame
    cv2.imshow("Dahua Camera Feed", frame)

    # Check for user input to quit the program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
