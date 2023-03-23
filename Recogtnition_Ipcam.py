import threading
import queue
import pickle
import cv2
import face_recognition
import cvzone
import numpy as np
import time

# Create a buffer to hold frames
buffer_size = 10
frame_buffer = queue.Queue(maxsize=buffer_size)

# Load known encodings
with open('EncodeFile.p', 'rb') as f:
    encodeListKnownWithIds = pickle.load(f)

encodeListKnown, studentIds = encodeListKnownWithIds

# Initialize FPS timer
start_time = time.time()
frame_count = 0


# Function to read frames from the camera and put them in the buffer
def read_frames():
    cap = cv2.VideoCapture(
        "rtsp://admin:admin123@192.168.0.150:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")

    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame, (0, 0), None, 0.5, 0.5)
        if not success:
            break
        # Put frame in the buffer
        if not frame_buffer.full():
            frame_buffer.put(frame)
        else:
            frame_buffer.get()
            frame_buffer.put(frame)

    cap.release()


# Function to process frames from the buffer
def process_frames():
    global start_time, frame_count
    while True:
        if not frame_buffer.empty():
            # Get frame from buffer
            frame = frame_buffer.get()

            # Resize and convert to RGB
            imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            # Detect faces and encode
            #faceCurFrame = face_recognition.face_locations(imgS, model="cnn")

            faceCurFrame = face_recognition.face_locations(imgS)
            encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

            # Check for matches and draw boxes
            for faceLoc in faceCurFrame:
                y1, x2, y2, x1 = [i * 4 for i in faceLoc]
                bbox = x1, y1, x2 - x1, y2 - y1
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                faceEncodings = face_recognition.face_encodings(imgS, [faceLoc])
                for faceEncoding in faceEncodings:
                    matches = face_recognition.compare_faces(encodeListKnown, faceEncoding, tolerance=0.5)
                    faceDis = face_recognition.face_distance(encodeListKnown, faceEncoding)

                    if True in matches:
                        matchIndex = np.argmin(faceDis)
                        id = studentIds[matchIndex]
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            # Display frame
            cv2.imshow("Face Attendance", frame)

            # Update FPS counter
            frame_count += 1
            if time.time() - start_time >= 1:
                fps = frame_count / (time.time() - start_time)
                print("FPS:", round(fps, 2))
                frame_count = 0
                start_time = time.time()

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()


# Create and start threads
read_thread = threading.Thread(target=read_frames)
process_thread = threading.Thread(target=process_frames)

read_thread.start()
process_thread.start()

# Wait for threads to finish
read_thread.join()
process_thread.join()
