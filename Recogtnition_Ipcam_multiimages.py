import csv
import threading
import queue
import pickle
import cv2
import face_recognition
import cvzone
import numpy as np
import time

filename = 'Data/attendance.csv'

# Create a buffer to hold frames
buffer_size = 10
frame_buffer = queue.Queue(maxsize=buffer_size)

# # Load known encodings
# with open('EncodeFile.p', 'rb') as f:
#     encodeListKnownWithIds = pickle.load(f)
#
# encodeListKnown, studentIds = encodeListKnownWithIds
#

# Load known encodings
with open('known_faces1.pkl', 'rb') as f:
    known_faces = pickle.load(f)

# Extract the encodings and IDs from the dictionary
encodeListKnown = []
studentIds = []
for person_id, face_encodings in known_faces.items():
    for encoding in face_encodings:
        encodeListKnown.append(encoding)
        studentIds.append(person_id)

# Initialize FPS timer
start_time = time.time()
frame_count = 0



def fetch_info(id):
    # simulate fetching data that takes some time
    global studentInfo, imgStudent

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # get the header row
        rows = list(reader)  # get the remaining rows
        print(id)
        for row in rows:
            if row[0] == id:
                # increase attendance by 1
                attendance = int(row[4]) + 1
                row[4] = str(attendance)
                # update the row in the file
                with open(filename, 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    writer.writerows(rows)
                return {
                    'id': row[0],
                    'name': row[1],
                    'department': row[2],
                    'starting_year': row[3],
                    'total_attendance': str(attendance),
                    'date': row[5],
                    'time': row[6]
                }
        return None

def show_info(info,faceDis):
    global studentInfo, imgStudent

    student_info = info

    if student_info:
        print(f"Student ID: {student_info['id']}")
        print(f"Name: {student_info['name']}")
        print(f"Department: {student_info['department']}")
        print(f"Starting Year: {student_info['starting_year']}")
        print(f"Total Attendance: {student_info['total_attendance']}")
        print(f"Date: {student_info['date']}")
        print(f"Time: {student_info['time']}")
        print(faceDis)
    else:
        print(f"No student found with that ID")

def update_time(id):
    # Get current date and time
    current_date = time.strftime('%Y-%m-%d')
    current_time = time.strftime('%H:%M:%S')

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # get the header row
        rows = list(reader)  # get the remaining rows
        #print(id)
        for row in rows:
            #print(row[0])
            if row[0] == id:
                row[7] = current_date
                row[8] = current_time
                # update the row in the file
                with open(filename, 'w') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(header)
                    writer.writerows(rows)


# Function to read frames from the camera and put them in the buffer
def read_frames():
    cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.0.150:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif")

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

    recognized_faces = set()
    entry_time = set()
    while True:
        if not frame_buffer.empty():
            # Get frame from buffer
            frame = frame_buffer.get()

            # Resize and convert to RGB
            #imgS = frame
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
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                faceEncodings = face_recognition.face_encodings(imgS, [faceLoc])
                for faceEncoding in faceEncodings:
                    matches = face_recognition.compare_faces(encodeListKnown, faceEncoding, tolerance=0.6)
                    faceDis = face_recognition.face_distance(encodeListKnown, faceEncoding)

                    if True in matches:
                        matchIndex = np.argmin(faceDis)
                        id = studentIds[matchIndex]
                        #frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        update_time(id)
                        if cv2.waitKey(1) == ord('r'):
                            recognized_faces.clear()
                        if id not in recognized_faces:
                            recognized_faces.add(id)
                            #entry_time =
                            show_info(fetch_info(id), faceDis)

                    else:
                        pass
                        #frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

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
