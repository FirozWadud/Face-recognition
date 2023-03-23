import threading
import os
import pickle
import cv2
import face_recognition
import cvzone
import numpy as np
import csv
from datetime import datetime


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")

id = -1
imgStudent = []
studentInfo = []
filename = 'Data/attendance.csv'

def fetch_info():
    # simulate fetching data that takes some time
    global studentInfo, imgStudent

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader) # get the header row
        rows = list(reader) # get the remaining rows
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

def show_info(info):
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
    else:
        print(f"No student found with that ID")

def adjust_brightness(value):
    global brightness
    brightness = value / 100
    print(f"Brightness set to {brightness}")

def adjust_contrast(value):
    global contrast
    contrast = value / 100
    print(f"Contrast set to {contrast}")

def main():
    global id, imgBackground, studentInfo, imgStudent, brightness, contrast
    brightness = 1.0
    contrast = 1.0

    cv2.namedWindow("Face Attendance")
    cv2.createTrackbar("Brightness", "Face Attendance", 100, 200, adjust_brightness)
    cv2.createTrackbar("Contrast", "Face Attendance", 100, 200, adjust_contrast)

    fetching_thread = None
    showing_thread = None

    while True:
        success, img = cap.read()

        # Adjust brightness and contrast
        img = cv2.convertScaleAbs(img, alpha=brightness, beta=(1.0 - contrast) * 255)

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace, tolerance=0.4)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis)



                print(np.min(faceDis))

                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = x1, y1, x2 - x1, y2 - y1
                    img = cvzone.cornerRect(img, bbox, rt=0)

                    id = studentIds[matchIndex]

                    if fetching_thread is None:
                        fetching_thread = threading.Thread(target=lambda: show_info(fetch_info()))
                        fetching_thread.start()

                    if fetching_thread is not None and not fetching_thread.is_alive():
                        fetching_thread.join()
                        fetching_thread = None

        cv2.imshow("Face Attendance", img)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
