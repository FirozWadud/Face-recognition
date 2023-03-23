import os
import cv2
import face_recognition
import pickle

known_faces = {}

# Traverse through each directory named after unique IDs within "Images" directory
for person_dir in os.listdir("Images_multi1"):
    person_id = person_dir  # Use the directory name as the person ID
    person_encodings = []
    for img_file in os.listdir(os.path.join("Images_multi1", person_dir)):
        img_path = os.path.join("Images_multi1", person_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            person_encodings.append(face_encodings[0])
    if len(person_encodings) > 0:
        known_faces[person_id] = person_encodings

# Save the encodings as a pickle file
with open("known_faces1.pkl", "wb") as f:
    pickle.dump(known_faces, f)
