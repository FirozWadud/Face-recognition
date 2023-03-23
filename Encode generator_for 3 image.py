import cv2
import face_recognition
import pickle
import os

# Set the path of the folder containing images
folderPath = 'Images'

# Get a list of all image filenames in the folder
pathList = os.listdir(folderPath)

# Create empty lists for the images and their respective IDs
imgList = []
studentIds = []

# Loop through the list of filenames
for path in pathList:
    # Check if the filename contains the underscore character
    if '_' in path:
        # Split the filename by the underscore character to get the ID and image letter
        id, letter = path.split('_')
        # Check if the ID is already in the list of student IDs
        if id not in studentIds:
            # If the ID is new, create a new list for the images of that student
            studentImgs = []
            # Add the first image to the list of student images
            studentImgs.append(cv2.imread(os.path.join(folderPath, path)))
            # Add the student ID to the list of student IDs
            studentIds.append(id)
        else:
            # If the ID already exists, add the image to the list of student images
            studentImgs.append(cv2.imread(os.path.join(folderPath, path)))
        # Print the ID and image letter for debugging
        print(f"Processing image {letter} of student {id}")
    else:
        # If the filename does not contain the underscore character, skip it
        print(f"Skipping file {path}")
        continue

# Create an empty list to hold the encodings for all images
encodeListKnown = []

# Loop through the list of student images
for studentImgList in studentImgs:
    # Create an empty list to hold the encodings for each image of the student
    encodeList = []
    # Loop through the images of the student
    for img in studentImgList:
        # Convert the image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Compute the face encoding for the image
        encode = face_recognition.face_encodings(img)[0]
        # Add the face encoding to the list of encodings for this student
        encodeList.append(encode)
    # Add the list of encodings for this student to the list of all encodings
    encodeListKnown.append(encodeList)

# Create a list of tuples, where each tuple contains the list of encodings for a student and their respective ID
encodeListKnownWithIds = list(zip(encodeListKnown, studentIds))

# Save the encoding data to a file using pickle
with open("EncodeFile.p", "wb") as f:
    pickle.dump(encodeListKnownWithIds, f)

print("Encoding complete.")
