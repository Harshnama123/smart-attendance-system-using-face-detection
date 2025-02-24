import cv2
import face_recognition
import os
import numpy as np
import pickle

# Path to student images
STUDENTS_PATH = "imgaes"

# List to store encodings and names
student_encodings = []
student_names = []

for filename in os.listdir(STUDENTS_PATH):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(STUDENTS_PATH, filename)
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)

        if encoding:
            student_encodings.append(encoding[0])
            student_names.append(os.path.splitext(filename)[0])  # Remove file extension

# Save encodings
with open("encodings.pkl", "wb") as f:
    pickle.dump((student_encodings, student_names), f)

print("Student faces registered successfully!")
