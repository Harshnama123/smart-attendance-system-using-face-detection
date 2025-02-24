import cv2
import face_recognition
import pickle
import numpy as np
import csv
from datetime import datetime

# Load encodings
with open("encodings.pkl", "rb") as f:
    student_encodings, student_names = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create/Open attendance file
attendance_file = "attendance.csv"

# Check if file exists, if not, create it with headers
try:
    with open(attendance_file, "x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Time"])
except FileExistsError:
    pass  # File already exists, continue

# Store already marked students to avoid duplicate entries
marked_students = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        # Compare with known encodings
        matches = face_recognition.compare_faces(student_encodings, face_encoding, tolerance=0.5)
        face_distances = face_recognition.face_distance(student_encodings, face_encoding)

        best_match_index = np.argmin(face_distances) if matches else None

        if best_match_index is not None and matches[best_match_index]:
            name = student_names[best_match_index]

            # If not already marked, add to attendance
            if name not in marked_students:
                marked_students.add(name)
                with open(attendance_file, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                print(f"✅ Attendance marked for: {name}")

            # Display name on the frame
            y1, x2, y2, x1 = face_locations[0]  # Scale back to original size
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Face Recognition Attendance System", frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
