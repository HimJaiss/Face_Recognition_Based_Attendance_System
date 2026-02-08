import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces
ankit_image = face_recognition.load_image_file("C:/Users//HIMANSHU JAISWAL//Desktop//Minor Project//face recognition//ankit.jpeg")
ankit_encoding = face_recognition.face_encodings(ankit_image)[0]
pragya_image = face_recognition.load_image_file("C:/Users//HIMANSHU JAISWAL//Desktop//Minor Project//face recognition//pragya.jpeg")
pragya_encoding = face_recognition.face_encodings(pragya_image)[0]
nisha_image = face_recognition.load_image_file("C:/Users//HIMANSHU JAISWAL//Desktop//Minor Project//face recognition//nisha.jpeg")
nisha_encoding = face_recognition.face_encodings(nisha_image)[0]
himanshu_image = face_recognition.load_image_file("C:/Users//HIMANSHU JAISWAL//Desktop//Minor Project//face recognition//himanshu.jpeg")
himanshu_encoding = face_recognition.face_encodings(himanshu_image)[0]
satyam_image = face_recognition.load_image_file("C:/Users//HIMANSHU JAISWAL//Desktop//Minor Project//face recognition//satyam.jpeg")
satyam_encoding = face_recognition.face_encodings(satyam_image)[0]
raushan_image = face_recognition.load_image_file("C:/Users//HIMANSHU JAISWAL//Desktop//Minor Project//face recognition//raushan.jpeg")
raushan_encoding = face_recognition.face_encodings(raushan_image)[0]


known_face_encodings = [ankit_encoding, pragya_encoding, nisha_encoding, himanshu_encoding, satyam_encoding, raushan_encoding]
known_face_names = ["Ankit", "Pragya", "Nisha", "Himanshu","Satyam","Raushan"]

# list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwritter = csv.writer(f)

name = ""  # Initialize name

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        # Add the text if a person is present
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwritter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty("Attendance", cv2.WND_PROP_VISIBLE) < 1:
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()