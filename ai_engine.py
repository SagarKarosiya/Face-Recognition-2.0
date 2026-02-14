import face_recognition
import cv2
import numpy as np
from database import get_faces


# ---------------- FACE ENCODING ----------------
def extract_encoding(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(encodings) > 0:
        return encodings[0], face_locations
    else:
        return None, None


# ---------------- FACE RECOGNITION ----------------
def recognize_face(encoding):
    names_db, encodings_db = get_faces()

    if len(encodings_db) == 0:
        return "No Database"

    matches = face_recognition.compare_faces(encodings_db, encoding)
    face_distances = face_recognition.face_distance(encodings_db, encoding)

    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            return names_db[best_match_index]

    return "Unknown"
