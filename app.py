from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
from database import *
from ai_engine import extract_encoding, recognize_face

app = Flask(__name__)
init_db()

# Global Variables
latest_frame = None
current_name = "Unknown"

# Initialize Camera
camera = cv2.VideoCapture(0)

# ---------------- DASHBOARD ----------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global latest_frame, current_name

    while True:
        success, frame = camera.read()
        if not success:
            break

        latest_frame = frame.copy()

        encoding, faces = extract_encoding(frame)

        if encoding is not None:
            name = recognize_face(encoding)
            current_name = name

            for (top, right, bottom, left) in faces:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name,
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- SAVE CAPTURE ----------------
@app.route("/save_capture", methods=["POST"])
def save_capture():
    global latest_frame

    name = request.form.get("name")

    if latest_frame is None or not name:
        return redirect(url_for("dashboard"))

    encoding, faces = extract_encoding(latest_frame)

    if encoding is not None:
        insert_face(name, encoding)

    return redirect(url_for("dashboard"))


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
