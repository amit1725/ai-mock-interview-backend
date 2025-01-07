import cv2
import numpy as np
import dlib
from keras.models import load_model
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load the pre-trained model
model = load_model('new_model.keras')

# Emotion labels based on AffectNet
emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

def preprocess_frame(frame, target_size=(96, 96)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, target_size)
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=(0, -1))

def analyze_eye_contact(landmarks):
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    def eye_openness(eye):
        width = np.linalg.norm(eye[0] - eye[3])
        height = (np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])) / 2
        return height / width

    left_openness = eye_openness(left_eye)
    right_openness = eye_openness(right_eye)

    return left_openness > 0.2 and right_openness > 0.2

def analyze_posture_and_movement(landmarks):
    nose_tip = landmarks[30]
    chin = landmarks[8]
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)

    eye_line = right_eye[0] - left_eye[0]
    tilt_ratio = abs(right_eye[1] - left_eye[1]) / eye_line
    tilt_threshold = 0.15

    if tilt_ratio < tilt_threshold and chin[1] > nose_tip[1]:
        return "Good"
    else:
        return "Bad"

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    files = request.files.getlist("video")
    total_frames = 0
    eye_contact_count = 0
    good_posture_count = 0
    emotion_totals = np.zeros(len(emotion_labels))
    results = []

    for file in files:
        # Read the image file into a numpy array using OpenCV
        in_memory_image = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(in_memory_image, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_frame)

        for face in faces:
            landmarks = predictor(gray_frame, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            input_data = preprocess_frame(face_crop)
            predictions = model.predict(input_data)[0]
            emotion_index = np.argmax(predictions)
            emotion = emotion_labels[emotion_index]
            confidence = predictions[emotion_index]

            eye_contact = analyze_eye_contact(landmarks)
            posture_analysis = analyze_posture_and_movement(landmarks)

            total_frames += 1
            if eye_contact:
                eye_contact_count += 1
            if posture_analysis == "Good":
                good_posture_count += 1
            emotion_totals += predictions

            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'eye_contact': 'Yes' if eye_contact else 'No',
                'posture': posture_analysis
            })

    if not results:
        return jsonify({"error": "No faces detected"}), 404

    # Calculate averages
    average_eye_contact = (eye_contact_count / total_frames) * 100 if total_frames > 0 else 0
    average_good_posture = (good_posture_count / total_frames) * 100 if total_frames > 0 else 0
    average_emotions = emotion_totals / total_frames if total_frames > 0 else np.zeros(len(emotion_labels))

    # Display session summary in the response
    summary = {
        "total_frames": total_frames,
        "average_eye_contact": average_eye_contact,
        "average_good_posture": average_good_posture,
        "average_emotions": {emotion_labels[i]: average_emotions[i] for i in range(len(emotion_labels))}
    }

    return jsonify(summary)

if __name__ == '__main__':
    app.run(debug=True)
