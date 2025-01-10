import os
import cv2
import numpy as np
import requests
from keras.models import load_model
from flask import Flask, jsonify, request
from flask_cors import CORS
from mtcnn import MTCNN  # Import MTCNN for face and landmark detection
import tensorflow as tf

# Disable GPU usage (use CPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit TensorFlow memory usage to a specified amount (e.g., 4096 MB)
physical_devices = tf.config.list_physical_devices('CPU')
if physical_devices:
    tf.config.set_logical_device_configuration(
        physical_devices[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # Set an appropriate limit in MB
    )

# Dropbox direct download URL
DROPBOX_URL = "https://www.dropbox.com/scl/fi/fm5a49snvm6sk3ouq0l3z/affectnet_model.keras?rlkey=a9cc28vkdnt8jb6az289nab0n&st=uns5oeie&dl=1"
MODEL_PATH = "affectnet_model.keras"

# Function to download the model from Dropbox
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        try:
            response = requests.get(DROPBOX_URL, stream=True)
            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Model downloaded successfully.")
            else:
                raise Exception(f"Failed to download model. HTTP Status: {response.status_code}")
        except Exception as e:
            print(f"Error during model download: {e}")
            raise Exception("Failed to download model from Dropbox.")
    else:
        print("Model already exists locally.")

# Download the model before starting the application
download_model()

# Load the pre-trained model
model = load_model(MODEL_PATH)

# Emotion labels based on AffectNet
emotion_labels = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize MTCNN detector
detector = MTCNN()

# Initialize Flask app and CORS
app = Flask(__name__)
CORS(app)

def preprocess_frame(frame, target_size=(96, 96)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, target_size)
    normalized_frame = resized_frame / 255.0
    return np.expand_dims(normalized_frame, axis=(0, -1))

def analyze_eye_contact(landmarks):
    if len(landmarks) < 10:  # Check if we have at least 10 landmarks (5 per eye)
        return False  # If not, return False indicating no eye contact

    # For MTCNN, landmarks for eyes are typically 5 points each
    left_eye = landmarks[0:5]  # Left eye landmarks
    right_eye = landmarks[5:10]  # Right eye landmarks

    def eye_openness(eye):
        if len(eye) < 5:
            return 0  # Return 0 if eye landmarks are not sufficient

        # Calculate the openness based on the first 4 points for width and height
        width = np.linalg.norm(eye[0] - eye[3])
        height = (np.linalg.norm(eye[1] - eye[4]) + np.linalg.norm(eye[2] - eye[3])) / 2
        return height / width

    left_openness = eye_openness(left_eye)
    right_openness = eye_openness(right_eye)

    return left_openness > 0.2 and right_openness > 0.2

def analyze_posture_and_movement(landmarks):
    if len(landmarks) < 30:  # Check if we have enough landmarks for posture analysis
        return "Bad"  # If not, return "Bad" as a default posture

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

@app.route('/')
def home():
    return "Backend is running! Access the API at /api/analyze"

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

        faces = detector.detect_faces(frame)

        for face in faces:
            # Get bounding box and landmarks
            x, y, w, h = face['box']
            landmarks = face['keypoints']
            
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                continue

            input_data = preprocess_frame(face_crop)
            predictions = model.predict(input_data)[0]
            emotion_index = np.argmax(predictions)
            emotion = emotion_labels[emotion_index]
            confidence = predictions[emotion_index]

            # Check eye contact (using the 5 points of the eyes)
            eye_contact = analyze_eye_contact(np.array(list(landmarks.values())))
            posture_analysis = analyze_posture_and_movement(np.array(list(landmarks.values())))

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

    # Calculate percentages
    eye_contact_percentage = (eye_contact_count / total_frames) * 100 if total_frames > 0 else 0
    good_posture_percentage = (good_posture_count / total_frames) * 100 if total_frames > 0 else 0
    average_emotions = emotion_totals / total_frames if total_frames > 0 else np.zeros(len(emotion_labels))

    # Display session summary in the response
    summary = {
        "total_frames": total_frames,
        "eye_contact_percentage": eye_contact_percentage,
        "good_posture_percentage": good_posture_percentage,
        "average_emotions": {emotion_labels[i]: average_emotions[i] for i in range(len(emotion_labels))}
    }

    return jsonify(summary)

if __name__ == '__main__':
    # Ensure the app binds to the correct port
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
