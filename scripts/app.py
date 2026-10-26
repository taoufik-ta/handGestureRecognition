import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model
from PIL import Image


from scripts.gesture_utils import build_feature_vector, draw_landmarks_on_frame, estimate_gesture_from_landmarks


MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.h5"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoder.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    scaler = joblib.load(SCALER_PATH)
    base_options = mp.tasks.BaseOptions(model_asset_path=str(PROJECT_ROOT / "models" / "hand_landmarker.task"))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)
    return model, encoder, scaler, landmarker


def extract_features_from_frame(frame, landmarker):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, None, None

    hand_landmarks = result.hand_landmarks[0]
    features = build_feature_vector(hand_landmarks)
    landmark_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks], dtype=np.float32)
    return features.reshape(1, -1), landmark_points, hand_landmarks


def process_frame(frame):
    model, encoder, scaler, landmarker = load_resources()
    features, landmark_points, hand_landmarks = extract_features_from_frame(frame, landmarker)

    if features is None or landmark_points is None:
        return frame, "No hand detected"

    features = scaler.transform(features)
    probabilities = model.predict(features, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    model_gesture = encoder.inverse_transform([predicted_index])[0]
    heuristic_gesture, heuristic_confidence = estimate_gesture_from_landmarks(landmark_points)
    if float(probabilities[predicted_index]) < 0.70 or (float(probabilities[predicted_index]) < 0.90 and heuristic_gesture != model_gesture):
        gesture_name = heuristic_gesture
    else:
        gesture_name = model_gesture

    draw_landmarks_on_frame(frame, hand_landmarks)
    cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return frame, gesture_name


def main():
    st.set_page_config(page_title="Hand Gesture Recognition", page_icon="✋")
    st.title("Hand Gesture Recognition")
    st.write("Use your webcam to recognize five hand gestures in real time.")

    frame = st.camera_input("Camera")
    if frame is not None:
        image = Image.open(frame)
        frame_array = np.array(image)
        frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
        processed_frame, predicted_gesture = process_frame(frame_array)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        st.image(processed_frame, channels="RGB")
        st.success(f"Predicted gesture: {predicted_gesture}")

        


if __name__ == "__main__":
    main()
