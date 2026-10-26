from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from mediapipe.tasks.python import vision

from scripts.gesture_utils import build_feature_vector as build_landmark_features
from scripts.gesture_utils import draw_landmarks_on_frame, estimate_gesture_from_landmarks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.h5"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoder.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"


def load_model_and_encoder():
    """Load the trained model, encoder, and scaler if they exist."""
    try:
        model = load_model(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        base_options = mp.tasks.BaseOptions(model_asset_path=str(PROJECT_ROOT / "models" / "hand_landmarker.task"))
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
        landmarker = vision.HandLandmarker.create_from_options(options)
        return model, encoder, scaler, landmarker
    except FileNotFoundError:
        print("Model files not found. Train the model first with train_model.py")
        return None, None, None, None


def extract_features_from_frame(frame: np.ndarray, landmarker) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, np.ndarray | None, object | None]:
    """Extract landmark features from the current frame."""
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None, None, None, None

    hand_landmarks = result.hand_landmarks[0]
    features = build_landmark_features(hand_landmarks)

    h, w = frame.shape[:2]
    coords = np.array([[landmark.x * w, landmark.y * h] for landmark in hand_landmarks], dtype=np.float32)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    pad = 20
    x1 = int(max(0, x_min - pad))
    y1 = int(max(0, y_min - pad))
    x2 = int(min(w, x_max + pad))
    y2 = int(min(h, y_max + pad))
    landmark_points = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks], dtype=np.float32)
    return features.reshape(1, -1), (x1, y1, max(1, x2 - x1), max(1, y2 - y1)), landmark_points, hand_landmarks


def run_real_time_detection() -> None:
    """Run the webcam and show the predicted gesture in real time."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the webcam.")

    model, encoder, scaler, landmarker = load_model_and_encoder()
    prediction_history = deque(maxlen=5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        features, bounding_box, landmark_points, hand_landmarks = (
            extract_features_from_frame(frame, landmarker) if landmarker is not None else (None, None, None, None)
        )

        gesture_name = "No hand detected"
        if features is not None and model is not None and encoder is not None and scaler is not None:
            x, y, w, h = bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            draw_landmarks_on_frame(frame, hand_landmarks)

            features = scaler.transform(features)
            probabilities = model.predict(features, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_index])
            prediction_history.append(predicted_index)

            if len(prediction_history) >= 3:
                counts = Counter(prediction_history)
                predicted_index = counts.most_common(1)[0][0]

            model_gesture = encoder.inverse_transform([predicted_index])[0]
            heuristic_gesture, heuristic_confidence = estimate_gesture_from_landmarks(landmark_points)

            if confidence < 0.70 or (confidence < 0.90 and heuristic_gesture != model_gesture):
                gesture_name = heuristic_gesture
                confidence_label = heuristic_confidence
            else:
                gesture_name = model_gesture
                confidence_label = confidence

            if confidence_label < 0.85:
                gesture_name = f"{gesture_name} (?)"
        elif model is None or encoder is None or scaler is None:
            if landmark_points is not None:
                heuristic_gesture, _ = estimate_gesture_from_landmarks(landmark_points)
                gesture_name = heuristic_gesture
            else:
                gesture_name = "Train model first"

        cv2.putText(frame, gesture_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "Press q to quit", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Hand Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_real_time_detection()
