import cv2
import mediapipe
import numpy as np
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "dataset"
CSV_PATH = PROJECT_ROOT / "data" / "gestures.csv"
ALLOWED_GESTURES = ["thumb_up", "open_hand", "fist", "peace", "ok"]


def build_feature_vector(hand_landmarks) -> np.ndarray:
    """Convert hand landmarks to scale- and position-invariant features."""
    landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark], dtype=np.float32)
    wrist = landmarks[0]
    relative = landmarks - wrist

    palm_scale = np.linalg.norm(landmarks[9] - landmarks[0])
    if palm_scale < 1e-6:
        palm_scale = np.linalg.norm(landmarks[5] - landmarks[0])
    if palm_scale < 1e-6:
        palm_scale = 1.0

    normalized = relative / palm_scale
    return normalized.reshape(-1)


def extract_landmarks_and_save() -> None:
    """Read images from the dataset folders and save landmark features to gestures.csv."""
    mp_hands = mediapipe.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

    rows = []

    for gesture_dir in sorted(DATASET_DIR.iterdir()):
        if not gesture_dir.is_dir():
            continue

        label = gesture_dir.name
        if label not in ALLOWED_GESTURES:
            continue

        for image_path in sorted(gesture_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if not results.multi_hand_landmarks:
                continue

            hand_landmarks = results.multi_hand_landmarks[0]
            rows.append(build_feature_vector(hand_landmarks).tolist() + [label])

    if not rows:
        raise RuntimeError("No hand landmarks were extracted. Make sure the dataset contains images.")

    columns = [f"feature_{i}" for i in range(63)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved landmark data to {CSV_PATH}")


if __name__ == "__main__":
    extract_landmarks_and_save()
