import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from mediapipe.tasks.python import vision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
CSV_PATH = PROJECT_ROOT / "data" / "gestures.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "gesture_model.h5"
ENCODER_PATH = PROJECT_ROOT / "models" / "encoder.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
ALLOWED_GESTURES = ["thumb_up", "open_hand", "fist", "peace", "ok"]


def build_feature_vector(hand_landmarks) -> np.ndarray:
    """Convert hand landmarks to a 63-feature vector."""
    landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks], dtype=np.float32)
    wrist = landmarks[0]
    relative = landmarks - wrist

    palm_scale = np.linalg.norm(landmarks[9] - landmarks[0])
    if palm_scale < 1e-6:
        palm_scale = np.linalg.norm(landmarks[5] - landmarks[0])
    if palm_scale < 1e-6:
        palm_scale = 1.0

    normalized = relative / palm_scale
    return normalized.reshape(-1)


def extract_landmark_features(image_path: Path) -> np.ndarray | None:
    """Create a 63-feature vector from a gesture image using MediaPipe hand landmarks."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    base_options = mp.tasks.BaseOptions(model_asset_path=str(PROJECT_ROOT / "models" / "hand_landmarker.task"))
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    landmarker = vision.HandLandmarker.create_from_options(options)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return None

    return build_feature_vector(result.hand_landmarks[0])


def augment_features(X: np.ndarray, y: np.ndarray, times: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Create simple synthetic variations to improve generalization."""
    rng = np.random.default_rng(42)
    augmented_X = []
    augmented_y = []

    for sample, label in zip(X, y):
        augmented_X.append(sample)
        augmented_y.append(label)
        for _ in range(times):
            noise = rng.normal(0, 0.03, size=sample.shape)
            jittered = np.clip(sample + noise, -1.5, 1.5)
            augmented_X.append(jittered)
            augmented_y.append(label)

    return np.array(augmented_X, dtype=np.float32), np.array(augmented_y, dtype=np.int32)


def train_model() -> None:
    """Build a 63-feature dataset from the image folders and train the ANN."""
    rows = []
    for gesture_dir in sorted(DATASET_ROOT.iterdir()):
        if not gesture_dir.is_dir():
            continue

        label = gesture_dir.name
        if label not in ALLOWED_GESTURES:
            continue

        for image_path in sorted(gesture_dir.glob("*.jpg")):
            features = extract_landmark_features(image_path)
            if features is None:
                continue
            rows.append(features.tolist() + [label])

    if not rows:
        raise ValueError("No valid gesture data found. Place images in the five expected folders under data/dataset.")

    columns = [f"feature_{i}" for i in range(63)] + ["label"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(CSV_PATH, index=False)

    X = df.drop(columns=["label"]).values.astype(np.float32)
    y = df["label"].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_aug, y_train_aug = augment_features(X_train, y_train, times=2)

    num_classes = len(label_encoder.classes_)
    model = Sequential(
        [
            Input(shape=(63,)),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)]
    model.fit(X_train_aug, y_train_aug, epochs=80, batch_size=32, validation_split=0.1, callbacks=callbacks, verbose=1)

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    model.save(MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved encoder to {ENCODER_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    train_model()
