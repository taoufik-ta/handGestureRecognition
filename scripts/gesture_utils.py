from __future__ import annotations

import cv2
import numpy as np

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


def build_feature_vector(hand_landmarks: object) -> np.ndarray:
    """Convert hand landmarks to a 63-feature vector."""
    if isinstance(hand_landmarks, np.ndarray):
        landmarks = hand_landmarks.astype(np.float32)
    else:
        landmarks = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks],
            dtype=np.float32,
        )
    wrist = landmarks[0]
    relative = landmarks - wrist

    palm_scale = np.linalg.norm(landmarks[9] - landmarks[0])
    if palm_scale < 1e-6:
        palm_scale = np.linalg.norm(landmarks[5] - landmarks[0])
    if palm_scale < 1e-6:
        palm_scale = 1.0

    normalized = relative / palm_scale
    return normalized.reshape(-1)


def estimate_gesture_from_landmarks(landmarks: np.ndarray) -> tuple[str, float]:
    """Use simple geometric rules to classify common hand gestures from landmarks."""
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (21, 3):
        raise ValueError("Expected a 21x3 landmark array")

    palm = landmarks[9]
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_dist = float(np.linalg.norm(thumb_tip - palm))
    index_dist = float(np.linalg.norm(index_tip - palm))
    middle_dist = float(np.linalg.norm(middle_tip - palm))
    ring_dist = float(np.linalg.norm(ring_tip - palm))
    pinky_dist = float(np.linalg.norm(pinky_tip - palm))
    finger_dists = np.array([index_dist, middle_dist, ring_dist, pinky_dist], dtype=np.float32)

    open_score = float(np.sum(finger_dists > 0.16) + (thumb_dist > 0.14))
    fist_score = float(np.sum(finger_dists < 0.10) + (thumb_dist < 0.12))
    thumb_up_score = float((thumb_dist > 0.16) + (np.mean(finger_dists) < 0.12) + (thumb_tip[0] > palm[0] + 0.05))
    peace_score = float(
        (index_dist > 0.16)
        + (middle_dist > 0.16)
        + (ring_dist < 0.10)
        + (pinky_dist < 0.10)
        + (abs(index_tip[1] - middle_tip[1]) < 0.06)
    )
    ok_score = float((thumb_dist > 0.12) + (index_dist > 0.10) + (np.linalg.norm(thumb_tip - index_tip) < 0.12))

    scores = {
        "open_hand": open_score,
        "fist": fist_score,
        "thumb_up": thumb_up_score,
        "peace": peace_score,
        "ok": ok_score,
    }

    gesture = max(scores, key=scores.get)
    confidence = min(0.99, max(0.8, scores[gesture] / 5.0))
    return gesture, confidence


def draw_landmarks_on_frame(frame: np.ndarray, hand_landmarks: object) -> None:
    """Draw the detected hand landmarks and their connections on the frame."""
    h, w = frame.shape[:2]
    points = np.array([[lm.x * w, lm.y * h] for lm in hand_landmarks], dtype=np.float32)

    for a, b in HAND_CONNECTIONS:
        pt_a = tuple(map(int, points[a]))
        pt_b = tuple(map(int, points[b]))
        cv2.line(frame, pt_a, pt_b, (0, 255, 0), 1)

    for idx, (x, y) in enumerate(points):
        color = (255, 0, 0) if idx == 0 else (0, 0, 255)
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
