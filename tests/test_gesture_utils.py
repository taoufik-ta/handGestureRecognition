import unittest

import numpy as np

from scripts.gesture_utils import build_feature_vector, estimate_gesture_from_landmarks


class GestureUtilsTests(unittest.TestCase):
    def test_build_feature_vector_has_expected_shape(self):
        landmarks = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.45, 0.48, 0.0],
                [0.42, 0.44, 0.0],
                [0.38, 0.40, 0.0],
                [0.34, 0.36, 0.0],
                [0.48, 0.38, 0.0],
                [0.42, 0.30, 0.0],
                [0.36, 0.22, 0.0],
                [0.30, 0.16, 0.0],
                [0.49, 0.36, 0.0],
                [0.43, 0.24, 0.0],
                [0.37, 0.16, 0.0],
                [0.31, 0.10, 0.0],
                [0.50, 0.34, 0.0],
                [0.44, 0.22, 0.0],
                [0.38, 0.14, 0.0],
                [0.32, 0.08, 0.0],
                [0.51, 0.32, 0.0],
                [0.45, 0.20, 0.0],
                [0.39, 0.12, 0.0],
                [0.33, 0.06, 0.0],
            ],
            dtype=np.float32,
        )

        features = build_feature_vector(landmarks)
        self.assertEqual(features.shape, (63,))

    def test_open_hand_is_detected_from_simple_landmarks(self):
        landmarks = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.45, 0.48, 0.0],
                [0.42, 0.44, 0.0],
                [0.38, 0.40, 0.0],
                [0.34, 0.36, 0.0],
                [0.48, 0.38, 0.0],
                [0.42, 0.30, 0.0],
                [0.36, 0.22, 0.0],
                [0.30, 0.16, 0.0],
                [0.49, 0.36, 0.0],
                [0.43, 0.24, 0.0],
                [0.37, 0.16, 0.0],
                [0.31, 0.10, 0.0],
                [0.50, 0.34, 0.0],
                [0.44, 0.22, 0.0],
                [0.38, 0.14, 0.0],
                [0.32, 0.08, 0.0],
                [0.51, 0.32, 0.0],
                [0.45, 0.20, 0.0],
                [0.39, 0.12, 0.0],
                [0.33, 0.06, 0.0],
            ],
            dtype=np.float32,
        )

        gesture, confidence = estimate_gesture_from_landmarks(landmarks)
        self.assertEqual(gesture, "open_hand")
        self.assertGreaterEqual(confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
