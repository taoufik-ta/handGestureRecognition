from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"


def capture_images(gesture_name: str, num_images: int = 100) -> None:
    """Open the webcam and capture a set of images for a gesture."""
    gesture_name = gesture_name.strip().lower().replace(" ", "_")
    if not gesture_name:
        raise ValueError("Gesture name cannot be empty.")

    save_dir = DATASET_DIR / gesture_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the webcam.")

    count = 0
    print(f"Capturing {num_images} images for gesture: {gesture_name}")
    print("Press 's' to capture an image.")
    print("Press 'q' to stop early.")

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame,
            f"Press 's' to capture {gesture_name}: {count}/{num_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Capture Images", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            image_path = save_dir / f"{gesture_name}_{count:03d}.jpg"
            cv2.imwrite(str(image_path), frame)
            count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images to {save_dir}")


if __name__ == "__main__":
    gesture_name = input("Enter the gesture name (for example: thumb_up): ").strip()
    capture_images(gesture_name)
