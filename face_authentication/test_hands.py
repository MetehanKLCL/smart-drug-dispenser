import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import os
import urllib.request

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(HAND_MODEL_PATH):
    print("Hand model downloading...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    print("Done.")

hand_options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def detect_gesture(hand):
    """Detects hand gesture based on finger positions."""
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb: compare x axis
    if hand[tips[0]].x < hand[tips[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: compare y axis
    for i in range(1, 5):
        if hand[tips[i]].y < hand[tips[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    total = sum(fingers)

    if fingers == [0, 1, 0, 0, 0]:
        return "One"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Two / Peace"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [1, 1, 1, 1, 1]:
        return "High Five"
    elif total == 0:
        return "Fist"
    elif fingers == [0, 1, 0, 0, 1]:
        return "Rock"
    else:
        return f"Fingers: {total}"


cap = cv2.VideoCapture(0)

with vision.HandLandmarker.create_from_options(hand_options) as detector:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.flip(image, 1)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        h, w, _ = image.shape

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                # Draw landmarks
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (cx, cy), 4, (0, 255, 0), -1)

                # Detect and show gesture
                gesture = detect_gesture(hand)
                cv2.putText(image, gesture, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        else:
            cv2.putText(image, "No hand", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()