import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
canvas = None
prev_index = None
prev_thumb = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        h, w = frame.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            handedness = results.multi_handedness[0].classification[0].label

            # Check index finger up
            index_up = landmarks[8].y < landmarks[6].y

            # Check thumb up
            thumb_tip = landmarks[4].x
            thumb_ip = landmarks[3].x
            if handedness == "Left":
                thumb_up = thumb_tip > thumb_ip
            else:
                thumb_up = thumb_tip < thumb_ip

            # Check all fingers up
            fingers_up = [
                landmarks[8].y < landmarks[6].y,   # index
                landmarks[12].y < landmarks[10].y, # middle
                landmarks[16].y < landmarks[14].y, # ring
                landmarks[20].y < landmarks[18].y  # pinky
            ]
            all_up = thumb_up and all(fingers_up)

            if all_up:
                break  # Stop the program

            # Clear canvas if palm shown (four fingers up, thumb down)
            if not thumb_up and all(fingers_up):
                canvas = np.zeros_like(canvas)

            # Get positions
            h, w, _ = frame.shape
            index_x, index_y = int(landmarks[8].x * w), int(landmarks[8].y * h)
            thumb_x, thumb_y = int(landmarks[4].x * w), int(landmarks[4].y * h)

            if index_up and not thumb_up:
                if prev_index:
                    cv2.line(canvas, prev_index, (index_x, index_y), (0, 255, 0), 5)
                prev_index = (index_x, index_y)
                prev_thumb = None
            elif thumb_up:
                if prev_thumb:
                    cv2.line(canvas, prev_thumb, (thumb_x, thumb_y), (0, 0, 0), 20)
                prev_thumb = (thumb_x, thumb_y)
                prev_index = None
            else:
                prev_index = None
                prev_thumb = None

    # Combine canvas and frame
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Drawing", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()