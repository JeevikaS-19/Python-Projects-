import cv2
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

addition_state = 'idle'
numbers = []
total_history = []
no_fingers_timer = 0

def count_raised_fingers(landmarks, handedness):
    finger_up = []
    tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            finger_up.append(True)
        else:
            finger_up.append(False)

    thumb_tip = landmarks[4].x
    thumb_ip = landmarks[3].x
    if handedness == "Left":
        thumb_up = thumb_tip > thumb_ip
    else:
        thumb_up = thumb_tip < thumb_ip

    count = sum(finger_up) + (1 if thumb_up else 0)
    return count

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if numbers:
        cv2.putText(frame, f"Current sum: {sum(numbers)}", (10, frame.shape[0] - 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    total_fingers = 0
    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = results.multi_handedness[idx].classification[0].label
            finger_count = count_raised_fingers(hand_landmarks.landmark, handedness)
            total_fingers += finger_count
            cv2.putText(frame, f"{handedness}: {finger_count} fingers", (10, 50 + idx * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Current fingers: {total_fingers}", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    total_history.append(total_fingers)
    if len(total_history) > 5:
        total_history.pop(0)
    recent_total = total_history[-3:] if len(total_history) >= 3 else []
    if len(recent_total) == 3 and all(t == recent_total[0] for t in recent_total):
        stable_total = recent_total[0]
        if stable_total == 0:
            no_fingers_timer += 1
        else:
            no_fingers_timer = 0
        if no_fingers_timer >= 150 and numbers:
            numbers = []
            addition_state = 'idle'
        if addition_state == 'idle' and stable_total > 0:
            numbers.append(stable_total)
            addition_state = 'wait_zero'
        elif addition_state == 'wait_zero' and stable_total == 0:
            addition_state = 'number'
        elif addition_state == 'number' and stable_total > 0:
            numbers.append(stable_total)
            addition_state = 'wait_zero'

    cv2.imshow("Finger Counter", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()