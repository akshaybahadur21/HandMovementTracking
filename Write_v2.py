import numpy as np
import cv2
from collections import deque
import mediapipe as mp
from utils.utils_v2 import get_idx_to_coordinates, rescale_frame

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def main():
    hands = mp_hands.Hands(
        min_detection_confidence=0.7, min_tracking_confidence=0.7)
    hand_landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
    hand_connection_drawing_spec = mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
    cap = cv2.VideoCapture(0)
    pts = deque(maxlen=64)
    while cap.isOpened():
        idx_to_coordinates = {}
        ret, image = cap.read()
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results_hand.multi_hand_landmarks:
            for hand_landmarks in results_hand.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_drawing_spec,
                    connection_drawing_spec=hand_connection_drawing_spec)
                idx_to_coordinates = get_idx_to_coordinates(image, results_hand)
        if 8 in idx_to_coordinates:
            pts.appendleft(idx_to_coordinates[8])  # Index Finger
        for i in range(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            thick = int(np.sqrt(len(pts) / float(i + 1)) * 4.5)
            cv2.line(image, pts[i - 1], pts[i], (0, 255, 0), thick)
        cv2.imshow("Res", rescale_frame(image, percent=130))
        if cv2.waitKey(5) & 0xFF == 27:
            break
    hands.close()
    cap.release()


if __name__ == '__main__':
    main()
