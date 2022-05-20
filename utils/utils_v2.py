from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
import cv2


def get_idx_to_coordinates(image, results, VISIBILITY_THRESHOLD=0.5, PRESENCE_THRESHOLD=0.5):
    idx_to_coordinates = {}
    image_rows, image_cols, _ = image.shape
    try:
        for idx, landmark in enumerate(results.multi_hand_landmarks[0].landmark):
            if ((landmark.HasField('visibility') and
                 landmark.visibility < VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                           image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px
    except:
        pass
    return idx_to_coordinates


def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
