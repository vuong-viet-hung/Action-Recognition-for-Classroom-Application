from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


DATA_PATH = "data/UCF101/UCF-101"
KEYPOINTS_DIMS = (120, 33 * 4)


def extract_keypoints(video_path: int | str) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    assert capture.isOpened()

    keypoints = []

    with mp.solutions.pose.Pose() as pose:
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break

            # To improve performance, mark the frame as not writeable so that it can be passed by reference.
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)

            # Draw the pose annotation on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

            # Display the frame.
            # cv2.imshow('MediaPipe Pose', frame)
            # if cv2.waitKey(5) & 0xFF == ord('q'):
            #     break

            # Add each frame's keypoint
            if results.pose_landmarks is None:
                continue
            keypoint = np.array(
                [
                    [value.x, value.y, value.z, value.visibility]
                    for value in results.pose_landmarks.landmark
                ]
            )
            keypoints.extend(keypoint)

    capture.release()

    try:
        keypoints = cv2.resize(np.array(keypoints), KEYPOINTS_DIMS[::-1])
    except cv2.error:
        return np.empty(KEYPOINTS_DIMS)

    return keypoints


def demo() -> None:
    video_path = Path(f"{DATA_PATH}/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi")
    assert video_path.is_file()
    keypoints = extract_keypoints(str(video_path))
    print(keypoints.shape)


if __name__ == "__main__":
    demo()
