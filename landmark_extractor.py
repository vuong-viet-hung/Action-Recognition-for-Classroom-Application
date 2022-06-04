from pathlib import Path

import cv2
import mediapipe as mp


DATA_PATH = "data/UCF101/UCF-101"


def extract_landmark(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    assert capture.isOpened()

    with mp.solutions.pose.Pose() as pose:
        while capture.isOpened():
            success, image = capture.read()
            if not success:
                break

            # To improve performance, mark the frane as not writeable so that it can be passed by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
            )

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Pose', cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    capture.release()


def demo() -> None:
    video_path = Path(f"{DATA_PATH}/BalanceBeam/v_BalanceBeam_g03_c01.avi")
    assert video_path.is_file()
    extract_landmark(video_path)


if __name__ == "__main__":
    demo()
