
# Mediapipe landmarks, path movements og path lengths
import cv2
import mediapipe as mp
import math


def tracking_and_path(video):
    global x, y
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    # Initialize VideoCapture object and read the first frame
    cap = cv2.VideoCapture(video) #Video.mp4
    ret, frame = cap.read()
    # Initialize Hand Tracking Module
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:

        # Initialize variables for storing the movement paths and path lengths of both hands
        path_left = []
        path_right = []
        path_length_left = 0
        path_length_right = 0

        while ret:

            # Convert frame to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process image with Mediapipe
            results = hands.process(image)

            # Extract hand landmarks and drawing specs from Mediapipe results
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extract the position of the wrist (landmark 0)
                    x, y = int(hand_landmarks.landmark[0].x * frame.shape[1]), \
                           int(hand_landmarks.landmark[0].y * frame.shape[0])

                    # Add wrist position to the corresponding path list
                    if hand_landmarks == results.multi_hand_landmarks[0]:
                        path_left.append((x, y))
                    elif hand_landmarks == results.multi_hand_landmarks[1]:
                        path_right.append((x, y))

            # Calculate the movement path length for each wrist
            if len(path_left) > 1:
                for i in range(1, len(path_left)):
                    path_length_left += math.sqrt((path_left[i][0] - path_left[i - 1][0]) ** 2
                                                  + (path_left[i][1] - path_left[i - 1][1]) ** 2)
            if len(path_right) > 1:
                for i in range(1, len(path_right)):
                    path_length_right += math.sqrt((path_right[i][0] - path_right[i - 1][0]) ** 2
                                                   + (path_right[i][1] - path_right[i - 1][1]) ** 2)

            # Show the video frame with hand landmarks

            resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('handtracking', resized_frame)

            # Draw the movement path of both wrists in a separate window
            path_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for i in range(len(path_left)):
                cv2.circle(path_image, path_left[i], 3, (0, 255, 0), -1)
            for i in range(len(path_right)):
                cv2.circle(path_image, path_right[i], 3, (255, 0, 0), -1)
            resized_frame2 = cv2.resize(path_image, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('Wrist Movement Path', resized_frame2)

            # Read the next frame
            ret, frame = cap.read()

            # Exit the loop if the video has ended
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Print the length of the movement path for each wrist
        print("Left Wrist Path Length:", path_length_left)
        print("Right Wrist Path Length:", path_length_right)
    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()


tracking_and_path("Handmovements.mp4")
