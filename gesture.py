# Step 1: Import Libraries
import cv2
import mediapipe as mp

# Explanation:
# cv2: OpenCV library for video capture and processing
# mediapipe: Library for hand tracking and landmark detection

# Step 2: Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Explanation:
# mp_hands: Accesses the hands solution in MediaPipe
# hands: Initializes the hands module for hand detection
# mp_draw: Utility to draw hand landmarks on the frames

# Step 3: Initialize Video Capture
cap = cv2.VideoCapture(0)

# Explanation:
# cap: Captures video from the default camera (usually the webcam)

# Step 4: Capture and Process Each Frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Explanation:
    # ret, frame: Reads a frame from the video capture
    # cv2.flip(frame, 1): Flips the frame horizontally for a mirror view
    # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB): Converts the frame from BGR to RGB
    # hands.process(frame_rgb): Processes the frame to detect hand landmarks
    # mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS): Draws the detected hand landmarks on the frame

    # Step 5: Display the Frame
    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Explanation:
    # cv2.imshow('Hand Gesture Recognition', frame): Displays the frame with hand landmarks
    # cv2.waitKey(1): Waits for 1 millisecond for a key press. If 'q' is pressed, the loop breaks

# Step 6: Release Resources
cap.release()
cv2.destroyAllWindows()

# Explanation:
# cap.release(): Releases the webcam
# cv2.destroyAllWindows(): Closes all OpenCV windows
