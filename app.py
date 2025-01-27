import streamlit as st
import cv2
import mediapipe as mp
import time

# Set up the Streamlit page
st.set_page_config(page_title="Lion Gesture Detection", page_icon="🦁")
st.title("🦁 Lion Gesture Detection")

# Function to try different camera indices until a valid one is found
def find_valid_camera_index():
    for i in range(10):  # Check up to 10 camera indexes (can be adjusted based on your system)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():  # If the camera is successfully opened
            print(f"Camera found at index {i}")
            return i
        cap.release()
    return -1  # Return -1 if no valid camera is found

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Try to find a valid camera index
camera_index = find_valid_camera_index()

if camera_index == -1:
    st.error("No camera found. Please check the connections and try again.")
else:
    # Open the camera with the found index
    cap = cv2.VideoCapture(camera_index)

    # Start capturing video
    stframe = st.empty()  # Placeholder for video stream display

    while True:
        ret, frame = cap.read()
        
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # If hands are detected, draw landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the hand landmarks on the frame
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert the frame back to BGR for Streamlit compatibility
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame in the Streamlit app
        stframe.image(frame_bgr, channels="BGR", use_column_width=True)

        # Optionally add a sleep interval for smooth video streaming
        time.sleep(0.01)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera
    cap.release()
