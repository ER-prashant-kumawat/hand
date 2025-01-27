import streamlit as st
import cv2
import mediapipe as mp
import time

# Function to find valid camera index
def find_valid_camera_index():
    for i in range(20):  # Try up to 20 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            return i
        cap.release()
    return -1  # Return -1 if no valid camera is found

# Set up the Streamlit page
st.set_page_config(page_title="Lion Gesture Detection", page_icon="🦁")
st.title("🦁 Lion Gesture Detection")

# Display a loading message while the camera is being set up
st.write("Setting up the camera...")

# Find a valid camera index
camera_index = find_valid_camera_index()

# If no camera found, display a message and stop
if camera_index == -1:
    st.error("No camera found. Please check the connections and try again.")
else:
    # Initialize MediaPipe Hands module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Start the video capture loop
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame")
            break

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and get hand landmarks
        results = hands.process(rgb_frame)

        # If hands are detected, draw landmarks
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        # Convert the frame back to BGR for display in OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Add a break condition to stop the loop gracefully when the Streamlit app is closed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera when done
    cap.release()
    cv2.destroyAllWindows()
