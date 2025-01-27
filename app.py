import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

# Set up Streamlit page
st.set_page_config(page_title="Lion Gesture Detection", page_icon="🦁")
st.title("🦁 Lion Gesture Detection")

# MediaPipe hands setup
@st.cache_resource
def setup_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils
    return mp_hands, hands, mp_draw

mp_hands, hands, mp_draw = setup_mediapipe()

def check_lion_gesture(hand_landmarks):
    """Checks for lion claw-like gesture"""
    finger_tips = [8, 12, 16, 20]
    finger_mids = [6, 10, 14, 18]
    
    is_lion_gesture = True
    
    for tip, mid in zip(finger_tips, finger_mids):
        tip_y = hand_landmarks.landmark[tip].y
        mid_y = hand_landmarks.landmark[mid].y
        
        if tip_y >= mid_y:
            is_lion_gesture = False
            break
    
    thumb_tip = hand_landmarks.landmark[4].x
    thumb_ip = hand_landmarks.landmark[3].x
    
    if hand_landmarks.landmark[5].x < hand_landmarks.landmark[17].x:
        if thumb_tip >= thumb_ip:
            is_lion_gesture = False
    else:
        if thumb_tip <= thumb_ip:
            is_lion_gesture = False
    
    return is_lion_gesture

# Sidebar instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Click 'Start Camera' to begin detecting gestures.
    2. Show your hand like a lion's claw:
        - All fingers extended
        - Slightly curved
    3. The app will detect the gesture and display results on the frame.
    4. Press 'Stop' to end the detection.
    """)

# Camera input (for online deployment)
frame_placeholder = st.empty()

# Camera start button
if st.button("Start Camera"):
    cap = cv2.VideoCapture(0)  # Access the webcam
    last_detection_time = 0
    COOLDOWN = 2  # Avoid continuous detection without cooldown
    
    stop_button = st.button("Stop")
    
    while cap.isOpened() and not stop_button:
        success, img = cap.read()
        if not success:
            st.warning("Unable to access webcam")
            break
        
        # Convert image to RGB (required for MediaPipe)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe Hands
        results = hands.process(imgRGB)

        status_text = "Waiting for Lion Gesture..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the image
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if lion gesture is detected
                if check_lion_gesture(hand_landmarks):
                    current_time = time.time()
                    if current_time - last_detection_time >= COOLDOWN:
                        status_text = "ROAR!!!"
                        last_detection_time = current_time
                        # Add visual effect
                        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
                    else:
                        status_text = "Lion Gesture Detected!"
        
        # Add status text
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display the frame with status
        frame_placeholder.image(img, channels="BGR", use_column_width=True)
    
    cap.release()  # Release the camera when done
    frame_placeholder.empty()
    st.success("Camera stopped")
