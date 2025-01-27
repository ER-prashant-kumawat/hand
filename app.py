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
    hands = mp_hands.Hands(min_detection_confidence=0.7)
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
    1. Upload a video file of your hand gestures.
    2. The app will process the video and detect the 'lion gesture':
        - All fingers extended
        - Slightly curved like a lion's claw
    3. Detection results will be displayed on the video frames.
    """)

# Video uploader
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    last_detection_time = 0
    COOLDOWN = 2

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        status_text = "Waiting for Lion Gesture..."

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if check_lion_gesture(hand_landmarks):
                    current_time = time.time()
                    if current_time - last_detection_time >= COOLDOWN:
                        last_detection_time = current_time
                        status_text = "ROAR!!!"
                        # Add visual effect
                        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 255), 10)
                    else:
                        status_text = "Lion Gesture Detected!"

        # Add status text to the frame
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the video frame
        frame_placeholder.image(img, channels="BGR")
        status_placeholder.text(status_text)

    cap.release()
    st.success("Video processing complete!")
else:
    st.info("Please upload a video file to start detection.")
