import streamlit as st
import cv2
import mediapipe as mp
import time

# Set up the Streamlit page
st.set_page_config(page_title="Lion Gesture Detection", page_icon="🦁")
st.title("🦁 Lion Gesture Detection")

# MediaPipe Hands setup
@st.cache_resource
def setup_mediapipe():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
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

# Start Camera button
if st.button("Start Camera"):
    # Create a placeholder for video frames
    frame_placeholder = st.empty()
    
    # Open the webcam for capturing video
    
    cap = cv2.VideoCapture(1)  # or try another index
    if not cap.isOpened():
        st.error("Camera not found. Please check your camera settings.")
    else:
        st.success("Camera is ready!")
    
    last_detection_time = 0
    COOLDOWN = 2  # cooldown time for gesture detection
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)
        
        # Convert the image to RGB (as OpenCV uses BGR)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame using Mediapipe
        results = hands.process(imgRGB)
        
        status_text = "Waiting for Lion Gesture..."
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if check_lion_gesture(hand_landmarks):
                    current_time = time.time()
                    if current_time - last_detection_time >= COOLDOWN:
                        last_detection_time = current_time
                        status_text = "ROAR!!!"
                        # Add a visual alert (red border)
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                    else:
                        status_text = "Lion Gesture Detected!"
        
        # Add status text on the frame
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Show the frame in Streamlit
        frame_placeholder.image(frame, channels="BGR")
        
        # Stop the loop if 'Stop Camera' button is pressed
        if st.button("Stop Camera"):
            cap.release()
            break
    
    # Release the camera when done
    cap.release()

