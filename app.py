import cv2
import mediapipe as mp

# Function to find available camera indices
def find_available_camera_indices(max_index=10):
    available_indices = []
    
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_indices.append(index)
            cap.release()
    
    return available_indices

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Setup hand tracking
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Find available camera indices
available_indices = find_available_camera_indices()

if available_indices:
    print(f"Using camera index: {available_indices[0]}")  # Use the first available camera
    cap = cv2.VideoCapture(available_indices[0])
else:
    print("No cameras found.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Draw landmarks if hand is detected
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame with hand tracking
    cv2.imshow("Hand Gesture Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
