import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import random
import time
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

# Mediapipe Initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Load ResNet50 Model
base_model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
model = Model(inputs=base_model.input, outputs=base_model.output)

# Parameters
EYE_AR_THRESH = 0.25
BLINK_CONSEC_FRAMES = 2
MOTION_LEVEL_THRESHOLD = 50
TEXTURE_THRESHOLD = 50
CHALLENGE_INTERVAL = 7  # Seconds between challenges
CHALLENGE_TIMEOUT = 10  # Seconds to complete a challenge
STABILITY_WINDOW = 5  # Number of challenges to average
LIVENESS_CONFIRMATION_THRESHOLD = 3  # Minimum successful challenges to confirm liveness
DATASET_SIZE = 300  # Number of embeddings to collect

# Challenges
CHALLENGES = [
    "Blink", "Nod Up", "Nod Down", "Turn Left", "Turn Right",
    "Tilt Left", "Tilt Right", "Shake Head"
]

# State Variables
blink_counter = 0
last_challenge_time = 0
challenge = None
challenge_start_time = None
challenge_success = False
successful_challenges = 0
completed_challenges = 0
prev_nose_y = None
prev_face_x = None
prev_face_y = None
stable_liveness = False

# Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Analyze Texture using Laplacian Variance
def analyze_texture(gray_frame):
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return laplacian_var

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Liveness Detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    challenge_success = False  # Reset challenge success for this frame

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            frame_height, frame_width = frame.shape[:2]

            # Eye Detection and Blink
            left_eye = [(face_landmarks.landmark[i].x * frame_width,
                         face_landmarks.landmark[i].y * frame_height) for i in range(33, 42)]
            right_eye = [(face_landmarks.landmark[i].x * frame_width,
                          face_landmarks.landmark[i].y * frame_height) for i in range(362, 373)]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= BLINK_CONSEC_FRAMES:
                    print("Blink detected!")
                    if challenge == "Blink":
                        challenge_success = True
                blink_counter = 0

            # Motion Detection
            motion_success = False
            if prev_face_x is not None and prev_face_y is not None:
                nose_x = face_landmarks.landmark[1].x * frame_width
                nose_y = face_landmarks.landmark[1].y * frame_height
                movement_x = nose_x - prev_face_x
                movement_y = nose_y - prev_face_y

                if challenge == "Turn Left" and movement_x < -20:
                    challenge_success = True
                elif challenge == "Turn Right" and movement_x > 20:
                    challenge_success = True
                elif challenge == "Tilt Left" and movement_y < -20:
                    challenge_success = True
                elif challenge == "Tilt Right" and movement_y > 20:
                    challenge_success = True
                elif challenge == "Nod Up" and movement_y < -10:
                    challenge_success = True
                elif challenge == "Nod Down" and movement_y > 10:
                    challenge_success = True

            prev_face_x = face_landmarks.landmark[1].x * frame_width
            prev_face_y = face_landmarks.landmark[1].y * frame_height

            # Texture Analysis
            texture_score = analyze_texture(gray)
            if texture_score > TEXTURE_THRESHOLD:
                print("Good texture detected!")

            # Challenge Timing and New Challenge
            current_time = time.time()
            if challenge is None or (current_time - last_challenge_time > CHALLENGE_INTERVAL):
                challenge = random.choice(CHALLENGES)
                last_challenge_time = current_time
                challenge_start_time = current_time
                print(f"New Challenge: {challenge}")

            # Challenge Timeout
            if current_time - challenge_start_time > CHALLENGE_TIMEOUT:
                print(f"Challenge Timeout: {challenge}")
                challenge = None

            # Track Success
            if challenge_success:
                print(f"Challenge {challenge} succeeded!")
                successful_challenges += 1
                completed_challenges += 1
                challenge = None
            else:
                completed_challenges += 1

            # Stability Check
            stable_liveness = successful_challenges >= LIVENESS_CONFIRMATION_THRESHOLD

    # Display Liveness Status and Challenges
    cv2.putText(frame, f"Liveness: {'Confirmed' if stable_liveness else 'Not Confirmed'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if stable_liveness else (0, 0, 255), 2)
    if challenge:
        cv2.putText(frame, f"Challenge: {challenge}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Liveness Detection", frame)

    # Stop if Liveness Confirmed
    if stable_liveness:
        print("Liveness Confirmed!")
        break

    if cv2.waitKey(1) == ord('q'):
        break

# Dataset Collection
face_embeddings = []
face_labels = []

print("Starting dataset collection...")
user_label = input("Enter your name: ")

while len(face_embeddings) < DATASET_SIZE:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Preprocess frame for ResNet
            face_resized = cv2.resize(frame, (224, 224))  # ResNet expects 224x224 input
            face_array = preprocess_input(np.expand_dims(face_resized, axis=0))

            # Extract embedding
            embedding = model.predict(face_array)[0]
            face_embeddings.append(embedding)
            face_labels.append(user_label)

            print(f"Dataset Collected: {len(face_embeddings)}/{DATASET_SIZE}")

    cv2.imshow("Dataset Collection", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save embeddings and labels
np.save("face_embeddings.npy", np.array(face_embeddings))
np.save("face_labels.npy", np.array(face_labels))
print("Dataset collection complete.")
