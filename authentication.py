import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as spatial_distance
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

# Load Stored Embeddings and Labels
known_embeddings = np.load("face_embeddings.npy")
known_labels = np.load("face_labels.npy", allow_pickle=True)

# Parameters
EYE_AR_THRESH = 0.25
BLINK_CONSEC_FRAMES = 2
MOTION_LEVEL_THRESHOLD = 50
TEXTURE_THRESHOLD = 50
CHALLENGE_INTERVAL = 7
CHALLENGE_TIMEOUT = 10
LIVENESS_CONFIRMATION_THRESHOLD = 3
AUTHENTICATION_THRESHOLD = 45.0  # Adjusted based on observed distances
CHALLENGES = [
    "Blink", "Nod Up", "Nod Down", "Turn Left", "Turn Right",
    "Tilt Left", "Tilt Right", "Shake Head"
]

# State Variables
blink_counter = 0
last_challenge_time = 0
challenge = None
challenge_start_time = None
successful_challenges = 0
stable_liveness = False
authenticated_person = None
prev_face_x = None
prev_face_y = None
authentication_attempted = False

# Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
    A = spatial_distance.euclidean(eye[1], eye[5])
    B = spatial_distance.euclidean(eye[2], eye[4])
    C = spatial_distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Analyze Texture using Laplacian Variance
def analyze_texture(gray_frame):
    laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return laplacian_var

# Compare Embeddings
def match_embedding(embedding, known_embeddings, known_labels, threshold):
    distances = [np.linalg.norm(embedding - known) for known in known_embeddings]
    print(f"Distances: {distances}")  # Debugging distances
    min_distance = min(distances)
    if min_distance < threshold:
        return known_labels[np.argmin(distances)], min_distance
    return None, min_distance

# Crop and preprocess face for ResNet50
def preprocess_face(frame, landmarks, frame_width, frame_height):
    # Get bounding box coordinates for face
    x_min = min(landmark.x for landmark in landmarks)
    y_min = min(landmark.y for landmark in landmarks)
    x_max = max(landmark.x for landmark in landmarks)
    y_max = max(landmark.y for landmark in landmarks)

    x_min = int(x_min * frame_width)
    y_min = int(y_min * frame_height)
    x_max = int(x_max * frame_width)
    y_max = int(y_max * frame_height)

    face_crop = frame[y_min:y_max, x_min:x_max]
    face_resized = cv2.resize(face_crop, (224, 224))
    face_array = preprocess_input(np.expand_dims(face_resized, axis=0))
    return face_array

# Initialize Video Capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Frame capture failed.")
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            frame_height, frame_width = frame.shape[:2]

            # Extract eye landmarks
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
                        successful_challenges += 1
                        challenge = None
                blink_counter = 0

            # Head Movement Detection (Challenges)
            nose_x = face_landmarks.landmark[1].x * frame_width
            nose_y = face_landmarks.landmark[1].y * frame_height

            if prev_face_x is not None and prev_face_y is not None:
                movement_x = nose_x - prev_face_x
                movement_y = nose_y - prev_face_y

                if challenge == "Turn Left" and movement_x < -20:
                    successful_challenges += 1
                    challenge = None
                elif challenge == "Turn Right" and movement_x > 20:
                    successful_challenges += 1
                    challenge = None
                elif challenge == "Nod Up" and movement_y < -10:
                    successful_challenges += 1
                    challenge = None
                elif challenge == "Nod Down" and movement_y > 10:
                    successful_challenges += 1
                    challenge = None

            prev_face_x = nose_x
            prev_face_y = nose_y

            # Challenge Timing Logic
            current_time = time.time()
            if not stable_liveness and (challenge is None or (current_time - last_challenge_time > CHALLENGE_INTERVAL)):
                challenge = random.choice(CHALLENGES)
                last_challenge_time = current_time
                challenge_start_time = current_time
                print(f"New Challenge: {challenge}")

            # Handle Challenge Timeout
            if challenge_start_time is not None and current_time - challenge_start_time > CHALLENGE_TIMEOUT:
                print(f"Challenge Timeout: {challenge}")
                challenge = None
                challenge_start_time = None

            # Check for Liveness Confirmation
            stable_liveness = successful_challenges >= LIVENESS_CONFIRMATION_THRESHOLD

            # Perform Authentication if Liveness Confirmed
            if stable_liveness and not authentication_attempted:
                face_array = preprocess_face(frame, face_landmarks.landmark, frame_width, frame_height)
                embedding = model.predict(face_array)[0]
                authenticated_person, dist = match_embedding(embedding, known_embeddings, known_labels, AUTHENTICATION_THRESHOLD)
                if authenticated_person:
                    print(f"Authenticated: {authenticated_person} with distance {dist}")
                else:
                    print(f"No match found. Closest distance: {dist}")
                authentication_attempted = True

    # Display Liveness and Authentication Status
    cv2.putText(frame, f"Liveness: {'Confirmed' if stable_liveness else 'Not Confirmed'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if stable_liveness else (0, 0, 255), 2)
    if authenticated_person:
        cv2.putText(frame, f"Authenticated: {authenticated_person}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif challenge:
        cv2.putText(frame, f"Challenge: {challenge}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Authentication", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
