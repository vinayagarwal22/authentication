# Liveness Detection and Face Authentication System

This project implements a real-time **Liveness Detection** and **Face Authentication System** using a combination of computer vision, deep learning, and pre-trained models. The system ensures secure user verification by challenging users with random tasks while also authenticating their identity against a dataset of known embeddings.

---

## Features

1. **Liveness Detection:**
   - Detects and verifies real users by analyzing their responses to challenges such as:
     - Blinking
     - Nodding (Up/Down)
     - Turning head (Left/Right)
     - Tilting head (Left/Right)
     - Shaking head
   - Uses Mediapipe's face mesh landmarks for accurate motion detection.
   - Analyzes texture to distinguish between real faces and spoofed images/videos.

2. **Face Authentication:**
   - Utilizes a pre-trained ResNet50 model to extract facial embeddings.
   - Compares embeddings to a pre-saved dataset for user authentication.
   - Ensures secure verification by matching embeddings within a threshold distance.

3. **Dataset Collection:**
   - Collects facial embeddings from real users and stores them for future authentication.
   - Uses Mediapipe to detect and crop face regions and ResNet50 to generate embeddings.

4. **Challenge-Response Validation:**
   - Randomly assigns challenges to users and validates their responses in real time.
   - Challenges have a timeout mechanism to prevent delays or spoof attempts.

---

## Workflow

1. **Dataset Collection:**
   - Run `detection.py` to capture facial embeddings and store them along with user labels.

2. **Liveness Detection and Authentication:**
   - Run `authentication.py` to perform real-time liveness detection and authentication.
   - The system confirms liveness before performing authentication against the stored dataset.

3. **User Interaction:**
   - Users are prompted with random challenges that must be successfully completed to confirm liveness.
   - Upon confirming liveness, the system authenticates the user by matching their face embedding.

---

## Technologies Used

- **Python Libraries:**
  - OpenCV
  - Mediapipe
  - NumPy
  - TensorFlow/Keras
  - SciPy
- **Pre-trained Model:**
  - ResNet50 from TensorFlow's Keras Applications

---

## Prerequisites

- Python 3.7 or above
- Installed dependencies from `requirements.txt`
- Camera (for real-time video capture)

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
