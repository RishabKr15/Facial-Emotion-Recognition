import cv2
import numpy as np
import os
import sys

# Add project root to path to access config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import tensorflow as tf
from config.config import CFG
from src.utils import setup_gpu
from collections import deque
from collections import Counter

# 0. Setup Hardware
# Enable memory growth to prevent OOM
setup_gpu()

# 1. Load the trained model
model_path = os.path.join(CFG.OUT_DIR, "best_ResNet50_FER_model.keras")
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    print("Please run main.py first to train and save the model.")
    exit()

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

# 2. Define Emotion Labels
# These must match the folder names in your 'train' directory, sorted alphabetically
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 3. Load Face Cascade for detection
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    print(f"Error: Could not load face cascade from {face_cascade_path}")
    exit()

# 4. Smoothing Buffer
# Keep track of the last N predictions to stabilize the output
prediction_buffer = deque(maxlen=10)

# 4. Initialize Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started. Press 'q' to quit.")

while True:
    # 5. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 6. Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 7. Detect faces (Using more conservative parameters to reduce multiple detections)
    # Increase minNeighbors to ensure a more solid detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # 8. Preprocess detected face for prediction
        # Extract face ROI
        roi_color = frame[y:y + h, x:x + w]
        
        # Resize to match training image size
        roi_color = cv2.resize(roi_color, CFG.IMG_SIZE)
        
        # Normalize/Rescale (training used 1./255)
        roi_color = roi_color.astype("float32") / 255.0
        
        # Expand dimensions to match model input shape (batch_size, height, width, channels)
        roi_color = np.expand_dims(roi_color, axis=0)

        # 9. Predict Emotion
        # Use direct model call instead of .predict() for lower latency
        preds = model(roi_color, training=False)
        preds_numpy = preds.numpy()
        max_idx = np.argmax(preds_numpy)
        label = emotion_labels[max_idx]
        confidence = preds_numpy[0][max_idx]

        # Add to buffer for smoothing
        prediction_buffer.append(label)
        
        # Get the most frequent label in the buffer (Smoothing)
        smoothed_label = Counter(prediction_buffer).most_common(1)[0][0]

        # 10. Overlay label on image (Displaying smoothed label)
        label_text = f"{smoothed_label} ({confidence*100:.1f}%)"
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 11. Display the resulting frame
    cv2.imshow('Facial Emotion Recognition', frame)

    # 12. Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 13. Cleanup
cap.release()
cv2.destroyAllWindows()
