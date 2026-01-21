import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import os
import sys

# Add project root to path to access config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import CFG

# 1. Load Model and Labels
model_path = os.path.join(CFG.OUT_DIR, "best_ResNet50_FER_model.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")

print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

# 2. Load Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion(input_img):
    if input_img is None:
        return None, "Please upload an image."
    
    # Convert RGB (Gradio) to BGR (OpenCV)
    full_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, {"Error": "No face detected. Please use a clearer photo."}
    
    # Process the largest face
    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    roi = full_img[y:y+h, x:x+w]
    
    # Preprocess
    roi = cv2.resize(roi, CFG.IMG_SIZE)
    roi = roi.astype("float32") / 255.0
    roi = np.expand_dims(roi, axis=0)
    
    # Predict
    preds = model.predict(roi, verbose=0)[0]
    
    # Format labels for Gradio Label component
    results = {emotion_labels[i]: float(preds[i]) for i in range(len(emotion_labels))}
    
    # Draw rectangle on original image for feedback
    cv2.rectangle(input_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
    
    return input_img, results

# 3. Build Gradio Interface
with gr.Blocks(title="Facial Emotion Recognition") as demo:
    gr.Markdown("# 🎭 Facial Emotion Recognition")
    gr.Markdown("Upload an image to detect the facial emotion using the trained ResNet50V2 model.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image")
            submit_btn = gr.Button("Analyze Emotion", variant="primary")
        
        with gr.Column():
            image_output = gr.Image(label="Detected Face")
            label_output = gr.Label(label="Predictions", num_top_classes=3)

    submit_btn.click(
        fn=predict_emotion,
        inputs=image_input,
        outputs=[image_output, label_output]
    )
    
    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "sample_image.webp")],
        inputs=image_input
    )

if __name__ == "__main__":
    demo.launch(share=True)
