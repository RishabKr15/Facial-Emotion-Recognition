import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import sys
from PIL import Image
import time

# Add project root to path to access config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import CFG

# 1. Page Configuration & Styling
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Title Styling */
    .title-text {
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(to right, #60a5fa, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    
    /* Result Card Glassmorphism */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-top: 1rem;
    }
    
    .emotion-badge {
        background: rgba(96, 165, 250, 0.2);
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 99px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-right: 8px;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Progress Bar Customization */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #60a5fa, #a855f7);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.6s ease-out forwards;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Header
st.markdown("""
<div class="fade-in">
    <h1 style="font-weight: 800; color: #60a5fa; margin-bottom: 0;">🎭 Facial Emotion Recognition</h1>
    <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 0; margin-bottom: 2rem;">
        Unveiling the spectrum of human feelings using deep learning.
    </p>
</div>
""", unsafe_allow_html=True)

# 3. Load Model and Labels (Cached)
@st.cache_resource
def load_model():
    model_path = os.path.join(CFG.OUT_DIR, "best_ResNet50_FER_model.keras")
    if not os.path.exists(model_path):
        return None
    try:
        return tf.keras.models.load_model(model_path)
    except:
        return None

model = load_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 4. Sidebar / Settings
with st.sidebar:
    st.markdown('<h1 style="text-align: center; font-size: 5rem; margin-bottom: 0;">🎭</h1>', unsafe_allow_html=True)
    st.title("Settings")
    confidence_threshold = st.slider("Min Confidence", 0.0, 1.0, 0.2, 0.05)
    
    st.markdown("---")
    st.subheader("Model Info")
    st.info("**Architecture**: ResNet50V2\n\n**Dataset**: FER-2013\n\n**Status**: Operational ✅")
    
    st.markdown("---")
    st.markdown("### Powered by [Facial Emotion Recognition](https://github.com/RishabKr15/Facial-Emotion-Recognition)")

# 5. Main Content Layout
col_main, col_gallery = st.columns([3, 1])

with col_main:
    uploaded_file = st.file_uploader("Drop your image here", type=["jpg", "jpeg", "png", "webp"])
    
    # Sample selection logic
    selected_sample = None
    with col_gallery:
        st.markdown("### 🖼️ Try Samples")
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_image.webp")
        if os.path.exists(sample_path):
            st.image(sample_path, width=150, caption="Sample Image")
            if st.button("Analyze This Sample"):
                selected_sample = sample_path
        else:
            st.caption("No sample image found.")

    # Process either upload or sample
    input_source = uploaded_file or selected_sample

    if input_source:
        image = Image.open(input_source).convert('RGB')
        input_img = np.array(image)
        
        container = st.container()
        c1, c2 = container.columns([1.2, 1])
        
        with c1:
            st.markdown('<div class="fade-in">', unsafe_allow_html=True)
            st.image(image, use_column_width=True, caption="Target Input")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with c2:
            if model is None:
                st.error("Model engine not found. Please ensure training is complete.")
            else:
                with st.spinner("System is analyzing gaze and posture..."):
                    # Process largest face
                    full_img = input_img.copy()
                    gray = cv2.cvtColor(full_img, cv2.COLOR_RGB2GRAY)
                    
                    # More robust detection parameters
                    faces = face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1, # More granular search
                        minNeighbors=3,  # Higher sensitivity
                        minSize=(48, 48) # Minimum face size
                    )
                    
                    if len(faces) == 0:
                        st.warning("⚠️ No face detected sharply. Attempting full-image analysis...")
                        roi = full_img
                    else:
                        (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
                        # Add slight padding to face box
                        pad = int(0.1 * w)
                        y1, y2 = max(0, y-pad), min(full_img.shape[0], y+h+pad)
                        x1, x2 = max(0, x-pad), min(full_img.shape[1], x+w+pad)
                        roi = full_img[y1:y2, x1:x2]
                    
                    # Preprocess ROI
                    roi = cv2.resize(roi, CFG.IMG_SIZE)
                    roi = roi.astype("float32") / 255.0
                    roi = np.expand_dims(roi, axis=0)
                    
                    # Prediction
                    preds = model.predict(roi, verbose=0)[0]
                    
                    st.markdown('<div class="result-card fade-in">', unsafe_allow_html=True)
                    st.subheader("Analysis Results")
                    
                    # Mapping emotions to emojis
                    emotion_emojis = {
                        'angry': '😠', 'disgust': '🤢', 'fear': '😨',
                        'happy': '😊', 'neutral': '😐', 'sad': '😢', 'surprise': '😲'
                    }
                    
                    top_idx = np.argmax(preds)
                    dominant_emotion = emotion_labels[top_idx]
                    dominant_emoji = emotion_emojis.get(dominant_emotion, '🤔')
                    
                    st.markdown(f"### Dominant: {dominant_emoji} **{dominant_emotion.upper()}**")
                    st.write("---")
                    
                    results = {emotion_labels[i]: float(preds[i]) for i in range(len(emotion_labels))}
                    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
                    
                    for emotion, score in sorted_results:
                        if score >= confidence_threshold:
                            emoji = emotion_emojis.get(emotion, '•')
                            st.write(f"**{emoji} {emotion.capitalize()}**")
                            st.progress(score)
                    
                    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>Made with ❤️ for Emotion Recognition Research</p>", unsafe_allow_html=True)
