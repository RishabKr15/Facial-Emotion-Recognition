# 🎭 Facial Emotion Recognition (FER)

A robust, production-ready system for Facial Emotion Recognition using **Transfer Learning** with `ResNet50V2` and `MobileNetV2`. Designed for high performance even on consumer-grade hardware.

---

## 🚀 Key Features

*   **🧠 Intelligent Hardware Selection**: Automatically detects GPU availability. Uses `ResNet50V2` for high-performance training on GPUs and `MobileNetV2` for efficient execution on CPUs.
*   **♻️ Smart Resumability**: Never lose training progress. The system automatically reads logs and resumes exactly from the last successful epoch after any interruption or crash.
*   **📉 VRAM Optimization**: Fine-tuned for hardware with limited memory (e.g., 4GB NVIDIA cards). Uses adaptive batch sizes (16 for frozen training, 4 for fine-tuning) to prevent "Out of Memory" errors.
*   **⚡ Two-Stage Training Logic**:
    1.  **Stage 1**: Trains the custom head with a frozen backbone to preserve pretrained knowledge.
    2.  **Stage 2**: Unfreezes the backbone for deep fine-tuning with a very low learning rate.
*   **📊 Integrated Analytics**: Automatically generates Accuracy/Loss history plots, Confusion Matrices, and ROC curves for formal evaluation.

---

## 📂 Project Structure

The project follows a modular architecture for better maintainability:

```text
Facial_Emotion_Recognition/
├── apps/
│   ├── streamlit_app.py   # Streamlit web interface
│   ├── gradio_app.py      # Gradio testing interface
│   └── inference_webcam.py # Real-time webcam inference
├── config/
│   └── config.py          # Hyperparameters & Paths
├── src/
│   ├── data_loader.py     # Image augmentation pipeline
│   ├── model.py           # Model architecture
│   ├── trainer.py         # Training engine
│   ├── evaluate.py        # Metrics & Plots
│   └── utils.py           # GPU & logging utilities
├── saved_models/          # Model checkpoints
├── logs/                  # Training metrics
├── main.py                # Pipeline orchestrator
└── requirements.txt       # Dependencies
```

---

## 🛠️ Setup & Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/RishabKr15/Facial-Emotion-Recognition.git
    cd Facial-Emotion-Recognition
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data**:
    Ensure your dataset is organized into `train/` and `test/` folders in the root directory, with each emotion in its own subfolder:
    ```text
    train/
    ├── happy/
    ├── sad/
    └── ...
    ```

---

## 🏃 Usage

### Start / Resume Training
Run the orchestrator script. It will automatically detect if training was interrupted and resume from the last epoch.
```bash
python main.py
```

### Configuration
Adjust hyperparameters like `BATCH_SIZE`, `LEARNING_RATE`, or `EPOCHS` in `config/config.py`.

---

## 🖥️ Inference & Apps

The project includes multiple ways to run inference:

### 1. Streamlit Web App (Recommended)
A modern interface to upload images and see predictions.
```bash
streamlit run apps/streamlit_app.py
```

### 2. Gradio Interface
A simple web UI for quick testing.
```bash
python apps/gradio_app.py
```

### 3. Real-time Webcam Inference
Monitor emotions in real-time through your webcam.
```bash
python apps/inference_webcam.py
```

---

## 📈 Evaluation Results

After training, the system produces:
- **Best Model Weights**: Saved in `saved_models/best_model.keras`.
- **Training Plots**: Accuracy and Loss curves over epochs.
- **Confusion Matrix**: Detailed breakdown of model performance across all categories.

---

## 📜 License
MIT License.
