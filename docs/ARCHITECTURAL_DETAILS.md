# Project Structure and Architecture Explanation

This document explains the refactoring of the Facial Emotion Recognition project from a monolithic script into a modular, production-ready architecture.

## 1. New Folder Structure

Instead of everything being in the root folder, files are organized by their responsibility:

*   **`config/`**: Stores settings and configuration.
*   **`src/`**: Stores the actual source code logic (Data, Model, Training, Evaluation).
*   **`saved_models/`**: Dedicated directory for `.keras` model files (keeps the root clean).
*   **`logs/`**: Dedicated directory for CSV training logs.
*   **`notebooks/`**: Contains Jupyter notebooks, separating experimentation from production code.

## 2. File-by-File Explanation

### A. `config/config.py` (The Control Center)
*   **What it does:** Holds all settings (Batch Size, Learning Rate, Epochs, Paths).
*   **Why:** Allows for easy adjustment of hyperparameters without modifying the core logic.
*   **Key Logic:** Defines the 4GB VRAM optimization (Batch Size 16 for frozen, 4 for fine-tuning).

### B. `src/model.py` (The Brain)
*   **What it does:** Defines the neural network architecture.
*   **Why:** Keeps the model definition separate, facilitating future architecture swaps (e.g., to EfficientNet).
*   **Key Logic:** Automatically switches between `ResNet50V2` (if GPU is detected) and `MobileNetV2` (if CPU is used).

### C. `src/data_loader.py` (The Feeder)
*   **What it does:** Handles image augmentation and creates the data generators.
*   **Why:** Isolates complex data preprocessing logic, making it easier to adjust augmentations without breaking the training loop.

### D. `src/trainer.py` (The Engine)
*   **What it does:** Contains the core training loop.
*   **Key Logic:**
    *   **Resumption:** Checks `logs/training.log` to automatically resume from the last successful epoch if a crash occurs.
    *   **Two-Stage Training:** Manages the "Frozen Base" (Stage 1) and "Fine-Tuning" (Stage 2) phases.
    *   **Callbacks:** Sets up ModelCheckpoint, EarlyStopping, and CSVLogger.

### E. `src/evaluate.py` (The Analyst)
*   **What it does:** Handles all graphs and metrics.
*   **Why:** Separates verbose plotting code (Matplotlib/Seaborn) from the main logic. Generates Accuracy/Loss plots, Confusion Matrices, and ROC Curves.

### F. `src/utils.py` (The Helpers)
*   **What it does:** Contains helper functions like `setup_gpu()` (to enable memory growth) and `get_last_epoch()` (to read logs).

### G. `main.py` (The Orchestrator)
*   **What it does:** A clean entry point script.
*   **How it works:** Imports tools from `src/` and executes the pipeline:
    1.  `setup_gpu()`
    2.  `train_model()`
    3.  `plot_history()`
    4.  `evaluate_model()`

## 3. Summary of Benefits

1.  **Resumability:** If training is interrupted (e.g., at Epoch 45), running `python main.py` again resumes exactly from that point.
2.  **Cleanliness:** Output files (models, logs) are organized into specific folders, keeping the workspace tidy.
3.  **Maintainability:** Specific behaviors are isolated (e.g., settings in `config.py`, architecture in `model.py`), making the codebase easier to manage and scale.
