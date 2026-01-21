import os
import tensorflow as tf

# ==============================================================
# Hardware Detection
# ==============================================================
try:
    gpus = tf.config.list_physical_devices('GPU')
    IS_GPU = len(gpus) > 0
    print(f"Config detected GPU: {IS_GPU}")
except Exception:
    IS_GPU = False
    print("Config could not detect GPU, defaulting to CPU settings.")

# ==============================================================
# Configuration
# ==============================================================
class CFG:
    # Paths
    # Assuming the project root is two levels up from this file (config/config.py)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    TRAIN_DIR   = os.path.join(BASE_DIR, 'train')
    TEST_DIR    = os.path.join(BASE_DIR, 'test')
    OUT_DIR     = os.path.join(BASE_DIR, 'saved_models')
    LOG_DIR     = os.path.join(BASE_DIR, 'logs')
    
    # Ensure directories exist
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Data Params
    IMG_SIZE    = (224, 224)
    
    # Training Params
    # Stage 1: Frozen Base - Use highest stable batch size
    BATCH_SIZE  = 16 if IS_GPU else 16 
    # Stage 2: Fine-Tuning - CRITICAL: Very small batch size for unfrozen ResNet50V2 on 4GB VRAM
    FINE_TUNE_BATCH_SIZE = 4 if IS_GPU else 4 
    
    EPOCHS      = 100 # Total number of epochs
    STAGE1_EPOCHS = 20 # Epochs for frozen base training
    
    LR          = 1e-4 # LR for initial training
    FINE_TUNE_LR= 1e-5 # LR for fine-tuning all layers
    SEED        = 42
