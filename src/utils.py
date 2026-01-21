import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

def setup_gpu():
    """
    Configures GPU memory growth and logging.
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("=" * 60)
    print("TensorFlow version:", tf.__version__)
    print("Checking GPU availability...")
    print("=" * 60)

    gpus = tf.config.list_physical_devices('GPU')
    print(f"Detected GPUs: {gpus}")

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            return True
        except Exception as e:
            print(f"GPU configuration error: {e}")
            print("Falling back to CPU")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            return False
    else:
        print("WARNING: No GPU detected!")
        print("\nRunning on CPU - training will be slower")
        return False

def get_last_epoch(log_path):
    """
    Reads the CSV log file to find the last completed epoch.
    Returns the next epoch index (0-based).
    """
    if not os.path.exists(log_path):
        return 0
    try:
        df = pd.read_csv(log_path)
        if df.empty:
            return 0
        # 'epoch' column in CSVLogger is 0-based index of the completed epoch
        return df['epoch'].max() + 1 
    except Exception as e:
        print(f"Error reading log file: {e}")
        return 0

def get_class_weights(train_gen):
    """
    Computes balanced class weights based on the training generator.
    """
    labels = np.array(train_gen.classes)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}
    print("Class weights:", class_weights_dict)
    return class_weights_dict
