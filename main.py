import os
import math
from config.config import CFG
from src.utils import setup_gpu
from src.trainer import train_model
from src.evaluate import plot_history, evaluate_model

def main():
    # 1. Setup
    setup_gpu()
    
    # 2. Train
    model, history, train_gen, val_gen = train_model()
    
    # 3. Plot History
    plot_history(history)
    
    # 4. Evaluate
    # Load best weights
    chk_path = os.path.join(CFG.OUT_DIR, 'best_model.keras')
    if os.path.exists(chk_path):
        print(f"Loading absolute best weights from {chk_path} before final evaluation.")
        model.load_weights(chk_path)
    else:
        print("No best weights file found. Using current model weights.")

    # Calculate steps for evaluation (using fine-tune batch size)
    steps_val = math.ceil(val_gen.samples / CFG.FINE_TUNE_BATCH_SIZE)
    
    evaluate_model(model, val_gen, steps_val)
    
    # 5. Save Final Model
    final_path = os.path.join(CFG.OUT_DIR, "best_ResNet50_FER_model.keras") 
    model.save(final_path)
    print(f"Final model saved → {final_path}")

if __name__ == "__main__":
    main()
