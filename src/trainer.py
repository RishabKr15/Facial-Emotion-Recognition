import os
import math
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger

from config.config import CFG
from src.utils import get_last_epoch, get_class_weights
from src.data_loader import make_generators
from src.model import build_model

try:
    from tqdm.keras import TqdmCallback
    HAS_TQDM_KERAS = True
except ImportError:
    HAS_TQDM_KERAS = False

def get_callbacks(chk_path, chk_path_last, log_path):
    cb = [
        ModelCheckpoint(filepath=chk_path, save_best_only=True,
                        monitor='val_loss', mode='min', verbose=1),
        ModelCheckpoint(filepath=chk_path_last, save_best_only=False, verbose=0), # Save every epoch for resumption
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1), 
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        CSVLogger(log_path, append=True)
    ]
    if HAS_TQDM_KERAS:
        cb.append(TqdmCallback(verbose=1))
    return cb

def train_model():
    # Paths
    chk_path = os.path.join(CFG.OUT_DIR, 'best_model.keras')
    chk_path_last = os.path.join(CFG.OUT_DIR, 'last_model.keras')
    log_path = os.path.join(CFG.LOG_DIR, 'training.log')

    # Callbacks
    cb = get_callbacks(chk_path, chk_path_last, log_path)

    # Initial Generators
    print(f"Set initial BATCH_SIZE (S1) to: {CFG.BATCH_SIZE}")
    train_gen, val_gen = make_generators(CFG.BATCH_SIZE)
    
    # Class Weights
    class_weights_dict = get_class_weights(train_gen)
    
    # Build Model
    model = build_model(num_classes=train_gen.num_classes, initial_freeze=True)
    model.summary()

    # Resumption Logic
    initial_epoch = get_last_epoch(log_path)
    print(f"Detected last epoch from log: {initial_epoch}")

    history = None

    # --- Stage 1: Feature Extraction (Frozen Base) ---
    if initial_epoch < CFG.STAGE1_EPOCHS:
        print("\n" + "="*20 + " Starting/Resuming Initial Feature Extraction (Frozen Base) " + "="*20)
        
        if initial_epoch > 0:
            if os.path.exists(chk_path_last):
                print(f"Resuming Stage 1 from last checkpoint: {chk_path_last}")
                model.load_weights(chk_path_last)
            elif os.path.exists(chk_path):
                print(f"Resuming Stage 1 from best checkpoint: {chk_path}")
                model.load_weights(chk_path)

        steps_train_s1 = math.ceil(train_gen.samples / CFG.BATCH_SIZE)
        steps_val_s1   = math.ceil(val_gen.samples   / CFG.BATCH_SIZE)

        train_gen.reset()
        val_gen.reset()

        # Determine verbosity: If using TQDM, set fit verbose to 0 to avoid double bars
        fit_verbose = 0 if HAS_TQDM_KERAS else 1

        history = model.fit(
            train_gen,
            steps_per_epoch=steps_train_s1,
            epochs=CFG.STAGE1_EPOCHS, 
            validation_data=val_gen,
            validation_steps=steps_val_s1,
            class_weight=class_weights_dict,
            callbacks=cb,
            initial_epoch=initial_epoch,
            verbose=fit_verbose
        )
    else:
        print(f"Skipping Stage 1 (Completed {initial_epoch} epochs, Target {CFG.STAGE1_EPOCHS})")
        # Create a dummy history object if needed, or just handle None later
        class History:
            history = {}
        history = History()

    # --- Stage 2: Fine-Tuning Stage (Unfreeze all layers) ---
    
    # Update initial_epoch
    initial_epoch = get_last_epoch(log_path)

    if initial_epoch < CFG.EPOCHS:
        print("\n" + "="*20 + " Starting/Resuming Fine-Tuning Stage (Unfrozen Base) " + "="*20)

        # Re-build model unfrozen
        model = build_model(num_classes=train_gen.num_classes, initial_freeze=False)

        # Load weights
        if initial_epoch == CFG.STAGE1_EPOCHS:
            print(f"Loading best weights from Stage 1 for Fine-Tuning: {chk_path}")
            model.load_weights(chk_path)
        elif initial_epoch > CFG.STAGE1_EPOCHS:
            if os.path.exists(chk_path_last):
                print(f"Resuming Stage 2 from last checkpoint: {chk_path_last}")
                model.load_weights(chk_path_last)
            else:
                print(f"Last checkpoint not found, falling back to best: {chk_path}")
                model.load_weights(chk_path)
        else:
            print(f"Warning: Unexpected epoch {initial_epoch}. Loading best weights.")
            model.load_weights(chk_path)

        # Recreate Generators with fine-tune batch size
        print(f"Fine-Tuning Stage: Recreating Data Generators with BATCH_SIZE={CFG.FINE_TUNE_BATCH_SIZE}")
        train_gen, val_gen = make_generators(CFG.FINE_TUNE_BATCH_SIZE) 

        steps_train_s2 = math.ceil(train_gen.samples / CFG.FINE_TUNE_BATCH_SIZE)
        steps_val_s2   = math.ceil(val_gen.samples / CFG.FINE_TUNE_BATCH_SIZE)

        train_gen.reset()
        val_gen.reset()

        # Determine verbosity: If using TQDM, set fit verbose to 0 to avoid double bars
        fit_verbose = 0 if HAS_TQDM_KERAS else 1

        history_ft = model.fit(
            train_gen,
            steps_per_epoch=steps_train_s2,
            epochs=CFG.EPOCHS, 
            validation_data=val_gen,
            validation_steps=steps_val_s2,
            class_weight=class_weights_dict,
            callbacks=cb,
            initial_epoch=initial_epoch, 
            verbose=fit_verbose
        )

        # Merge history
        if hasattr(history, 'history') and history.history:
            for key in history_ft.history.keys():
                if key in history.history:
                    history.history[key].extend(history_ft.history[key])
        else:
            history = history_ft

    else:
        print("Training already completed!")
        # Ensure we have the fine-tune model structure for evaluation
        model = build_model(num_classes=train_gen.num_classes, initial_freeze=False)
        # Ensure generators are set to fine-tune batch size
        train_gen, val_gen = make_generators(CFG.FINE_TUNE_BATCH_SIZE) 

    return model, history, train_gen, val_gen
