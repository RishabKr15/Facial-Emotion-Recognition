import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tqdm import tqdm

def plot_history(h):
    if not hasattr(h, 'history') or not h.history:
        print("No training history to plot.")
        return
    try:
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(h.history['accuracy'], label='Train')
        plt.plot(h.history['val_accuracy'], label='Val')
        plt.title('Accuracy'); plt.legend()
        plt.subplot(1,2,2)
        plt.plot(h.history['loss'], label='Train')
        plt.plot(h.history['val_loss'], label='Val')
        plt.title('Loss'); plt.legend()
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"Could not plot history: {e}")

def evaluate_model(model, val_gen, steps_val):
    print("\n" + "="*20 + " Starting Evaluation " + "="*20)
    
    # Basic Evaluation
    val_loss, val_acc = model.evaluate(val_gen, steps=steps_val, verbose=0)
    print(f"Val acc: {val_acc:.4f} | Val loss: {val_loss:.4f}")

    # Predictions
    val_gen.reset() 
    y_true = val_gen.classes
    y_pred_prob = model.predict(val_gen, steps=steps_val, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    labels = list(val_gen.class_indices.keys())

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9,7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix'); plt.show()

    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    # ROC Curves
    y_true_onehot = pd.get_dummies(y_true).values
    plt.figure(figsize=(10,6))
    for i in tqdm(range(len(labels)), desc="ROC classes"):
        fpr, tpr, _ = roc_curve(y_true_onehot[:,i], y_pred_prob[:,i])
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{labels[i]} (AUC={auc_score:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0,1]); plt.ylim([0,1.05])
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC Curves'); plt.legend(); plt.show()
