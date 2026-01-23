import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report

CLASS_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

def evaluate_model(model, test_gen):
    """
    Evaluates the model on the test generator.
    Prints ROC-AUC per class and classification report.
    """
    print("\n=== Evaluating on Test Set ===")
    y_true, y_pred = [], []

    for X_batch, y_batch in test_gen:
        y_true.append(y_batch)
        preds = model.predict(X_batch, verbose=0)
        y_pred.append(preds)

    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    print(f"y_true shape: {y_true.shape}")
    print(f"y_pred shape: {y_pred.shape}")

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    try:
        aucs = roc_auc_score(y_true, y_pred, average=None)
        for name, auc in zip(CLASS_NAMES, aucs):
            print(f"{name}: AUC = {auc:.4f}")
    except ValueError as e:
        print("AUC Error:", e)

    print("\nClassification Report (Threshold = 0.5):")
    report = classification_report(
        y_true,
        (y_pred > 0.5).astype(int),
        target_names=CLASS_NAMES,
        zero_division=0
    )
    print(report)
