"""
baseline_model.py — Baseline Random Forest on Raw Flux Features

This module trains a traditional Random Forest directly on the raw flux values
(after normalization + SMOTE), WITHOUT any CNN feature extraction.

This serves as the comparison baseline to demonstrate that the hybrid
CNN + RF approach outperforms classical ML on raw features.
"""

import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, precision_recall_curve, auc, f1_score


def train_baseline_rf(X_train, y_train, n_estimators=200, random_state=42):
    """
    Train a baseline Random Forest on raw flux features, wrapped in a
    sklearn Pipeline with StandardScaler and probability calibration.

    Parameters
    ----------
    X_train : np.ndarray
        Raw normalized flux values
    y_train : np.ndarray
        Binary labels (after SMOTE balancing)
    n_estimators : int
    random_state : int

    Returns
    -------
    pipeline : Pipeline
        Trained pipeline (StandardScaler -> CalibratedClassifierCV(RF))
    """
    print(f"[INFO] Training BASELINE Random Forest ({n_estimators} trees) with Pipeline...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight='balanced',
        max_depth=30,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', calibrated_rf)
    ])

    pipeline.fit(X_train, y_train)
    print("  Baseline pipeline training complete.")
    return pipeline


def find_optimal_threshold_baseline(pipeline, X_val, y_val):
    """
    Find the threshold that maximizes F1 for the baseline model.
    """
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.10, 0.91, 0.01):
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    if best_f1 == 0:
        best_threshold = 0.2
        print(f"[WARN] No baseline threshold improvement â€” using fallback: {best_threshold:.2f}")
    else:
        print(f"[INFO] Optimal baseline threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold, best_f1


def evaluate_baseline(model, X_test, y_test):
    """
    Evaluate the baseline model and return metrics.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=['Non-Planet', 'Exoplanet'])

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    report_dict = classification_report(y_test, y_pred, target_names=['Non-Planet', 'Exoplanet'],
                                        output_dict=True)

    metrics = {
        'report': report,
        'precision': report_dict['Exoplanet']['precision'],
        'recall': report_dict['Exoplanet']['recall'],
        'f1': report_dict['Exoplanet']['f1-score'],
        'pr_auc': pr_auc,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

    print("\n[BASELINE] Classification Report:")
    print(report)
    print(f"[BASELINE] Precision-Recall AUC: {pr_auc:.4f}")

    return metrics


def save_baseline(model, path):
    """Save the baseline model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Baseline model saved to: {path}")
