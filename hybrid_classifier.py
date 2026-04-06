"""
hybrid_classifier.py — CNN Feature Extraction + Random Forest Classification

This module implements the core "hybrid" approach:
  1. Takes a trained CNN and extracts 64-dim feature vectors
  2. Trains a Random Forest (200 trees, max_depth=8) on SMOTE-balanced features
  3. Wraps in Pipeline with StandardScaler + CalibratedClassifierCV
  
NOTE: Ensemble blending REMOVED — it degraded probability calibration
and worsened PR-AUC by mixing incompatible probability distributions.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

import config as cfg


def extract_cnn_features(cnn_model, X):
    """
    Extract 64-dimensional feature embeddings from the CNN's feature_layer.
    """
    feature_layer = cnn_model.get_layer('feature_layer')
    feature_extractor = models.Model(
        inputs=cnn_model.inputs,
        outputs=feature_layer.output,
        name='feature_extractor'
    )

    print(f"[INFO] Extracting CNN features (Dense-{cfg.CNN_FEATURE_DIM})...")
    features = feature_extractor.predict(X, verbose=0)
    print(f"  Extracted feature shape: {features.shape}")
    return features


def train_random_forest(X_features, y, random_state=None):
    """
    Train a Random Forest on CNN-extracted features with probability calibration.

    class_weight='balanced' is used because SMOTE at 0.1 ratio only brings
    positives to ~429 vs ~4292 negatives (still 10:1 imbalance). Without
    balanced weights, the RF assigns low probability to positives.
    Uses 200 trees with max_depth=8 to prevent overfitting on 64 features.
    """
    if random_state is None:
        random_state = cfg.GLOBAL_SEED

    print(f"[INFO] Training Hybrid RF ({cfg.RF_N_ESTIMATORS} trees, "
          f"max_depth={cfg.RF_MAX_DEPTH}, min_leaf={cfg.RF_MIN_SAMPLES_LEAF})...")

    rf = RandomForestClassifier(
        n_estimators=cfg.RF_N_ESTIMATORS,
        max_depth=cfg.RF_MAX_DEPTH,
        min_samples_leaf=cfg.RF_MIN_SAMPLES_LEAF,
        min_samples_split=cfg.RF_MIN_SAMPLES_SPLIT,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    calibrated_rf = CalibratedClassifierCV(rf, cv=3, method='isotonic')

    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', calibrated_rf)
    ])

    final_pipeline.fit(X_features, y)
    print("  Hybrid pipeline trained (StandardScaler -> CalibratedClassifierCV(RF)).")
    return final_pipeline


def find_optimal_threshold_rf(pipeline, X_val, y_val):
    """
    Find the classification threshold that maximizes F1-score for the RF pipeline,
    with a precision constraint: prefer thresholds where Precision >= 0.4.

    Selection rule:
      1. Among thresholds where Precision >= 0.4, pick the one with highest F1
      2. If no threshold satisfies Precision >= 0.4, fallback to best F1

    Uses np.linspace(0.30, 0.60, 40) for stable threshold search.
    """
    from sklearn.metrics import precision_score

    y_prob = pipeline.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_threshold = cfg.THRESHOLD_FALLBACK
    best_f1_constrained = 0
    best_threshold_constrained = cfg.THRESHOLD_FALLBACK

    thresholds = np.linspace(cfg.THRESHOLD_MIN, cfg.THRESHOLD_MAX, cfg.THRESHOLD_STEPS)
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        prec = precision_score(y_val, y_pred_t, zero_division=0)

        # Track best F1 overall (fallback)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

        # Track best F1 among precision-satisfying thresholds
        if prec >= 0.4 and f1 > best_f1_constrained:
            best_f1_constrained = f1
            best_threshold_constrained = t

    # Use precision-constrained threshold if found, else fallback to best F1
    if best_f1_constrained > 0:
        best_threshold = best_threshold_constrained
        best_f1 = best_f1_constrained
        print(f"[INFO] Optimal RF threshold (precision-aware): {best_threshold:.2f} (F1={best_f1:.4f})")
    elif best_f1 > 0:
        print(f"[INFO] Optimal RF threshold (F1-only fallback): {best_threshold:.2f} (F1={best_f1:.4f})")
    else:
        best_threshold = cfg.THRESHOLD_FALLBACK
        print(f"[WARN] No threshold improvement found — using fallback: {best_threshold:.2f}")

    return best_threshold, best_f1


def save_rf(model, path):
    """Save the trained pipeline to disk using pickle."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Hybrid pipeline saved to: {path}")


def load_rf(path):
    """Load a saved pipeline from disk."""
    with open(path, 'rb') as f:
        model = pickle.load(f)
    print(f"[INFO] Hybrid pipeline loaded from: {path}")
    return model
