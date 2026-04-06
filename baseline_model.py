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
from sklearn.metrics import f1_score

import config as cfg


def train_baseline_rf(X_train, y_train, random_state=None):
    """
    Train a baseline Random Forest on raw flux features, wrapped in a
    sklearn Pipeline with StandardScaler.

    NOTE: class_weight is NOT set to 'balanced' because SMOTE has already
    balanced the classes. Using both would double-correct.

    NOTE: CalibratedClassifierCV REMOVED — sigmoid calibration was compressing
    all probabilities into ~0.001–0.07 range, making threshold search fail
    on the test set. Raw RF vote-proportion probabilities are more spread out
    and transfer better from validation to test.

    Parameters
    ----------
    X_train : np.ndarray
        Raw normalized flux values (post-SMOTE)
    y_train : np.ndarray
        Binary labels (post-SMOTE, already balanced)
    random_state : int

    Returns
    -------
    pipeline : Pipeline
        Trained pipeline (StandardScaler -> RF)
    """
    if random_state is None:
        random_state = cfg.GLOBAL_SEED

    print(f"[INFO] Training BASELINE Random Forest ({cfg.BASELINE_N_ESTIMATORS} trees) with Pipeline...")
    rf = RandomForestClassifier(
        n_estimators=cfg.BASELINE_N_ESTIMATORS,
        max_depth=cfg.BASELINE_MAX_DEPTH,
        min_samples_leaf=cfg.BASELINE_MIN_SAMPLES_LEAF,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    # FIX: Removed CalibratedClassifierCV — sigmoid calibration compressed
    # all probabilities into 0.001–0.07 range, causing zero positive predictions
    # on test set. Raw RF probabilities are wider and threshold-stable.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', rf)
    ])

    pipeline.fit(X_train, y_train)
    print("  Baseline pipeline training complete.")
    return pipeline


def find_optimal_threshold_baseline(pipeline, X_val, y_val):
    """
    Find the threshold that maximizes F1 for the baseline model.

    Uses BASELINE-SPECIFIC threshold config (decoupled from hybrid):
      - BASELINE_THRESHOLD_MIN = 0.01 (very low floor)
      - BASELINE_THRESHOLD_MAX = 0.50
      - BASELINE_THRESHOLD_STEPS = 50
      - BASELINE_THRESHOLD_FALLBACK = 0.01

    Includes a hard percentile fallback guarantee: if no threshold in the
    linspace search produces any positives (F1=0), forces the top-1% of
    probability samples to be predicted positive.
    """
    y_prob = pipeline.predict_proba(X_val)[:, 1]

    best_f1 = 0
    best_threshold = cfg.BASELINE_THRESHOLD_FALLBACK

    thresholds = np.linspace(
        cfg.BASELINE_THRESHOLD_MIN,
        cfg.BASELINE_THRESHOLD_MAX,
        cfg.BASELINE_THRESHOLD_STEPS
    )

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    # ── HARD GUARANTEE: if no threshold found any positives, force top-N ──
    if best_f1 == 0:
        print("[WARN] Linspace threshold search found F1=0 — activating percentile fallback")

        # Try top 1% of samples as positive
        percentile_threshold = np.percentile(y_prob, 99)
        percentile_threshold = max(percentile_threshold, cfg.BASELINE_THRESHOLD_FALLBACK)
        y_pred_forced = (y_prob >= percentile_threshold).astype(int)
        f1_forced = f1_score(y_val, y_pred_forced, zero_division=0)
        print(f"[FALLBACK] Top-1% threshold: {percentile_threshold:.6f} "
              f"(positives={y_pred_forced.sum()}, F1={f1_forced:.4f})")

        if f1_forced > 0:
            best_threshold = percentile_threshold
            best_f1 = f1_forced
        else:
            # Last resort: top 0.5%
            percentile_threshold = np.percentile(y_prob, 99.5)
            percentile_threshold = max(percentile_threshold, cfg.BASELINE_THRESHOLD_FALLBACK)
            y_pred_last = (y_prob >= percentile_threshold).astype(int)
            f1_last = f1_score(y_val, y_pred_last, zero_division=0)
            best_threshold = percentile_threshold
            best_f1 = f1_last
            print(f"[LAST RESORT] Top-0.5% threshold: {percentile_threshold:.6f} "
                  f"(positives={y_pred_last.sum()}, F1={f1_last:.4f})")

    if best_f1 == 0:
        print(f"[CRITICAL] Baseline still at F1=0 after all fallbacks — using threshold: {best_threshold:.6f}")
    else:
        print(f"[INFO] Optimal baseline threshold: {best_threshold:.6f} (F1={best_f1:.4f})")

    return best_threshold, best_f1


def save_baseline(model, path):
    """Save the baseline model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[INFO] Baseline model saved to: {path}")
