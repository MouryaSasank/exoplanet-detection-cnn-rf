"""
cnn_feature_extractor.py — 1D CNN Architecture and Training

Reduced architecture optimized for the small Kepler dataset (37 positives):
  Input (n_flux, 1)
  -> Conv1D(32, 7, L2) -> BN -> MaxPool(2)
  -> Conv1D(64, 5, L2) -> BN -> MaxPool(2)
  -> Conv1D(128, 3, L2) -> BN -> MaxPool(2)
  -> GlobalAveragePooling1D -> Dense(64, ReLU) [feature layer]
  -> Dropout(0.6) -> Dense(1, Sigmoid)

Key design decisions:
  - Small filter counts (32→64→128) prevent overfitting on 37 samples
  - 64-dim feature layer gives RF enough signal without noise
  - L2 regularization on conv layers constrains weight magnitudes
  - Focal Loss handles class imbalance better than BCE
  - Trains on REAL data only (no SMOTE) — class weights handle imbalance
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import f1_score

import config as cfg


def focal_loss(gamma=None, alpha=None):
    """
    Focal Loss for handling class imbalance.

    Down-weights well-classified examples and focuses training on hard,
    misclassified examples. Critical for our dataset where the model
    can achieve 99% accuracy by predicting all non-planets.
    """
    if gamma is None:
        gamma = cfg.FOCAL_GAMMA
    if alpha is None:
        alpha = cfg.FOCAL_ALPHA

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, gamma)
        ce = -(y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(alpha_t * focal_weight * ce)

    loss_fn.__name__ = 'focal_loss'
    return loss_fn


def build_cnn(input_shape):
    """
    Build a compact 1D CNN for exoplanet detection.

    Architecture is intentionally small (32→64→128) because the dataset
    has only 37 positive samples — a large network would memorize them.
    L2 regularization further prevents overfitting.
    """
    l2 = regularizers.l2(cfg.L2_REG)
    inputs = layers.Input(shape=input_shape, name='input')

    # --- Block 1: Large kernel captures broad transit dip shapes ---
    x = layers.Conv1D(32, kernel_size=7, activation='relu',
                      padding='same', kernel_regularizer=l2, name='conv1d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)

    # --- Block 2: Medium kernel refines transit features ---
    x = layers.Conv1D(64, kernel_size=5, activation='relu',
                      padding='same', kernel_regularizer=l2, name='conv1d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_2')(x)

    # --- Block 3: Small kernel captures fine-grained patterns ---
    x = layers.Conv1D(128, kernel_size=3, activation='relu',
                      padding='same', kernel_regularizer=l2, name='conv1d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_3')(x)

    # --- Aggregate temporal activations ---
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    # --- Feature Extraction Layer (64 dims) ---
    x = layers.Dense(cfg.CNN_FEATURE_DIM, activation='relu', name='feature_layer')(x)

    # --- High dropout for small dataset ---
    x = layers.Dropout(cfg.CNN_DROPOUT, name='dropout')(x)

    # --- Output ---
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='exoplanet_cnn')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.CNN_LEARNING_RATE),
        loss=focal_loss(),
        metrics=['accuracy']
    )

    print("[INFO] CNN Model Summary:")
    model.summary()
    return model


def train_cnn(model, X_train, y_train, X_val, y_val,
              epochs=None, batch_size=None):
    """
    Train the CNN on REAL (non-SMOTE) data with class weights.

    Class weights handle imbalance instead of SMOTE, because SMOTE
    on raw flux creates unrealistic synthetic light curves that
    don't represent real transit signals.
    """
    if epochs is None:
        epochs = cfg.CNN_EPOCHS
    if batch_size is None:
        batch_size = cfg.CNN_BATCH_SIZE

    n_total = len(y_train)
    n_pos = int(y_train.sum())
    n_neg = n_total - n_pos

    class_weight = {
        0: n_total / (2.0 * max(n_neg, 1)),
        1: min(n_total / (2.0 * max(n_pos, 1)), cfg.CLASS_WEIGHT_CAP)
    }
    print(f"[INFO] Class weights: 0={class_weight[0]:.3f}, 1={class_weight[1]:.3f}")

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=cfg.CNN_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=cfg.CNN_LR_FACTOR,
        patience=cfg.CNN_LR_PATIENCE,
        min_lr=cfg.CNN_LR_MIN,
        verbose=1
    )

    print(f"[INFO] Training CNN for up to {epochs} epochs (batch_size={batch_size})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return history


def find_optimal_threshold(model, X_val, y_val):
    """
    Find the classification threshold that maximizes F1-score on validation data.
    """
    y_prob = model.predict(X_val, verbose=0).ravel()

    best_f1 = 0
    best_threshold = cfg.THRESHOLD_FALLBACK

    thresholds = np.linspace(cfg.THRESHOLD_MIN, cfg.THRESHOLD_MAX, cfg.THRESHOLD_STEPS)
    if len(thresholds) <= 10:
        print(f"[WARN] Threshold search testing only {len(thresholds)} values — check THRESHOLD_STEPS in config.py")
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    if best_f1 == 0:
        print(f"[WARN] CNN threshold search found no positive predictions — model may not be learning")
    print(f"[INFO] Optimal CNN threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold, best_f1


def save_cnn(model, path):
    """Save the trained CNN model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[INFO] CNN model saved to: {path}")


def load_cnn(path):
    """Load a previously saved CNN model from disk."""
    model = models.load_model(path, custom_objects={'focal_loss': focal_loss()})
    print(f"[INFO] CNN model loaded from: {path}")
    return model
