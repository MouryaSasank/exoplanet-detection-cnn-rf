"""
cnn_feature_extractor.py — 1D CNN Architecture and Training

This module defines a 1D Convolutional Neural Network that learns to detect
exoplanet transit patterns directly from raw Kepler light curve flux data.

Architecture:
  Input (n_flux, 1) -> Conv1D(64, 5) -> BN -> MaxPool(2)
  -> Conv1D(128, 3) -> BN -> MaxPool(2)
  -> Conv1D(256, 3) -> BN -> MaxPool(2)
  -> GlobalAveragePooling1D -> Dense(256, ReLU) [feature layer]
  -> Dropout(0.5) -> Dense(1, Sigmoid)

The Dense(256) layer serves as the "feature extraction" layer: its outputs
are the learned representations that we later feed to the Random Forest.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import precision_recall_curve, f1_score


def build_cnn(input_shape):
    """
    Build the 1D CNN model for exoplanet detection (Functional API).

    Includes BatchNormalization for training stability and slightly
    higher dropout (0.5) to reduce overfitting on the majority class.

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sample, e.g. (3197, 1)

    Returns
    -------
    model : tf.keras.Model
        Compiled CNN model
    """
    inputs = layers.Input(shape=input_shape, name='input')

    # --- First Convolutional Block ---
    x = layers.Conv1D(filters=64, kernel_size=5, activation='relu',
                      padding='same', name='conv1d_1')(inputs)
    x = layers.BatchNormalization(name='bn_1')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_1')(x)

    # --- Second Convolutional Block ---
    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu',
                      padding='same', name='conv1d_2')(x)
    x = layers.BatchNormalization(name='bn_2')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_2')(x)

    # --- Third Convolutional Block ---
    # 256 filters capture high-level temporal patterns after two rounds of pooling
    x = layers.Conv1D(filters=256, kernel_size=3, activation='relu',
                      padding='same', name='conv1d_3')(x)
    x = layers.BatchNormalization(name='bn_3')(x)
    x = layers.MaxPooling1D(pool_size=2, name='maxpool_3')(x)

    # --- Aggregate temporal activations without exploding parameter count ---
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    # --- Feature Extraction Layer ---
    # This is the key layer: its 256-dim output is the learned feature vector
    x = layers.Dense(256, activation='relu', name='feature_layer')(x)

    # --- Regularization (increased from 0.4 to 0.5) ---
    x = layers.Dropout(0.5, name='dropout')(x)

    # --- Output Layer ---
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='exoplanet_cnn')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print("[INFO] CNN Model Summary:")
    model.summary()
    return model


def train_cnn(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train the CNN with class weighting and early stopping.

    Class weights are computed to handle the severe imbalance between
    exoplanet and non-exoplanet samples.

    Parameters
    ----------
    model : tf.keras.Model
    X_train, y_train : training data (3D for CNN)
    X_val, y_val : validation data (3D for CNN)
    epochs : int
    batch_size : int

    Returns
    -------
    history : tf.keras.callbacks.History
    """
    n_total = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_total - n_pos
    class_weight = {
        0: n_total / (2.0 * n_neg),
        1: min(n_total / (2.0 * n_pos), 20.0)
    }
    print(f"[INFO] Class weights: 0 (non-planet): {class_weight[0]:.3f}, 1 (exoplanet): {class_weight[1]:.3f}")

    early_stop = callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Reduce learning rate when validation loss plateaus
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print(f"[INFO] Training CNN for up to {epochs} epochs...")
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

    Instead of using the default 0.5 threshold, we sweep thresholds from 0.1 to
    0.9 and pick the one that gives the best F1-score for the exoplanet class.

    Parameters
    ----------
    model : tf.keras.Model
        The trained CNN
    X_val : np.ndarray
        Validation features (3D for CNN)
    y_val : np.ndarray
        True labels

    Returns
    -------
    best_threshold : float
        The threshold that maximizes F1-score
    best_f1 : float
        The F1-score at that threshold
    """
    y_prob = model.predict(X_val, verbose=0).ravel()
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)

    best_f1 = 0
    best_threshold = 0.5

    for t in np.arange(0.10, 0.91, 0.01):
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred_t, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"[INFO] Optimal CNN threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return best_threshold, best_f1


def save_cnn(model, path):
    """Save the trained CNN model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"[INFO] CNN model saved to: {path}")


def load_cnn(path):
    """Load a previously saved CNN model from disk."""
    model = models.load_model(path)
    print(f"[INFO] CNN model loaded from: {path}")
    return model
