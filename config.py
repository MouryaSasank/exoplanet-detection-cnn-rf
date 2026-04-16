"""
config.py — Centralized Hyperparameters and Configuration

All tuneable values in one place for reproducibility and easy experimentation.
"""

# ─── RANDOM SEEDS ─────────────────────────────────────────────
GLOBAL_SEED = 42

# ─── DATA ─────────────────────────────────────────────────────
VALIDATION_SPLIT = 0.15
GAUSSIAN_SIGMA = 3          # Gaussian smoothing sigma for denoising light curves

# ─── CNN ──────────────────────────────────────────────────────
CNN_EPOCHS = 100             # Extended from 60 for deeper CNN convergence (kept from Run 2)
CNN_BATCH_SIZE = 32
CNN_LEARNING_RATE = 0.0003   # Lower LR for stability with small positive set
CNN_FEATURE_DIM = 64         # HARD CONSTRAINT: 64 features — never change
CNN_DROPOUT = 0.4            # Reduced from 0.6 to improve feature learning stability
CNN_PATIENCE = 20            # EarlyStopping patience (kept from Run 2)
CNN_LR_PATIENCE = 5          # ReduceLROnPlateau patience — reverted from 7 to 5
CNN_LR_FACTOR = 0.5
CNN_LR_MIN = 1e-6
L2_REG = 0.001              # L2 regularization for conv layers

# Focal loss parameters (better than BCE for imbalanced data)
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0

# Class weight cap for CNN training
CLASS_WEIGHT_CAP = 50.0     # Kept from Run 2 — preserves more imbalance signal

# ─── RANDOM FOREST (HYBRID) ──────────────────────────────────
# Tuned: depth=10 gives expressiveness without SMOTE memorization (depth=12+leaf=1 overfits)
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RF_MIN_SAMPLES_LEAF = 2
RF_MIN_SAMPLES_SPLIT = 3

# ─── RANDOM FOREST (BASELINE) ────────────────────────────────
BASELINE_N_ESTIMATORS = 300
BASELINE_MAX_DEPTH = 20
BASELINE_MIN_SAMPLES_LEAF = 2

# ─── SMOTE ────────────────────────────────────────────────────
# Raised from 0.1 → 0.15 — creates ~645 synthetic positives (safe middle ground, 0.3 caused regression)
SMOTE_SAMPLING_STRATEGY = 0.15   # Hybrid only — on CNN features
SMOTE_K_NEIGHBORS = 5
BASELINE_SMOTE_STRATEGY = 0.3    # Baseline only — on raw flux (unchanged)

# ─── THRESHOLD OPTIMIZATION (HYBRID) ─────────────────────────
# Hybrid threshold search: np.linspace(0.08, 0.55, 60) — tuned for isotonic calibration range
THRESHOLD_MIN = 0.08             # Catches lower isotonic-calibrated probs
THRESHOLD_MAX = 0.55             # Isotonic probs rarely exceed 0.55 on imbalanced data
THRESHOLD_STEPS = 60
THRESHOLD_FALLBACK = 0.15        # Lower fallback for calibrated probabilities

# ─── THRESHOLD OPTIMIZATION (BASELINE) ───────────────────────
# CRITICAL FIX: Baseline sigmoid probabilities are compressed far below 0.20
# ROC-AUC=0.9512 proves the model learned — only the threshold was wrong
# Floor at 0.01 guarantees the search finds the probability range
BASELINE_THRESHOLD_MIN = 0.01    # Very low floor — baseline probs may be < 0.10
BASELINE_THRESHOLD_MAX = 0.50    # Ceiling for baseline threshold search
BASELINE_THRESHOLD_STEPS = 50    # More steps for finer granularity at low end
BASELINE_THRESHOLD_FALLBACK = 0.01  # Last-resort fallback — will still predict some positives

# ─── CROSS-VALIDATION ────────────────────────────────────────
CV_N_SPLITS = 5
CV_RF_N_ESTIMATORS = 200
CV_RF_MAX_DEPTH = 10  # Must match RF_MAX_DEPTH for honest cross-validation

# ─── PATHS (relative to BASE_DIR) ────────────────────────────
DATA_SUBDIR = 'data'
MODEL_SUBDIR = 'models'
OUTPUT_SUBDIR = 'outputs'
TRAIN_FILENAME = 'exoTrain.csv'
TEST_FILENAME = 'exoTest.csv'
