# 🪐 Exoplanet Detection — Hybrid Deep Feature Extraction + Ensemble Learning

A Python-based machine learning pipeline for detecting exoplanets using NASA Kepler Space Telescope light curves. This project combines **1D CNN deep feature extraction** with **Random Forest ensemble classification** for robust transit detection.

## 📁 Project Structure

```
exoplanet_detection/
├── data/                       # Place dataset CSVs here
│   ├── exoTrain.csv
│   └── exoTest.csv
├── models/                     # Saved trained models (auto-created)
│   ├── cnn_model.keras
│   ├── hybrid_rf.pkl
│   └── baseline_rf.pkl
├── outputs/                    # Generated plots & report (auto-created)
│   ├── confusion_matrix_hybrid.png
│   ├── confusion_matrix_baseline.png
│   ├── pr_curve.png
│   ├── feature_importance.png
│   ├── sample_light_curves.png
│   ├── comparison_table.csv
│   └── report.html
├── preprocessing.py            # Data loading, normalization, SMOTE
├── cnn_feature_extractor.py    # 1D CNN architecture and training
├── hybrid_classifier.py        # CNN feature extraction + Random Forest
├── baseline_model.py           # Baseline RF on raw features
├── evaluate.py                 # Metrics and visualizations
├── main.py                     # Full pipeline runner
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🛠️ Setup & Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Kepler Labelled Time Series dataset from Kaggle:
- **URL**: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data
- Place `exoTrain.csv` and `exoTest.csv` inside the `data/` folder.

### 3. Run the Pipeline

```bash
python main.py
```

The pipeline will:
1. Load and preprocess the Kepler light curves
2. Train a 1D CNN on the flux data
3. Extract deep features from the CNN
4. Train a Hybrid Random Forest on CNN features (with SMOTE)
5. Train a Baseline Random Forest on raw flux (with SMOTE)
6. Evaluate both models and generate all plots
7. Open the HTML report in your browser

## 📊 Output

After running, check the `outputs/` folder for:
- **Confusion matrices** for both models
- **Precision-Recall curve** comparison
- **ROC curve** comparison
- **Feature importance** chart
- **SHAP summary** explainability plot
- **Training curves** (CNN loss + accuracy)
- **Sample light curves** visualization
- **Comparison table** (CSV)
- **Cross-validation results** (`cv_results.csv`)
- **Ablation study** (`ablation_table.csv`) — CNN-only vs PCA+RF vs Hybrid
- **Bootstrap CIs** (`bootstrap_ci.csv`) — 95% confidence intervals
- **HTML report** with all results embedded

## 🧪 Technologies Used

| Library           | Purpose                          |
|-------------------|----------------------------------|
| TensorFlow/Keras  | 1D CNN model                     |
| Scikit-learn      | Random Forest, metrics           |
| imbalanced-learn  | SMOTE oversampling               |
| NumPy / Pandas    | Data manipulation                |
| Matplotlib / Seaborn | Visualizations                |
| SHAP              | Model explainability             |
| SciPy             | Gaussian denoising               |

## 📈 Results Summary

| Metric | Hybrid CNN+RF | Baseline RF |
|--------|:---:|:---:|
| PR-AUC | **0.5606** | 0.1764 |
| F1-Score | **0.5000** | 0.2222 |
| Recall | 0.4000 | 0.2000 |
| ROC-AUC | **0.8727** | 0.6195 |
| MCC | **0.5132** | 0.2149 |

> **Note**: Metrics are from a reference run. Re-running `python main.py` may produce
> slightly different numbers due to CNN training stochasticity, but relative rankings
> are stable across runs.

### 🔬 Key Academic Contributions

- **Ablation Study**: CNN+RF Hybrid vs CNN-only vs PCA(64)+RF — proves CNN-learned features are superior to PCA dimensionality reduction
- **Bootstrap Confidence Intervals**: 1000-resample CIs quantify statistical uncertainty with only 5 test positives
- **SHAP Explainability**: TreeExplainer reveals which CNN-learned features drive each prediction
- **5-Fold Stratified CV**: Validates robustness despite extreme class imbalance (37 vs 5,050)
- **Isotonic Probability Calibration**: CalibratedClassifierCV improves probability reliability for threshold optimization

## ⚠️ Known Limitations

- Only 5 exoplanets in the test set — point metrics are statistically fragile
- CV F1 ranges from 0.14 to 0.60 across folds (high variance due to small positive set)
- SMOTE generates synthetic samples from only 37 real exoplanets
- See the HTML report's "Limitations & Future Work" section for full discussion

