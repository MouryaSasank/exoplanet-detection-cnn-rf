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
- **Feature importance** chart
- **Sample light curves** visualization
- **Comparison table** (CSV)
- **HTML report** with all embedded results

## 🧪 Technologies Used

| Library           | Purpose                          |
|-------------------|----------------------------------|
| TensorFlow/Keras  | 1D CNN model                     |
| Scikit-learn      | Random Forest, metrics           |
| imbalanced-learn  | SMOTE oversampling               |
| NumPy / Pandas    | Data manipulation                |
| Matplotlib / Seaborn | Visualizations                |
