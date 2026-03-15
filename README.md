# Network Intrusion Detection using SOM-KNN and ANN

Detecting network intrusion attacks using **Self-Organizing Maps (SOM)** with **K-Nearest Neighbor** scoring and **Artificial Neural Networks (ANN)**.

## Project Structure

```
├── data/                          # NSL-KDD dataset (CSV files)
├── src/
│   ├── som_anomaly_detector/      # SOM-based anomaly detection
│   │   ├── kohonen_som.py         # Custom Kohonen SOM (arbitrary dimensions)
│   │   ├── anomaly_detection.py   # SOM + KNN anomaly detector
│   │   └── minisom.py             # Lightweight 2D SOM implementation
│   ├── ann_classifier/            # ANN-based classifiers
│   │   ├── nsl_kdd_classifier.py  # ANN for NSL-KDD classification
│   │   └── mirai_classifier.py    # ANN for Mirai botnet detection
│   ├── data_utils.py              # Data loading and preprocessing
│   └── visualization.py           # Plotting utilities
├── notebooks/
│   ├── som_knn_intrusion_detection.ipynb  # SOM-KNN analysis (main)
│   └── ann_nsl_kdd_classification.ipynb   # ANN classification analysis
├── tests/                         # Unit tests
└── requirements.txt
```

## Approach

1. **SOM-KNN Anomaly Detection** — Train SOM on benign traffic, detect anomalies via KNN distance scoring
2. **ANN Classification** — Supervised multi-class classification of attack types
3. **Bayesian Optimization** — Hyperparameter tuning with hyperopt (TPE)
4. **Feature Selection** — ExtraTreesClassifier for dimensionality reduction

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html) — 41 network traffic features, 5 attack categories:

| Category | Description | Detection Rate |
|----------|-------------|----------------|
| Normal   | Benign traffic | 1.7% FP |
| Probe    | Network scanning | 90.1% |
| R2L      | Remote-to-Local | 5.4% |
| DoS      | Denial of Service | 85.7% |
| U2R      | User-to-Root | 100% |

## Setup

```bash
pip install -r requirements.txt
```

Place NSL-KDD CSV files (`KDDTrain+.csv`, `KDDTrain+_20Percent.csv`, `KDDtest.csv`) in the `data/` directory.

## Usage

Open the notebooks in `notebooks/` to run the analysis:

```bash
cd notebooks
jupyter notebook som_knn_intrusion_detection.ipynb
```
