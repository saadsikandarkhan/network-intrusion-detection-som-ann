# Network Intrusion Detection using SOM-KNN and ANN

This repository contains the code for the research paper:

> **Discover botnets in IoT sensor networks: A lightweight deep learning framework with hybrid self-organizing maps**
> by Saad Khan
> [Published in Microprocessors and Microsystems](https://www.sciencedirect.com/science/article/abs/pii/S0141933122002824)

## About the Paper

The paper proposes a lightweight deep learning framework for detecting botnet attacks in IoT sensor networks. As IoT devices are resource-constrained, traditional intrusion detection systems are too computationally expensive to deploy at the edge. This work addresses that challenge by combining **Self-Organizing Maps (SOM)** with **K-Nearest Neighbor (KNN)** distance scoring for unsupervised anomaly detection, alongside a supervised **Artificial Neural Network (ANN)** classifier. The SOM learns the topology of normal network traffic and flags deviations as potential intrusions, while the ANN provides multi-class attack classification. Bayesian optimization (TPE) is used to tune hyperparameters for both models.

## Reproducing the Results

The two notebooks in `notebooks/` reproduce the key experiments from the paper:

1. **`som_knn_intrusion_detection.ipynb`** — Runs the full SOM-KNN anomaly detection pipeline: data preprocessing, feature selection, SOM hyperparameter tuning via Bayesian optimization, SOM training on benign traffic, and per-category detection rate evaluation.
2. **`ann_nsl_kdd_classification.ipynb`** — Trains and evaluates the ANN classifier with cross-validation and GridSearchCV hyperparameter tuning.

Both notebooks use the NSL-KDD dataset and the modules in `src/` to produce the detection rates and evaluation metrics reported in the paper.

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
│   ├── preprocessing.py           # Data loading and preprocessing
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
