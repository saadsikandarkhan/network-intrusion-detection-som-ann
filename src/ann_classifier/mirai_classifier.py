"""ANN classifier for Mirai botnet traffic detection."""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

TARGET_NAMES = ["Benign", "ACK", "Scan", "SYN", "UDP"]


def load_mirai_data(file_paths):
    """Load and label Mirai botnet traffic datasets.

    Parameters
    ----------
    file_paths : dict
        Mapping of class_label (int) -> csv file path.
        Example: {0: "benign_traffic.csv", 1: "ack.csv", ...}

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        (features, labels)
    """
    frames = []
    for label, path in sorted(file_paths.items()):
        df = pd.read_csv(path)
        df["class"] = label
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    targets = combined["class"].values.astype(int)
    features = combined.drop(columns=["class"]).values

    return features, targets


def preprocess_mirai(features, targets, test_size=0.25, random_state=0):
    """Feature selection, normalization, and train-test split.

    Returns
    -------
    dict
        Keys: X_train, X_test, y_train, y_test, selector, n_features.
    """
    clf = ExtraTreesClassifier(random_state=42)
    clf.fit(features, targets)
    selector = SelectFromModel(clf, prefit=True)
    features_selected = selector.transform(features)

    X_train, X_test, y_train, y_test = train_test_split(
        features_selected, targets, test_size=test_size, random_state=random_state
    )

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "selector": selector,
        "n_features": X_train.shape[1],
    }


def build_mirai_model(input_dim, num_classes=5, hidden_units=21, dropout_rate=0.2):
    """Build a Sequential ANN for Mirai traffic classification."""
    from keras.models import Sequential
    from keras.layers import Dense, Dropout

    model = Sequential([
        Dense(hidden_units, kernel_initializer="uniform",
              activation="relu", input_dim=input_dim),
        Dropout(rate=dropout_rate),
        Dense(hidden_units, kernel_initializer="uniform", activation="relu"),
        Dense(num_classes, kernel_initializer="uniform", activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(X_train, y_train, X_test, y_test,
                       batch_size=10, epochs=3, hidden_units=21):
    """Train the Mirai classifier and return metrics."""
    model = build_mirai_model(input_dim=X_train.shape[1], hidden_units=hidden_units)

    start = time.time()
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    train_time = time.time() - start

    y_pred = model.predict(X_test).argmax(axis=-1)

    return {
        "model": model,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "multilabel_confusion_matrix": multilabel_confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, digits=3, target_names=TARGET_NAMES
        ),
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, average="micro"),
        "f1": f1_score(y_test, y_pred, average="micro"),
        "precision": precision_score(y_test, y_pred, average="micro"),
        "train_time": train_time,
    }
