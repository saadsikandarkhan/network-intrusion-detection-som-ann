"""ANN classifier for NSL-KDD network intrusion detection."""

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import normalize


def build_ann_model(input_dim, num_classes=5, hidden_units=45, dropout_rate=0.2):
    """Build a Sequential ANN for multi-class classification.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int
        Number of output classes.
    hidden_units : int
        Neurons in each hidden layer.
    dropout_rate : float
        Dropout rate after first hidden layer.

    Returns
    -------
    keras.Sequential
        Compiled model.
    """
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
                       batch_size=10, epochs=3, hidden_units=45):
    """Train the ANN and return evaluation metrics.

    Returns
    -------
    dict
        Keys: model, y_pred, confusion_matrix, classification_report,
        accuracy, recall, f1, precision, train_time, predict_time.
    """
    X_train_norm = normalize(X_train)
    X_test_norm = normalize(X_test)

    model = build_ann_model(
        input_dim=X_train_norm.shape[1],
        hidden_units=hidden_units,
    )

    start = time.time()
    model.fit(X_train_norm, y_train, batch_size=batch_size, epochs=epochs)
    train_time = time.time() - start

    start = time.time()
    y_pred = model.predict(X_test_norm).argmax(axis=-1)
    predict_time = time.time() - start

    return {
        "model": model,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "multilabel_confusion_matrix": multilabel_confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, digits=3),
        "accuracy": accuracy_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred, average="micro"),
        "f1": f1_score(y_test, y_pred, average="micro"),
        "precision": precision_score(y_test, y_pred, average="micro"),
        "train_time": train_time,
        "predict_time": predict_time,
    }


def grid_search_ann(X_train, y_train, input_dim,
                    param_grid=None, cv=10):
    """Run GridSearchCV to find optimal hyperparameters.

    Parameters
    ----------
    X_train : numpy.ndarray
        Normalized training features.
    y_train : numpy.ndarray
        Training labels.
    input_dim : int
        Number of input features.
    param_grid : dict, optional
        Parameter grid for search.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        Keys: best_params, best_score, grid_search object.
    """
    from scikeras.wrappers import KerasClassifier

    if param_grid is None:
        param_grid = {
            "batch_size": [10, 20],
            "epochs": [5, 10, 15],
            "optimizer": ["adam", "rmsprop"],
        }

    def _build_model(optimizer="adam"):
        return build_ann_model(input_dim=input_dim)

    classifier = KerasClassifier(model=_build_model)
    grid = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        n_jobs=1,
        cv=cv,
    )
    grid.fit(X_train, y_train)

    return {
        "best_params": grid.best_params_,
        "best_score": grid.best_score_,
        "grid_search": grid,
    }
