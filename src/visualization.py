"""Visualization utilities for SOM and anomaly detection results."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Markers and colors for up to 8 classes
_MARKERS = ["o", "s", "D", "+", "v", "p", "*", "x"]
_COLORS = ["r", "g", "b", "y", "pink", "orange", "purple", "grey"]


def plot_som(som, data, target=None, figsize=(16, 12)):
    """Visualize SOM distance map with data points colored by class.

    Parameters
    ----------
    som : MiniSom
        Trained MiniSom instance.
    data : numpy.ndarray
        Data samples to plot on the SOM.
    target : numpy.ndarray, optional
        Class labels for coloring the markers.
    figsize : tuple
        Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(som.distance_map().T, cmap="bone")
    plt.colorbar(im, ax=ax)

    for cnt, xx in enumerate(data):
        w = som.winner(xx)
        label = target[cnt] if target is not None else 0
        ax.plot(
            w[0] + 0.5, w[1] + 0.5,
            _MARKERS[label],
            markerfacecolor="None",
            markeredgecolor=_COLORS[label],
            markersize=12,
            markeredgewidth=2,
        )

    ax.set_xlim(0, som.weights.shape[0])
    ax.set_ylim(0, som.weights.shape[1])
    plt.tight_layout()
    plt.show()


def plot_anomaly_density(benign_metrics, anomaly_metrics, threshold,
                         title="Anomaly Detection", figsize=(12, 4)):
    """Plot density distributions with anomaly threshold line.

    Parameters
    ----------
    benign_metrics : numpy.ndarray
        Anomaly scores for benign data.
    anomaly_metrics : numpy.ndarray
        Anomaly scores for data being evaluated.
    threshold : float
        Decision threshold value.
    title : str
        Plot title prefix.
    figsize : tuple
        Figure size.
    """
    xs = np.linspace(0, 1, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    density_benign = gaussian_kde(benign_metrics)
    ax1.axvline(x=threshold, color="red", linestyle="--", label="Threshold")
    ax1.plot(xs, density_benign(xs))
    ax1.set_title(f"{title} - Benign Distribution")
    ax1.legend()

    density_anomaly = gaussian_kde(anomaly_metrics)
    ax2.axvline(x=threshold, color="red", linestyle="--", label="Threshold")
    ax2.plot(xs, density_anomaly(xs))
    ax2.set_title(f"{title} - Anomaly Distribution")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def get_anomalies(benign_metrics, anomaly_metrics, alpha=3, return_outliers=True):
    """Identify anomalies using standard-deviation threshold.

    Parameters
    ----------
    benign_metrics : numpy.ndarray
        Anomaly scores from benign data (to compute threshold).
    anomaly_metrics : numpy.ndarray
        Anomaly scores to classify.
    alpha : float
        Number of standard deviations above mean for threshold.
    return_outliers : bool
        Whether to return outlier indices.

    Returns
    -------
    numpy.ndarray or None
        Indices of detected anomalies (if ``return_outliers`` is True).
    """
    limit = np.mean(benign_metrics) + np.std(benign_metrics) * alpha
    outliers = np.argwhere(np.abs(anomaly_metrics) > limit)
    pct = len(outliers) / len(anomaly_metrics) * 100
    print(f"Determined: {pct:.2f}% as anomaly")

    if return_outliers:
        return outliers
