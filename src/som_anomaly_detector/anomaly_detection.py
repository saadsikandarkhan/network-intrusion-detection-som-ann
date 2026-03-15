"""SOM-based anomaly detection using K-Nearest Neighbors distance scoring."""

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .kohonen_som import KohonenSom


class AnomalyDetection(KohonenSom):
    """Anomaly detector that combines Kohonen SOM with KNN distance scoring.

    After training the SOM on normal/benign data, sparse nodes are pruned.
    Anomaly scores are computed as the mean distance to the K nearest
    remaining SOM nodes.

    Parameters
    ----------
    shape : tuple of int
        SOM grid shape.
    input_size : int
        Dimensionality of input vectors.
    learning_rate : float
        Initial SOM learning rate.
    learning_decay : float
        Decay factor for learning rate.
    initial_radius : float
        Initial SOM neighborhood radius.
    radius_decay : float
        Decay factor for radius.
    min_bmu_count : int
        Minimum activations for a node to be kept (pruning threshold).
    n_neighbors : int
        Number of neighbors for KNN distance scoring.
    """

    def __init__(
        self,
        shape,
        input_size,
        learning_rate,
        learning_decay=0.1,
        initial_radius=1,
        radius_decay=0.1,
        min_bmu_count=1,
        n_neighbors=3,
    ):
        super().__init__(
            shape, input_size, learning_rate, learning_decay, initial_radius, radius_decay
        )
        self.min_bmu_count = min_bmu_count
        self.n_neighbors = n_neighbors
        self.allowed_nodes = None
        self.bmu_counts = None

    def get_bmu_counts(self, training_data):
        """Count how many training samples map to each SOM node.

        Parameters
        ----------
        training_data : numpy.ndarray
            Training data array.

        Returns
        -------
        numpy.ndarray
            Array of same shape as SOM grid with activation counts.
        """
        bmu_counts = np.zeros(shape=self.shape)
        for observation in training_data:
            bmu = self.get_bmu(observation)
            bmu_counts[bmu] += 1
        return bmu_counts

    def fit(self, training_data, num_iterations):
        """Train the SOM and prune sparse nodes.

        Parameters
        ----------
        training_data : numpy.ndarray
            Training data (should contain only normal/benign samples).
        num_iterations : int
            Number of SOM training iterations.

        Returns
        -------
        numpy.ndarray
            The allowed (non-pruned) SOM nodes.
        """
        self.reset()
        super().fit(training_data, num_iterations)
        bmu_counts = self.get_bmu_counts(training_data)
        self.bmu_counts = bmu_counts.flatten()
        self.allowed_nodes = self.grid[bmu_counts >= self.min_bmu_count]
        return self.allowed_nodes

    def evaluate(self, data):
        """Compute anomaly scores for the given data.

        Each sample's score is the mean distance to its K nearest
        allowed SOM nodes. Higher scores indicate greater anomaly.

        Parameters
        ----------
        data : numpy.ndarray
            Data to evaluate.

        Returns
        -------
        numpy.ndarray
            1D array of anomaly scores.

        Raises
        ------
        RuntimeError
            If ``fit`` has not been called or no nodes survived pruning.
        """
        if self.allowed_nodes is None:
            raise RuntimeError("Call fit() before evaluating data.")
        if self.allowed_nodes.shape[0] < 2:
            raise RuntimeError(
                "No nodes satisfy the minimum BMU count threshold. "
                "Try lowering min_bmu_count or increasing training iterations."
            )

        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        knn.fit(self.allowed_nodes)
        distances, _ = knn.kneighbors(data)
        return distances.mean(axis=1)
