"""Kohonen Self-Organizing Map implementation supporting arbitrary dimensions."""

import math

import numpy as np
from scipy.stats import multivariate_normal


class KohonenSom:
    """Kohonen SOM algorithm supporting SOMs of arbitrary dimension.

    Parameters
    ----------
    shape : tuple of int
        The shape of the network. Each entry corresponds to one dimension.
    input_size : int
        Dimensionality of the input vectors.
    learning_rate : float
        Initial learning rate (should be > 0).
    learning_decay : float
        Exponential decay factor for learning rate.
    initial_radius : float
        Initial neighborhood radius (should be > 0).
    radius_decay : float
        Exponential decay factor for neighborhood radius.
    """

    def __init__(
        self,
        shape,
        input_size,
        learning_rate,
        learning_decay=1,
        initial_radius=1,
        radius_decay=1,
    ):
        self.shape = shape
        self.dimension = len(shape)
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.learning_decay = learning_decay
        self.initial_radius = initial_radius
        self.radius_decay = radius_decay

        # Pre-compute distance matrix with Gaussian transformation
        distance = np.fromfunction(
            self._distance_function,
            tuple(2 * i + 1 for i in shape),
        )
        gaussian_transform = np.vectorize(
            lambda x: multivariate_normal.pdf(x, mean=0, cov=1)
        )
        self.distance = gaussian_transform(distance)

        # Add extra dimension for element-wise multiplication with weights
        self.distance = np.repeat(self.distance, self.input_size, self.dimension - 1)
        self.distance = np.reshape(
            self.distance, shape=(distance.shape + (self.input_size,))
        )

        # Initialize grid with random weights in [-1, 1]
        self.grid = np.random.rand(*(self.shape + (self.input_size,))) * 2 - 1

    def reset(self):
        """Reinitialize the grid with random weights for a fresh training run."""
        self.grid = np.random.rand(*(self.shape + (self.input_size,))) * 2 - 1

    def _distance_function(self, *args):
        """Compute squared Euclidean distance centered at ``self.shape``."""
        return sum((i - x) ** 2 for i, x in zip(args, self.shape))

    def get_bmu(self, sample):
        """Find the Best Matching Unit for a given sample.

        Parameters
        ----------
        sample : numpy.ndarray
            Input vector to match against the SOM grid.

        Returns
        -------
        tuple
            Index of the BMU in the grid.
        """
        distances = np.square(self.grid - sample).sum(axis=self.dimension)
        bmu_index = np.unravel_index(distances.argmin().astype(int), self.shape)
        return bmu_index

    def fit(self, training_data, num_iterations):
        """Train the SOM on the given dataset.

        Parameters
        ----------
        training_data : numpy.ndarray
            2D array where each row is a training sample.
        num_iterations : int
            Number of training iterations.

        Returns
        -------
        self
        """
        sigma = self.initial_radius
        lr = self.learning_rate

        for i in range(1, num_iterations):
            obs = training_data[np.random.choice(training_data.shape[0], 1)][0]
            bmu = self.get_bmu(obs)
            self._update_weights(obs, bmu, sigma, lr)

            # Exponential decay of radius and learning rate
            sigma = self.initial_radius * math.exp(-i * self.radius_decay)
            lr = self.learning_rate * math.exp(-i * self.learning_decay)

        return self

    def _update_weights(self, training_vector, bmu, sigma, learning_speed):
        """Update grid weights based on the neighborhood of the BMU."""
        # Roll distance matrix so BMU is at center
        bmu_distance = self.distance
        for i, bmu_ind in enumerate(bmu):
            bmu_distance = np.roll(bmu_distance, bmu_ind, axis=i)

        # Extract the relevant quadrant (distances from BMU to each node)
        for i, shape_ind in enumerate(self.shape):
            slc = [slice(None)] * len(bmu_distance.shape)
            slc[i] = slice(shape_ind, 2 * shape_ind)
            bmu_distance = bmu_distance[tuple(slc)]

        # Scale by sigma for decreasing radius effect
        bmu_distance = sigma * bmu_distance

        learning_matrix = -(self.grid - training_vector)
        self.grid += learning_speed * (bmu_distance * learning_matrix)
