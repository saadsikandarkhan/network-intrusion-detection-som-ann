"""Minimalistic Self-Organizing Maps implementation.

A lightweight, 2D SOM with Gaussian neighborhood function.
"""

from collections import defaultdict
from math import sqrt

from numpy import (
    arange,
    array,
    dot,
    exp,
    meshgrid,
    nditer,
    outer,
    pi,
    power,
    random,
    subtract,
    unravel_index,
    zeros,
)
from numpy import linalg
from warnings import warn


def fast_norm(x):
    """Compute L2 norm of a 1D array (faster than ``linalg.norm``)."""
    return sqrt(dot(x, x.T))


class MiniSom:
    """Minimalistic 2D Self-Organizing Map.

    Parameters
    ----------
    x, y : int
        Dimensions of the SOM grid.
    input_len : int
        Length of input vectors.
    sigma : float
        Spread of the Gaussian neighborhood function.
    learning_rate : float
        Initial learning rate.
    decay_function : callable, optional
        Function(x, t, max_iter) that reduces learning_rate and sigma.
    random_seed : int, optional
        Seed for reproducibility.
    """

    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5,
                 decay_function=None, random_seed=None):
        if sigma >= x / 2.0 or sigma >= y / 2.0:
            warn("Warning: sigma is too high for the dimension of the map.")

        self.random_generator = random.default_rng(random_seed)

        if decay_function:
            self._decay_function = decay_function
        else:
            self._decay_function = lambda x, t, max_iter: x / (1 + t / max_iter)

        self.learning_rate = learning_rate
        self.sigma = sigma

        # Random initialization with normalization
        self.weights = self.random_generator.random((x, y, input_len)) * 2 - 1
        for i in range(x):
            for j in range(y):
                self.weights[i, j] /= fast_norm(self.weights[i, j])

        self.activation_map = zeros((x, y))
        self.neigx = arange(x)
        self.neigy = arange(y)
        self.neighborhood = self.gaussian

    def _activate(self, x):
        """Update activation_map with response of each neuron to ``x``."""
        s = subtract(x, self.weights)
        it = nditer(self.activation_map, flags=["multi_index"])
        while not it.finished:
            self.activation_map[it.multi_index] = fast_norm(s[it.multi_index])
            it.iternext()

    def activate(self, x):
        """Return the activation map for input ``x``."""
        self._activate(x)
        return self.activation_map

    def gaussian(self, c, sigma):
        """Return a Gaussian neighborhood centered at ``c``."""
        d = 2 * pi * sigma * sigma
        ax = exp(-power(self.neigx - c[0], 2) / d)
        ay = exp(-power(self.neigy - c[1], 2) / d)
        return outer(ax, ay)

    def diff_gaussian(self, c, sigma):
        """Return a Mexican-hat neighborhood centered at ``c``."""
        xx, yy = meshgrid(self.neigx, self.neigy)
        p = power(xx - c[0], 2) + power(yy - c[1], 2)
        d = 2 * pi * sigma * sigma
        return exp(-p / d) * (1 - 2 / d * p)

    def winner(self, x):
        """Return the (i, j) coordinates of the winning neuron for ``x``."""
        self._activate(x)
        return unravel_index(
            self.activation_map.argmin(), self.activation_map.shape
        )

    def update(self, x, win, t):
        """Update neuron weights given current pattern, winner, and iteration."""
        eta = self._decay_function(self.learning_rate, t, self.T)
        sig = self._decay_function(self.sigma, t, self.T)
        g = self.neighborhood(win, sig) * eta
        it = nditer(g, flags=["multi_index"])
        while not it.finished:
            self.weights[it.multi_index] += g[it.multi_index] * (
                x - self.weights[it.multi_index]
            )
            self.weights[it.multi_index] /= fast_norm(self.weights[it.multi_index])
            it.iternext()

    def quantization(self, data):
        """Assign the winning neuron's weight vector to each sample."""
        q = zeros(data.shape)
        for i, x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q

    def random_weights_init(self, data):
        """Initialize weights by picking random samples from ``data``."""
        it = nditer(self.activation_map, flags=["multi_index"])
        while not it.finished:
            self.weights[it.multi_index] = data[
                self.random_generator.integers(len(data))
            ]
            self.weights[it.multi_index] /= fast_norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self, data, num_iteration):
        """Train by picking random samples from ``data``."""
        self._init_T(num_iteration)
        for iteration in range(num_iteration):
            rand_i = self.random_generator.integers(len(data))
            self.update(data[rand_i], self.winner(data[rand_i]), iteration)

    def train_batch(self, data, num_iteration):
        """Train using all vectors in ``data`` sequentially."""
        self._init_T(len(data) * num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data) - 1)
            self.update(data[idx], self.winner(data[idx]), iteration)
            iteration += 1

    def _init_T(self, num_iteration):
        """Set decay denominator (keeps rate ~constant for last half)."""
        self.T = num_iteration / 2

    def distance_map(self):
        """Return normalized distance map (U-matrix) of the weights."""
        um = zeros((self.weights.shape[0], self.weights.shape[1]))
        it = nditer(um, flags=["multi_index"])
        while not it.finished:
            for ii in range(it.multi_index[0] - 1, it.multi_index[0] + 2):
                for jj in range(it.multi_index[1] - 1, it.multi_index[1] + 2):
                    if 0 <= ii < self.weights.shape[0] and 0 <= jj < self.weights.shape[1]:
                        um[it.multi_index] += fast_norm(
                            self.weights[ii, jj, :] - self.weights[it.multi_index]
                        )
            it.iternext()
        um /= um.max()
        return um

    def activation_response(self, data):
        """Return matrix counting how often each neuron won."""
        a = zeros((self.weights.shape[0], self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self, data):
        """Return average distance between each sample and its BMU."""
        error = 0
        for x in data:
            error += fast_norm(x - self.weights[self.winner(x)])
        return error / len(data)

    def win_map(self, data):
        """Return dict mapping (i, j) -> list of patterns mapped there."""
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap
