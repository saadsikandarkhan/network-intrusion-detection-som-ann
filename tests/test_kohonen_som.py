"""Unit tests for KohonenSom and AnomalyDetection."""

import numpy as np

from src.som_anomaly_detector.kohonen_som import KohonenSom
from src.som_anomaly_detector.anomaly_detection import AnomalyDetection


class TestKohonenSom:
    def test_initialization(self):
        som = KohonenSom(shape=(5, 5), input_size=3, learning_rate=0.5)
        assert som.grid.shape == (5, 5, 3)
        assert som.dimension == 2

    def test_reset(self):
        som = KohonenSom(shape=(5, 5), input_size=3, learning_rate=0.5)
        original = som.grid.copy()
        som.reset()
        assert not np.array_equal(som.grid, original)

    def test_get_bmu(self):
        som = KohonenSom(shape=(3, 3), input_size=2, learning_rate=0.5)
        sample = np.array([0.5, 0.5])
        bmu = som.get_bmu(sample)
        assert len(bmu) == 2
        assert all(0 <= b < 3 for b in bmu)

    def test_fit(self):
        som = KohonenSom(shape=(3, 3), input_size=2, learning_rate=0.5)
        data = np.random.rand(20, 2)
        result = som.fit(data, 10)
        assert result is som


class TestAnomalyDetection:
    def test_fit_and_evaluate(self):
        detector = AnomalyDetection(
            shape=(5, 5), input_size=2, learning_rate=0.5,
            min_bmu_count=0, n_neighbors=2,
        )
        training = np.random.rand(50, 2)
        detector.fit(training, 100)
        scores = detector.evaluate(training)
        assert scores.shape == (50,)

    def test_evaluate_before_fit_raises(self):
        detector = AnomalyDetection(
            shape=(3, 3), input_size=2, learning_rate=0.5,
        )
        try:
            detector.evaluate(np.random.rand(5, 2))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass
