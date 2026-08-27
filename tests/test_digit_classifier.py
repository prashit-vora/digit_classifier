import unittest

import numpy as np

from digit_classifier import MLPClassifier, cross_entropy, softmax


class TestMath(unittest.TestCase):
    def test_softmax_rows_sum_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
        probabilities = softmax(logits)
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))

    def test_cross_entropy_prefers_correct_predictions(self):
        labels = np.array([0, 1])
        good = np.array([[0.9, 0.1], [0.1, 0.9]])
        bad = np.array([[0.1, 0.9], [0.9, 0.1]])
        self.assertLess(cross_entropy(good, labels), cross_entropy(bad, labels))


class TestMLPClassifier(unittest.TestCase):
    def test_training_learns_small_problem(self):
        x = np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        labels = np.array([0, 1, 1, 0])
        model = MLPClassifier(2, [8], 2, learning_rate=0.2, seed=4)

        initial_loss, _ = model.evaluate(x, labels)
        model.fit(x, labels, epochs=500, batch_size=4, verbose=False)
        final_loss, final_accuracy = model.evaluate(x, labels)

        self.assertLess(final_loss, initial_loss)
        self.assertEqual(final_accuracy, 1.0)

    def test_seed_reproduces_initialization(self):
        first = MLPClassifier(3, [4], 2, seed=7)
        second = MLPClassifier(3, [4], 2, seed=7)
        for first_weight, second_weight in zip(first.weights, second.weights):
            np.testing.assert_array_equal(first_weight, second_weight)


if __name__ == "__main__":
    unittest.main()
