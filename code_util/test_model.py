import unittest
from unittest.mock import Mock

from httpx import patch
from model_util import train_model, evaluate_model

class TestModelUtil(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.model = Mock(), Mock(), Mock()

    def test_train_model_calls_fit(self):
        train_model(self.X, self.y, self.model)
        self.model.fit.assert_called_once_with(self.X, self.y)

    def test_evaluate_model_correctness(self):
        self.model.predict.return_value = self.y
        metrics = evaluate_model(self.model, self.X, self.y)
        self.assertIsInstance(metrics, dict)  # Assuming metrics are returned as a dictionary

    def test_evaluate_model_accuracy(self):
        self.model.predict.return_value = self.y
        with patch('model_util.accuracy_score') as mocked_accuracy:
            mocked_accuracy.return_value = 1.0
            metrics = evaluate_model(self.model, self.X, self.y)
            self.assertEqual(metrics.get('accuracy'), 1.0)

# Run the tests
if __name__ == '__main__':
    unittest.main()