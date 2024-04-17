import unittest
from unittest.mock import Mock, patch
from data_cleaning import preprocess_document, prepare_dataset
from nltk.stem import WordNetLemmatizer

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        self.lemmatizer = WordNetLemmatizer()

    def test_preprocess_document_basic(self):
        document = "Here's an EXAMPLE. Testing, one, two, three!"
        expected_output = "here example testing one two three"
        result = preprocess_document(document, self.lemmatizer)
        self.assertEqual(result, expected_output)

    def test_preprocess_document_with_numbers(self):
        document = "Data 2024 with 100% certainty."
        expected_output = "data 2024 with 100 certainty"
        result = preprocess_document(document, self.lemmatizer)
        self.assertEqual(result, expected_output)

    def test_preprocess_document_empty(self):
        document = ""
        expected_output = ""
        result = preprocess_document(document, self.lemmatizer)
        self.assertEqual(result, expected_output)

    @patch('data_cleaning.OneHotEncoder')
    @patch('data_cleaning.TfidfVectorizer')
    @patch('data_cleaning.StandardScaler')
    def test_prepare_dataset_integration(self, MockScaler, MockVectorizer, MockEncoder):
        df = Mock()
        scaler = MockScaler()
        text_transformer = MockVectorizer()
        categorical_transformer = MockEncoder()
        # Mock behavior
        scaler.transform.return_value = 'mocked_numerical_features'
        text_transformer.transform.return_value = 'mocked_text_features'
        categorical_transformer.transform.return_value = 'mocked_categorical_features'
        result = prepare_dataset(df, text_transformer, categorical_transformer, scaler, ['text'], ['cat'], ['num'])
        self.assertIsInstance(result, str)  # Update with actual expected type based on implementation

# Run the tests
if __name__ == '__main__':
    unittest.main()