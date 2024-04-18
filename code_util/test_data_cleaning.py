import unittest
from unittest.mock import Mock, patch
from data_cleaning import preprocess_document, prepare_dataset
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import scipy.sparse as sp

class TestDataCleaning(unittest.TestCase):
    def setUp(self):
        self.lemmatizer = WordNetLemmatizer()

    def test_preprocess_document_basic(self):
        document = "Here's an EXAmPLE. Testing one, two three!"
        expected_output = "example testing one two three"
        result = preprocess_document(document, self.lemmatizer)
        self.assertEqual(result, expected_output)

    @patch('sklearn.feature_extraction.text.TfidfVectorizer.transform', return_value=sp.csr_matrix([[1.0]]))
    @patch('sklearn.preprocessing.OneHotEncoder.transform', return_value=sp.csr_matrix([[1.0]]))
    @patch('sklearn.preprocessing.StandardScaler.transform', return_value=sp.csr_matrix([[1.0]]))
    def test_prepare_dataset_integration(self, mock_scaler, mock_vectorizer, mock_encoder):
        df = pd.DataFrame({
            'text': ['sample text'],
            'cat': ['category'],
            'num': [123]
        })
        scaler = StandardScaler()
        text_transformer = TfidfVectorizer()
        categorical_transformer = OneHotEncoder()
        result = prepare_dataset(df, text_transformer, categorical_transformer, scaler, ['text'], ['cat'], ['num'])
        self.assertTrue(isinstance(result, sp.csr_matrix))

if __name__ == '__main__':
    unittest.main()