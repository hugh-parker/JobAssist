import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import text_embedding

class TestTextEmbedding(unittest.TestCase):

    def setUp(self):
        # Set up dummy word vectors using a MagicMock
        self.word_vectors = MagicMock()
        self.word_vectors.__contains__.side_effect = lambda x: x in ['software', 'engineer', 'development']
        self.word_vectors.__getitem__.side_effect = lambda x: np.random.random(100)
        self.word_vectors.vector_size = 100

        # Sample DataFrame updated to include necessary columns
        self.df = pd.DataFrame({
            'Job Description': ['software engineer responsibilities include development'],
            'Preprocessed Job Description': ['software engineer responsibilities include development'],
            'Preprocessed Responsibilities': ['develop high quality software'],
            'category1': ['tech'],  # Example categorical data
            'category2': ['full-time'],  # Example categorical data
            'num1': [5],  # Example numerical data
            'num2': [10]  # Example numerical data
        })

        # Transformers
        self.categorical_transformer = MagicMock()
        self.numerical_transformer = MagicMock()

        # Dummy transformed data (sparse format)
        self.categorical_transformer.transform.return_value = csr_matrix(np.random.random((1, 5)))
        self.numerical_transformer.transform.return_value = csr_matrix(np.random.random((1, 3)))

    def test_document_vector(self):
        doc = 'software engineer'
        vector = text_embedding.document_vector(doc, self.word_vectors)
        self.assertEqual(vector.shape, (100,))  # Check if the vector is of correct shape

    def test_embed_w2v_dataframe(self):
        categorical_features = ['category1', 'category2']
        numerical_features = ['num1', 'num2']

        # Call the function
        result = text_embedding.embed_w2v_dataframe(self.df, self.word_vectors, self.categorical_transformer, self.numerical_transformer, categorical_features, numerical_features)

        # Check the result is a sparse matrix
        self.assertIsInstance(result, csr_matrix)
        self.assertTrue(result.shape[1] > 0)  # Ensure that some features are concatenated

    def test_embed_description(self):
        result = text_embedding.embed_description(self.df, self.word_vectors)
        self.assertIsInstance(result, np.ndarray)  # Ensure the output is a numpy array
        self.assertEqual(result.shape[1], 100)  # Check the vector dimensionality

if __name__ == '__main__':
    unittest.main()