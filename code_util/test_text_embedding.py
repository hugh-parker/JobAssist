import unittest
from unittest.mock import Mock, patch
from text_embedding import embed_description

class TestTextEmbedding(unittest.TestCase):
    def setUp(self):
        self.df = Mock()
        self.word_vectors = Mock()

    def test_embed_description_basic(self):
        self.df['Job Description'].apply.return_value = self.df
        self.df.__getitem__.return_value.tolist.return_value = ['vector']
        result = embed_description(self.df, self.word_vectors)
        self.assertEqual(result, ['vector'])

    def test_embed_description_with_empty_df(self):
        self.df['Job Description'].apply.return_value = self.df
        self.df.__getitem__.return_value.tolist.return_value = []
        result = embed_description(self.df, self.word_vectors)
        self.assertEqual(result, [])

    def test_embed_description_handles_exceptions(self):
        self.df['Job Description'].apply.side_effect = Exception("Error processing")
        with self.assertRaises(Exception):
            embed_description(self.df, self.word_vectors)

# Run the tests
if __name__ == '__main__':
    unittest.main()