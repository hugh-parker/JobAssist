import unittest
from unittest.mock import MagicMock, patch, mock_open
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import model_util
from sklearn.model_selection import GridSearchCV
from unittest.mock import patch, MagicMock, mock_open


class TestModelUtil(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe
        self.df = pd.DataFrame({
            'job_description': ['software developer', 'web developer', 'data scientist'],
            'skills': [['python', 'java'], ['html', 'css'], ['python', 'r']]
        })
        self.job_description = "software engineer"
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_vectorizer.fit(self.df['job_description'])

    def test_simple_get_skills(self):
        with patch('model_util.dc.preprocess_document', return_value='software engineer') as mock_preprocess:
            skills = model_util.simple_get_skills(self.df, self.job_description, self.tfidf_vectorizer, skill_amt=2)
            self.assertIsInstance(skills, set)
            self.assertEqual(len(skills), 3)  # Assuming overlap in top 2 similar job descriptions

    @patch('pickle.load')
    @patch('builtins.open', new_callable=mock_open, read_data=b'some data')
    def test_load_model_exists(self, mock_file, mock_pickle):
        mock_pickle.return_value = 'Loaded model'
        model = model_util.load_model(None, 'path/to/model.pkl', {}, None, None, 'accuracy', force_reload=False)
        self.assertEqual(model, 'Loaded model')

    @patch('pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('model_util.GridSearchCV')
    def test_load_model_force_reload(self, mock_GridSearchCV, mock_file, mock_pickle_dump):
        # Configure the mock GridSearchCV to simulate its behavior
        mock_grid_search = mock_GridSearchCV.return_value
        mock_grid_search.fit.return_value = None  # fit() does not actually return anything
        mock_grid_search.best_estimator_ = 'Trained model'
        mock_grid_search.best_params_ = {'param': 'value'}

        # Make sure the base model and data are set up correctly
        base_model = MagicMock()
        X_train = pd.DataFrame({"feature": [1, 2, 3]})
        y_train = pd.Series([0, 1, 1])
        
        # Call the function
        model = model_util.load_model(base_model, 'path/to/model.pkl', {}, X_train, y_train, 'accuracy', force_reload=True)
        
        # Check if the model returned is the trained model
        self.assertEqual(model, 'Trained model')
        
        # Ensure that pickle.dump was called (hence file.write was called)
        mock_pickle_dump.assert_called_once()
        mock_file.assert_called_once_with('path/to/model.pkl', 'wb')  # Check if the file was opened in write-binary mode

if __name__ == '__main__':
    unittest.main()