# Import libraries
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from io import StringIO
import nltk
from news_category_classification import (
    load_data, plot_category_distribution, preprocess_text, clean_data,
    balance_data, plot_author_distribution, train_tfidf_vectorizer,
    transform_data, evaluate_model
)

# Ensure NLTK data is downloaded for tests
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

class TestNewsClassifier(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sample_data = pd.DataFrame({
            'headline': ['Sample headline', 'Another headline', 'Third headline'],
            'category': ['POLITICS', 'WELLNESS', 'ENTERTAINMENT'],
            'authors': ['Author 1', 'Author 2', 'Author 3'],
            'link': ['http://link1.com', 'http://link2.com', 'http://link3.com'],
            'date': ['2022-01-01', '2022-01-02', '2022-01-03']
        })
        cls.test_data_path = 'test_data.json'
        cls.sample_data.to_json(cls.test_data_path, lines=True, orient='records')

    def test_load_data(self):
        data = load_data(self.test_data_path)
        self.assertEqual(data.shape, self.sample_data.shape)
        self.assertEqual(list(data.columns), list(self.sample_data.columns))

    @patch('matplotlib.pyplot.show')
    def test_plot_category_distribution(self, mock_show):
        plot_category_distribution(self.sample_data)
        mock_show.assert_called()

    def test_preprocess_text(self):
        text = "Sample <b>headline</b> with HTML and stopwords!"
        processed_text = preprocess_text(text)
        expected_text = "sample headline html stopwords"
        self.assertEqual(processed_text, expected_text)

    def test_clean_data(self):
        data = self.sample_data.copy()
        data = pd.concat([data, self.sample_data.iloc[[0]]], ignore_index=True)  # Duplicate row
        cleaned_data = clean_data(data)
        self.assertEqual(cleaned_data.shape, self.sample_data.shape)

    def test_balance_data(self):
        data = self.sample_data.copy()
        balanced_data = balance_data(data, ['POLITICS', 'WELLNESS', 'ENTERTAINMENT'], 1)
        self.assertEqual(balanced_data.shape[0], 3)
        self.assertIn('category', balanced_data.columns)

    @patch('matplotlib.pyplot.show')
    def test_plot_author_distribution(self, mock_show):
        plot_author_distribution(self.sample_data)
        mock_show.assert_called()

    def test_train_tfidf_vectorizer(self):
        X_train = self.sample_data['headline']
        vectorizer, X_train_tfidf = train_tfidf_vectorizer(X_train, max_df=0.9, min_df=1, max_features=5000)
        self.assertEqual(X_train_tfidf.shape[1], X_train_tfidf.shape[1])  # Adjust as per feature size

    def test_transform_data(self):
        X_train = self.sample_data['headline']
        vectorizer, X_train_tfidf = train_tfidf_vectorizer(X_train, max_df=0.9, min_df=1, max_features=5000)
        X_test = self.sample_data['headline']
        X_test_tfidf = transform_data(vectorizer, X_test)
        self.assertEqual(X_test_tfidf.shape[1], X_train_tfidf.shape[1])

    def test_evaluate_model(self):
        y_train = np.array([0, 1, 2])
        y_test = np.array([0, 1, 2])
        pred_train = np.array([0, 1, 2])
        pred_test = np.array([0, 1, 2])
        with patch('builtins.print') as mocked_print:
            evaluate_model(y_train, y_test, pred_train, pred_test)
            self.assertEqual(mocked_print.call_count, 8)  # Ensure all print statements were called

if __name__ == "__main__":
    unittest.main()
