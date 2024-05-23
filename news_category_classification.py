# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.utils import resample, shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.naive_bayes import MultinomialNB

# Ensure NLTK data is downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

def load_data(file_path):
    """Loads data from a JSON file."""
    return pd.read_json(file_path, lines=True)

def plot_category_distribution(data):
    """Plots the distribution of categories in the dataset."""
    category = data['category'].value_counts()
    plt.figure(figsize=(25, 8))
    sns.barplot(x=category.index, y=category.values)
    plt.title("The distribution of categories")
    plt.xlabel("Category")
    plt.ylabel("The number of samples")
    plt.xticks(rotation=60, fontsize=14)
    plt.show()

    plt.figure(figsize=(20, 20))
    plt.pie(category.values, autopct="%1.1f%%", labels=category.index)
    plt.show()
    plt.savefig(r"./category_pie.png")

    plt.figure(figsize=(25, 13))
    sns.barplot(y=category.index, x=category.values)
    plt.title("The distribution of categories")
    plt.xlabel("Category")
    plt.ylabel("The number of samples")
    plt.yticks(rotation=0, fontsize=16)
    plt.show()
    plt.savefig(r"./category_bar.png")

def preprocess_text(headline):
    """Preprocesses the input text."""
    headline = re.sub(r'<.*?>', '', headline)  # Remove HTML tags
    headline = ''.join([char if char.isalnum() else ' ' for char in headline])  # Remove special characters
    headline = headline.lower()  # Convert to lowercase
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(headline)
    headline = ' '.join([word for word in words if word not in stop_words])  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    headline = ' '.join([lemmatizer.lemmatize(word) for word in headline.split()])  # Lemmatize words
    return headline

def clean_data(data):
    """Cleans the dataset by handling duplicates and null values."""
    data.drop_duplicates(keep='last', inplace=True)
    data.drop_duplicates(subset=['headline'], keep='last', inplace=True)
    data = data[data['headline'] != '']
    data = data[data['authors'] != '']
    return data

def balance_data(data, categories, sample_size, random_state=134):
    """Balances the dataset by resampling each category."""
    resampled_data = [
        resample(data[data['category'] == category], 
                 replace=False, 
                 n_samples=sample_size, 
                 random_state=random_state)
        for category in categories
    ]
    return pd.concat(resampled_data)

def plot_author_distribution(data):
    """Plots the distribution of authors in the dataset."""
    author_count = data['authors'].value_counts()
    plt.figure(figsize=(25, 18))
    sns.barplot(y=author_count[:25].index, x=author_count[:25].values)
    plt.title("The distribution of authors")
    plt.xlabel("Author Name")
    plt.ylabel("The number of samples")
    plt.yticks(rotation=0, fontsize=18)
    plt.show()
    plt.savefig(r"./author_bar.png")

def train_tfidf_vectorizer(X_train, max_df=0.99, min_df=10, max_features=5000):
    """Trains a TF-IDF vectorizer and transforms the data."""
    vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, ngram_range=(1, 2), lowercase=False, max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
    return vectorizer, X_train_tfidf

def transform_data(vectorizer, X):
    """Transforms the data using a fitted vectorizer."""
    return vectorizer.transform(X).toarray()

def evaluate_model(y_train, y_val, pred_train, pred_test):
    """Evaluates the model and prints the scores."""
    print("Train data accuracy score: ", accuracy_score(y_train, pred_train))    
    print("Test data accuracy score: ", accuracy_score(y_val, pred_test))
    print("Recall score on train data: ", recall_score(y_train, pred_train, average='macro'))
    print("Recall score on test data: ", recall_score(y_val, pred_test, average='macro'))
    print("Precision score on train data: ", precision_score(y_train, pred_train, average='macro'))
    print("Precision score on test data: ", precision_score(y_val, pred_test, average='macro'))
    print("F1 score on train data: ", f1_score(y_train, pred_train, average='macro'))
    print("F1 score on test data: ", f1_score(y_val, pred_test, average='macro'))

def main():
    # Load data
    train_data = load_data('C:/Users/shrut/Downloads/CPS/CPS_use_case_classification_training.json')
    
    # Initial data exploration
    print(train_data.shape)
    print(train_data.head())
    print(train_data.info())
    
    # Plot category distribution
    plot_category_distribution(train_data)
    
    # Clean data
    train_data = clean_data(train_data)
    
    # Balance the dataset
    categories = [
        'POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'TRAVEL', 'STYLE & BEAUTY', 
        'PARENTING', 'HEALTHY LIVING', 'QUEER VOICES', 'FOOD & DRINK', 
        'BUSINESS', 'COMEDY', 'PARENTS', 'SPORTS', 'HOME & LIVING'
    ]
    train_data = balance_data(train_data, categories, sample_size=2500)
    print(train_data['category'].value_counts())
    
    # Drop unnecessary columns
    train_data.drop(['link', 'authors', 'date'], axis=1, inplace=True)
    
    # Shuffle the dataset
    train_data = shuffle(train_data).reset_index(drop=True)
    
    # Preprocess text data
    train_data['headline'] = train_data['headline'].apply(preprocess_text)

    # Plot category distribution
    plot_category_distribution(train_data)
    
    # Train and test split
    X = train_data['headline']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(train_data['category'])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=2)
    print(f"The X_train shape: {X_train.shape}")
    print(f"The X_val shape: {X_val.shape}")
    print(f"The y_train shape: {y_train.shape}")
    print(f"The y_val shape: {y_val.shape}")
    
    # Train TF-IDF Vectorizer and transform data
    vectorizer, X_train_tfidf = train_tfidf_vectorizer(X_train)
    X_val_tfidf = transform_data(vectorizer, X_val)
    
    # Model training
    print("Multinomial Naive Bayes Algorithm")
    multinb = MultinomialNB()
    multinb.fit(X_train_tfidf, y_train)
    y_train_pred = multinb.predict(X_train_tfidf)
    y_val_pred = multinb.predict(X_val_tfidf)
    evaluate_model(y_train, y_val, y_train_pred, y_val_pred)
    
    # Test data handling
    test_data = load_data('C:/Users/shrut/Downloads/CPS/CPS_use_case_classification_response.json')
    test_data = clean_data(test_data)
    test_data['headline'] = test_data['headline'].apply(preprocess_text)
    X_test_text = test_data['headline']
    X_test_tfidf_text = transform_data(vectorizer, X_test_text)
    y_predict = multinb.predict(X_test_tfidf_text)

    predicted_categories = label_encoder.inverse_transform(y_predict)
    
    # Create submission
    test_predictions = pd.DataFrame(list(zip(test_data['headline'], predicted_categories)), columns=['headline', 'category'])
    test_predictions.to_excel('category_predictions.xlsx', index=False)

if __name__ == "__main__":
    main()