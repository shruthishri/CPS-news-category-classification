**CPS News Category Classification**

This repository contains code for training a news category classification model using a Naive Bayes algorithm. The goal is to classify news headlines into different categories. The project includes data preprocessing, model training, evaluation, and testing.

**Prerequisites**

Ensure you have the necessary libraries installed. You can do this by installing the dependencies listed in `requirements.txt`.

**Setup**

**1. \*\*Clone the Repository\*\***

`   ````bash

`   `git clone https://github.com/your\_username/news-category-classification.git

`   `cd news-category-classification

**2. Install Dependencies**

pip install -r requirements.txt

**3. Download NLTK** 

Ensure you have the necessary NLTK data downloaded:

import nltk

nltk.download('stopwords')

nltk.download('punkt')

nltk.download('wordnet')

**4. Place Data Files** 

Place your JSON data files in the appropriate paths. For example:

- **CPS\_use\_case\_classification\_training.json** for training data
- **CPS\_use\_case\_classification\_response.json** for test data

**Usage**

**Running the Script**

Execute the main script:

python news\_category\_classification.py 

**Main Script**

The main script **news\_category\_classification.py** performs the following steps:

1. **Load Data** Loads the training and test datasets from JSON files.
1. **Initial Data Exploration** Prints the shape, head, and info of the training data for initial inspection.
1. **Plot Category Distribution** Visualizes the distribution of categories in the dataset.
1. **Clean Data** Handles duplicates and null values in the dataset.
1. **Balance Data** Resamples each category to balance the dataset.
1. **Preprocess Text Data** Preprocesses headlines by removing HTML tags, special characters, stopwords, and performing lemmatization.
1. **Split Data** Splits the data into training and validation sets.
1. **Train TF-IDF Vectorizer** Trains a TF-IDF vectorizer on the training data and transforms both training and validation data.
1. **Train Naive Bayes Model** Trains a Multinomial Naive Bayes model on the TF-IDF features.
1. **Evaluate Model** Evaluates the model using accuracy, recall, precision, and F1 score.
1. **Predict on Test Data** Preprocesses the test data, transforms it using the trained TF-IDF vectorizer, and makes predictions using the trained model.
1. **Save Predictions** Saves the predictions to an Excel file.

**Running Tests**

Unit tests are provided in the **test\_news\_category\_classification.py** file. Run the tests using:

python -m unittest discover 

**Functions**

**load\_data(file\_path)**

Loads data from a JSON file.

**plot\_category\_distribution(data)**

Plots the distribution of categories in the dataset.

**preprocess\_text(headline)**

Preprocesses the input text by removing HTML tags, special characters, stopwords, and performing lemmatization.

**clean\_data(data)**

Cleans the dataset by handling duplicates and null values.

**balance\_data(data, categories, sample\_size, random\_state=134)**

Balances the dataset by resampling each category.

**plot\_author\_distribution(data)**

Plots the distribution of authors in the dataset.

**train\_tfidf\_vectorizer(X\_train, max\_df=0.99, min\_df=10, max\_features=5000)**

Trains a TF-IDF vectorizer and transforms the data.

**transform\_data(vectorizer, X)**

Transforms the data using a fitted vectorizer.

**evaluate\_model(y\_train, y\_val, pred\_train, pred\_test)**

Evaluates the model and prints the scores.

**Author**

- Shruthi Shri 
