**CPS News Category Classification**

This repository contains code for training a news category classification model using a Naive Bayes algorithm. The goal is to classify news headlines into different categories. The project includes data preprocessing, model training, evaluation, and testing.

**Prerequisites**

Ensure you have the necessary libraries installed. You can do this by installing the dependencies listed in `requirements.txt`.

**Setup**

**1. \*\*Clone the Repository\*\***

bash command

git clone https://github.com/your\_username/news-category-classification.git

cd news-category-classification

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

**Main Script**

Execute the main script:

python news\_category\_classification.py 

**Test Script**

Execute the test script:

python -m unittest news_category_classification_unittest.py

**Author**

- Shruthi Shri 
