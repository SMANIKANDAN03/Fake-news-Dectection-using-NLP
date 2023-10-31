# Fake News Detection

## Overview

This project implements a fake news detection system using Natural Language Processing (NLP) techniques and machine learning models. It combines datasets of fake and true news articles, preprocesses the text, and trains two classification models (Naive Bayes and Random Forest) for identification.

## Dependencies

- pandas
- nltk
- scikit-learn
- transformers (for BERT-based model)
- torch

Install the dependencies using:

pip install pandas 
pip install  nltk
pip install scikit-learn
pip install tensorflow(For BERT only)
pip install transformers(For BERT only)
pip install torch(For BERT only)

## How to run
Download NLTK Resources:

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

Download and Load Datasets:

Download the fake and true news datasets, and update the file paths in the script accordingly. The datasets used in the script are Fake.csv and True.csv.
The dataset is available in Kaggle.
The dataset link : https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Run the Script

## Fake News Detection Algorithm:

Step 1: Import Libraries:
Import necessary libraries, including pandas, nltk, re, and scikit-learn.

Step 2 : Download NLTK Resources:
Download NLTK resources for stopwords, punkt, and wordnet.

Step 3 : Load Datasets:
Load fake and true news datasets (Fake.csv and True.csv).

Step 4 : Combine Datasets:
Assign labels (1 for fake, 0 for true) to the datasets.
Combine datasets into one DataFrame.

Step 5 : Data Preprocessing:
Remove HTML tags and non-alphabetical characters.
Convert text to lowercase.
Tokenize the text.
Remove stopwords.
Lemmatize the tokens.

Step 6 : Text Vectorization (TF-IDF):
Use TF-IDF vectorization to convert text data into numerical 

Step 7 : Split the Data:
Split the dataset into training and testing sets.

Step 8 : Train Naive Bayes Model:
Train a Naive Bayes classifier on the TF-IDF features.

Step 9 : Evaluate Naive Bayes Model:
Make predictions on the test set.
Calculate and print the accuracy of the Naive Bayes model.

Step 10 : Train Random Forest Model:
Train a Random Forest classifier on the TF-IDF features.

Step 11 : Evaluate Random Forest Model:
Make predictions on the test set.
Calculate and print the accuracy of the Random Forest model.

## Code

!pip install transformers

import pandas as pd
import nltkimport re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenizefromnltk.stemimportWordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrixfrom transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


fake_data = pd.read_csv('/content/Fake.csv')
print(fake_data.info())
print(fake_data.head())


true_data = pd.read_csv('/content/True.csv')
print(true_data.info())
print(true_data.head())


fake_data['label'] = 1
true_data['label']=0
data = pd.concat([fake_data, true_data], ignore_index=True)print(data.info())

data['text'] = data['text'].apply(lambda x: re.sub('<[^>]+>', '', x))
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-Z\s]', '', x))


data['text'] = data['text'].str.lower()

data['tokens'] = data['text'].apply(word_tokenize)


stop_words = set(stopwords.words('english'))
data['filtered_tokens'] = data['tokens'].apply(lambda tokens: [word for word in tokens if word not in stop_words])


lemmatizer = WordNetLemmatizer()
data['lemmatized_tokens'] = data['filtered_tokens'].apply(lambda tokens: [lemmatizer.lemmatize(word) for word in tokens])


tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(data['lemmatized_tokens'].apply(' '.join))

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, data['label'], test_size=0.2, random_state=42)


nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)print(f'NaiveBayesAccuracy:{nb_accuracy}')
print(f'Random Forest Accuracy: {rf_accuracy}')

## Result
After running the script, you will see the accuracy of the Naive Bayes model printed in the console. 