# deep_tweets

## Overview

This repository contains the code and resources for a deep learning-based tweet classification problem. The objective of this project is to classify tweets into predefined categories (Sports or Politics). The project is based on a Kaggle competition, and it involves various machine learning and deep learning models, also some natural language processing techniques.

## Dataset

- **Source**: The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/competitions/deeptweets/data).

- **Data Description**:
  File descriptions :
  - train.csv - the training set
  - test.csv - the test set
  - sample_saubmission.csv - a sample submission file in the correct format
  
  Data fields :
  - TweetId - an anonymous id unique to a given tweet
  - Label - the associated label which is either Sports or Politics
  - TweetText - the text in a tweet

## Technologies
![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB.svg?style=for-the-badge&logo=Python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243.svg?style=for-the-badge&logo=NumPy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Scikitlearn](https://img.shields.io/badge/scikitlearn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)



## Models

In this project, there are sevrale models used such as :
- **RNN** (recurrent neural networks) : for this model a Bidirectional LSTM is used, other dropouts to ever came overfitting.
- **Multinomial Naive Bayes** (MultinomialNB)
- **Logistic Regression** (LogisticRegression)


For natural language processing techniques :
- **Embedding** is used with pre-trained weights from [GloVe](https://nlp.stanford.edu/projects/glove/).
- **TF-IDF**  (Term Frequency-Inverse Document Frequency) is used for some models for bette comparison.

Performing hyperparameter tuning using a grid search approach with **GridSearchCV** for the best selected model.
