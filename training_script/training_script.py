import os
import re
import sys
import nltk
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier

def preproccess_text(text_messages):
    # Remove punctuation
    processed = re.sub(r'[.,\/#!%\^&\*;:+{}=\-_`~()?]', ' ', text_messages)

    # Replace whitespace between terms with a single space
    processed = re.sub(r'\s+', ' ', processed)

    # Remove leading and trailing whitespace
    processed = re.sub(r'^\s+|\s+?$', '', processed)
    return processed

def preprocess_df(text_messages):
    # Remove punctuation
    processed = text_messages.str.replace(r'[.,\/#!%\^&\*;:{}=\-_`~()?]', ' ')

    # Replace whitespace between terms with a single space
    processed = processed.str.replace(r'\s+', ' ')

    # Remove leading and trailing whitespace
    processed = processed.str.replace(r'^\s+|\s+?$', '')
    
    return processed

def tokenizer(message):
    words = word_tokenize(message)
    tokenized_words = {}
    for word in words:
        tokenized_words[word] = word

    return tokenized_words

df = pd.read_csv('dataset/data.csv')

classes = df['rating']
sentiment_data = preprocess_df(df['reviewTitle'])

encoder = LabelEncoder()
Y = encoder.fit_transform(classes)

# Now lets do it for all the review
review = list(zip(sentiment_data, Y))

# define a seed for reproducibility
seed = 1
np.random.seed = seed
np.random.shuffle(review)

# call find_features function for each SMS message
training = [(tokenizer(text), label) for (text, label) in review]

from sklearn import model_selection

# Define models to train
names = ["K Nearest Neighbors", "Decision Tree", "Random Forest", "Logistic Regression", "SGD Classifier",
         "Naive Bayes", "SVM Linear"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    LogisticRegression(),
    SGDClassifier(max_iter = 100),
    MultinomialNB(),
    SVC(kernel = 'linear')
]

models = zip(names, classifiers)

negative_msg = 0
neutral_msg = 0
positive_msg = 0

for name, model in models:
    nltk_model = SklearnClassifier(model)
    classifier = nltk_model.train(training)
    f = open('../sentiment_analysis_pickle/' + name + ' Classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close
    result = classifier.classify(tokenizer(preproccess_text('bagus banget recommended')))
    
    if result == 0 or result ==1:
        negative_msg+=1
    elif result == 2:
        neutral_msg+=1
    elif result == 3 or result == 4:
        positive_msg+=1
    
if negative_msg >= neutral_msg and negative_msg >= positive_msg:
    best_result = "negative review"
    confidence = negative_msg / (negative_msg + neutral_msg + positive_msg)
elif neutral_msg >= negative_msg and neutral_msg >= positive_msg:
    best_result = "neutral review"
    confidence = neutral_msg / (negative_msg + neutral_msg + positive_msg)
elif positive_msg >= negative_msg and positive_msg >= neutral_msg:
    best_result = "positive review"
    confidence = positive_msg / (negative_msg + neutral_msg + positive_msg)

print("Algorithm Confidence = {}".format(confidence*100))
print("Model thinks this is a {}".format(best_result))