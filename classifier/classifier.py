import nltk
import random
import string
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Removes punctuation and capitalization from a string
def remove_punct(post):
    punctuation = string.punctuation 
    for element in punctuation:
        post = post.replace(element, "")
    post = post.replace("’", "")
    post = post.replace("—", "")
    post = post.replace("“", "")
    post = post.replace("”", "")
    return post.lower()

# Returns the documents features
def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features

# Opens a file and reads in all the posts
#file_name = input("What file would you like to train/test?\n")
file_name = "training_data.csv"
posts = pd.read_csv(file_name, delimiter = ",")

# Initialize variables for future use
words = []
documents = []
stemmer = nltk.PorterStemmer()

# Iterate through dataframe's rows
for index in posts.index:
        # Remove punctuation/stopwords, tokenize, and stem
        legitWords = []
        post = remove_punct(posts['text'][index])
        for w in word_tokenize(post):
            if w not in stopwords.words('english'):
                legitWords.append(stemmer.stem(w))
                words.append(stemmer.stem(w))

        # Store edited string as a new labeled document
        documents.append((legitWords, posts['polarized'][index]))

# Shuffle documents
random.shuffle(documents)


all_words = nltk.FreqDist(w.lower() for w in words)
word_features = [word[0] for word in all_words.most_common()[:300]]
featuresets = [(document_features(d), c) for (d,c) in documents]


size = int(input("What n-fold cross validation would you like to use?\n"))

# Train and tests a Naive Bayes classifier using n-fold cross validation
curr = 0
count = 0
temp = 0
while curr < len(featuresets):
    train_set = []
    test_set = []
    count = count + 1
    for i in range(len(featuresets)):
        if i in range(curr, curr + size):
            train_set.append(featuresets[i])
        else:
            test_set.append(featuresets[i])
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    temp = temp + nltk.classify.accuracy(classifier, test_set)
    curr = curr + size

print(str.format("Average accuracy after {} trials: {}", count, temp/count))
