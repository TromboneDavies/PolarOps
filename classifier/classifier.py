import nltk
import random
import string
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv

#Creates a list of lists (which are the individual bot names with /u/)
with open('botnames.csv',newline='') as f:
    reader=csv.reader(f)
    botnames=list(reader)
#Turning that list of lists into a list of strings
botnames = [item for sublist in botnames for item in sublist]

# Removes punctuation and capitalization from a string
def remove_punct(post):
    post = post.replace("\\n", "")
    post = post.replace(">>", " inthreadquote newcomment ")
    post = post.replace(">", " newcomment ")
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
    temp = 0
    links = 0
    num_comments = 0
    num_in_thread_quotes = 0
    for word in word_features:
        features[word] = (word in document_words)
    for word in document:
        if "newcom" in word:
            num_comments = num_comments + 1
        elif "inthreadquot" in word:
            num_in_thread_quotes = num_in_thread_quotes + 1
        elif "http" in word:
            links = links + 1
        else:
            temp = temp + len(word)

    features["Average Word Length"] = temp/len(document)

    if num_comments == 0:
        num_comments = 1

    features["Frequency of Links"] = links/num_comments
    #features["Number of Comments"] = num_comments

    features["Frequency of in-thread quotes"] = num_in_thread_quotes/num_comments
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
            test_set.append(featuresets[i])
        else:
            train_set.append(featuresets[i])
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    temp = temp + nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(10)
    curr = curr + size

print(str.format("Average accuracy after {} trials: {}", count, temp/count))
