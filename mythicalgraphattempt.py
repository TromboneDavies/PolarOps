import nltk
import random
import string
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv

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

classifier = nltk.NaiveBayesClassifier.train(featuresets)
#Classifier trained on 100% of the training_data.csv


# Attempting to classify congress.csv

# Opens a Congress file and reads in all the posts
congress = "congress.csv"
posts = pd.read_csv(congress, delimiter = ",")

# Re-Initialize variables for future use
words = []
documents = []

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
        documents.append(legitWords)
        
all_words = nltk.FreqDist(w.lower() for w in words)
word_features = [word[0] for word in all_words.most_common()[:300]]
featuresets = [document_features(d) for d in documents] 

#Getting the dataframe with the year for graph purposes
cdf=pd.read_csv(congress)
dtinfo = cdf.date.astype(int).astype("datetime64[s]")
cdf['year'] = dtinfo.dt.year.astype(int) 

#Classifying the threads and adding them to the dataframe
classifier_labels = classifier.classify_many(featuresets)
cdf['class_labels'] = classifier_labels

#Contingency table to make graph
year_label_m=pd.crosstab(cdf.year,cdf.class_labels,margins=True)
ylm_r=year_label_m.div(year_label_m['All'],axis=0)*100
#Make it a data frame for plotting purposes, remove 'All' row
ylm_r=(pd.DataFrame(ylm_r)).drop(['All'])

import matplotlib.pyplot as plt
plt.plot(ylm_r.index.values, ylm_r.yes)

