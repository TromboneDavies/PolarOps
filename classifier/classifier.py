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


size = int(len(featuresets)/2)

# Train and tests a Naive Bayes classifier
train_set = featuresets[int(size * .7):]
test_set = featuresets[:int(size * .3)]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Print classifier accuracy and most informative features
print(nltk.classify.accuracy(classifier, test_set))
classifier.show_most_informative_features(10)
