# bootstrap.py: a program to accumulate more training data by doing predictions
# on unlabeled data and accepting only those which are predicted with extreme
# confidence.
#
# After running this program, two additional .csv files will be created with
# the same name as the file given, but with "_polar" and "_nonpolar" appended
# to the names. These will contain threads deemed "safely polar/non-polar,"
# respectively, according to the thresholds min_bound and max_bound (see
# below).

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import datetime as dt
import tensorflow.compat.v1.logging
from polarops import create_model, create_vectorizer
import sys


############################################################################

# Usage: bootstrap.py pathToUnlabeledFile.csv.
if len(sys.argv) == 2:
    BOOTSTRAP_DATA_FILE = sys.argv[1]
else:
    BOOTSTRAP_DATA_FILE = input("What is the path to the unlabeled .csv file" +
        " with the data you want to bootstrap? (Hint: probably" +
        " ../data_collection/something.csv")

if not BOOTSTRAP_DATA_FILE.endswith('.csv'):
    BOOTSTRAP_DATA_FILE += '.csv'

# The prediction thresholds beyond which we will consider a bootstrapped sample
# to be "safe" to use. If its prediction is < min_bound, we consider it safely
# non-polarized. If its prediction is > max_bound, we consider it safely
# polarized. Otherwise, we consider it unreliable and won't use it.
min_bound = .1
max_bound = .95

# int: number of most common words/bigrams to retain
numTopFeatures = 5000

# str: binary, freq, count, or tfidf
method = 'count'

# int: number of "neurons" in our only layer
numNeurons = 20

# int: number of neural net training epochs
numEpochs = 20

# bool: remove stopwords, or leave them? (see "note" below)
removeStopwords = False

# bool: use bigrams, or just unigrams?
useBigrams = True

# float: ignore unigrams/bigrams with document frequency higher than this
maxDf = .9

############################################################################



# load and shuffle the hand-tagged training data
ht = pd.read_csv("../classifier/training_data.csv")
ht = ht.sample(frac=1)

## load all hand-tagged threads and labels.
handTaggedThreads = ht.text
handTaggedLabels = np.where(ht.polarized=="yes",1,0)

# load the data to bootstrap
bootstrap = pd.read_csv(BOOTSTRAP_DATA_FILE)
bootstrapThreads = bootstrap.text

allThreads = np.append(handTaggedThreads,bootstrapThreads)

vectorizer = create_vectorizer(numTopFeatures, method,
    removeStopwords, useBigrams, maxDf)
allVectorized = vectorizer.fit_transform(allThreads).toarray()

def get_features(currFeatures, threads):
    document_words = set(document)
    features = {}
    temp = 0
    links = 0
    num_comments = 0
    num_in_thread_quotes = 0

    listToAdd = []

    for thread in threads:
        ready_thread = remove_punct(thread)
        for word in ready_thread.split(" "):
            if "newcom" in word:
                num_comments = num_comments + 1
            elif "inthreadquot" in word:
                num_in_thread_quotes = num_in_thread_quotes + 1
            elif "http" in word:
                links = links + 1
            else:
                temp = temp + len(word)
            
        listToAdd.append([temp/len(document), num_comments, links/num_comments,
            num_in_thread_quotes/num_comments])

    features = np.append(currFeatures, listToAdd)
    return features

# Create a "blank" neural net with the right number of dimensions.
model = create_model(allVectorized.shape[1], numNeurons)

# Train the neural net for numEpochs epochs on the training data.
histo = model.fit(allVectorized[0:len(handTaggedThreads),:],
    handTaggedLabels, epochs=numEpochs, verbose=0)

bootstrap['prediction'] = model.predict(
    allVectorized[len(handTaggedThreads):,:])[:,0]

safelyNonpolarized = bootstrap[bootstrap.prediction < min_bound]
safelyPolarized = bootstrap[bootstrap.prediction > max_bound]

safelyNonpolarized.to_csv(BOOTSTRAP_DATA_FILE.replace(".csv","_nonpolar.csv"),
    index=False, encoding="utf-8")
safelyPolarized.to_csv(BOOTSTRAP_DATA_FILE.replace(".csv","_polar.csv"),
    index=False, encoding="utf-8")