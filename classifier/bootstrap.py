# bootstrap.py: a program to accumulate more training data by doing predictions
# on unlabeled data and accepting only those which are predicted with extreme
# confidence.
#
# After running this program, additional .csv files that will be created are
# bootstrapped_data.csv (containing only bootstrapped data) and
# training_data.csv (containing both hand tagged and bootstrapped). These will
# be in the classifier directory.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import datetime as dt
import tensorflow.compat.v1.logging
from polarops import create_model, create_vectorizer, remove_punct, get_features
import sys
from validate import get_classifier, perform_cross_validation, encode_features


############################################################################

# The number of new (unlabeled) data points to use in each iteration of the
# bootstrap process.
BOOTSTRAP_SIZE = 100

# The number of random training sets to use for each model cross-validation.
NUM_MODELS = 20

# The prediction thresholds beyond which we will consider a bootstrapped sample
# to be "safe" to use. If its prediction is < min_bound, we consider it safely
# non-polarized. If its prediction is > max_bound, we consider it safely
# polarized. Otherwise, we consider it unreliable and won't use it.
min_bound = .1
max_bound = .95

# int: number of most common words/bigrams to retain
numTopFeatures = 3000

# str: binary, freq, count, or tfidf
method = 'binary'

# int: number of "neurons" in our only layer
numNeurons = 20

# int: number of neural net training epochs
numEpochs = 20

# bool: remove stopwords, or leave them? (see "note" below)
removeStopwords = False

# bool: use bigrams, or just unigrams?
useBigrams = True

# bool: Use number of comments as a feature
comments = True

# bool: Use average number of in thread quotes as a feature
itquotes = True

# bool: Use average number of links as a feature
links = True

# bool: Use average word length as a feature
wordLength = True

# bool: Use lexical diversity as a feature
ld = True

# float: ignore unigrams/bigrams with document frequency higher than this
maxDf = .9

############################################################################


# Usage: bootstrap.py pathToUnlabeledFile.csv.
if len(sys.argv) == 2:
    BOOTSTRAP_DATA_FILE = sys.argv[1]
else:
    BOOTSTRAP_DATA_FILE = input("What is the name of the unlabeled .csv file" +
        " (from the data_collection directory, of course) with the data you" +
        " want to bootstrap?\n")

if not BOOTSTRAP_DATA_FILE.endswith('.csv'):
    BOOTSTRAP_DATA_FILE += '.csv'
if not BOOTSTRAP_DATA_FILE.startswith('../data_collection/'):
    BOOTSTRAP_DATA_FILE = "../data_collection/" + BOOTSTRAP_DATA_FILE



# load and shuffle the hand-tagged training data
ht = pd.read_csv("hand_tagged_data.csv")
del ht['ttype']
del ht['community']
ht = ht.sample(frac=1)

# load all hand-tagged threads and labels.
handTaggedThreads = ht.text
handTaggedLabels = np.where(ht.polarized=="yes",1,0)

# load and shuffle the data to bootstrap
bootstrapReservoir = pd.read_csv(BOOTSTRAP_DATA_FILE).sample(frac=1)

# Start the boostrapping process by *only* using what we've hand-tagged.
handTaggedPlus = ht.copy()
trainingData = handTaggedThreads.copy()
trainingLabels = handTaggedLabels.copy()
previousAccuracy = 0
print("\n    Initial model training...")
currAccuracy = perform_cross_validation(NUM_MODELS, trainingData,
    trainingLabels).mean()

print("\n    Starting bootstrapping...")
loop_num = 0
while previousAccuracy <= currAccuracy:

    # TJ makes another good point: what if we run out of the bootstrap
    # reservoir before we go down in accuracy? Yeah, T, handle that too.

    if len(bootstrapReservoir) < BOOTSTRAP_SIZE:
        print("No more bootstrap data!")
        break
    bootstrap = bootstrapReservoir.sample(BOOTSTRAP_SIZE)
    bootstrapReservoir = bootstrapReservoir.drop(bootstrap.index)

    classifier = get_classifier(trainingData, trainingLabels)
    bootstrapResults = classifier.predict(encode_features(bootstrap.text))
    del bootstrap['batch_num']
    toKeep = bootstrap[(bootstrapResults < min_bound) | 
        (bootstrapResults > max_bound)].copy()
    predictions = bootstrapResults[(bootstrapResults < min_bound) | 
        (bootstrapResults > max_bound)]
    
    toKeep['polarized'] = np.where(predictions < .5, 0, 1)
    handTaggedPlus = handTaggedPlus.append(toKeep)

    trainingData = np.append(trainingData, toKeep.text)
    trainingLabels = np.append(trainingLabels, toKeep['polarized'])

    previousAccuracy = currAccuracy
    currAccuracy = perform_cross_validation(NUM_MODELS, trainingData,
        trainingLabels).mean()
    if previousAccuracy > currAccuracy:
        handTaggedPlus = handTaggedPlus.drop(toKeep.index)
    else:
        print(("\n    We just added {} samples, and went from {:.2f}% to " +
            "{:.2f}% accuracy.").format(len(toKeep), previousAccuracy,
            currAccuracy))
    loop_num+= 1


# Finally, write our hand-tagged training data, with all our accuracy-
# increasing bootstapped samples, to a file.
handTaggedPlus.to_csv("hand_tagged_plus.csv",encoding="utf-8",index=False)

