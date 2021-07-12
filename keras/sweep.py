# sweep.py: a program to systematically run through various sets of
# configuration settings, testing each one on multiple models to gauge
# accuracy.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime as dt
from polarops import create_model, create_vectorizer


# To run, set the ALL_CAPS variables below. The first two should be single
# numbers, which control how many models and how big the training set should be
# each time. The remaining ones should be *lists* of values that you want to
# experiment with. Note that all combinations of all values will be run, so
# don't make all the lists long or it'll be a combinatorial explosion!
#
# After running sweep.py (and waiting a while), three things will happen:
# (1) In your home directory will be an accuracy histogram plot for every 
# settings configuration (i.e., for every combination of values) with an
# appropriate filename.
# (2) The "results" DataFrame will contain the average accuracy for each of
# these settings.
# (3) The results DataFrame will also be written to disk, in a .csv file whose
# name begins with "results_" and which has a date stamp following that.

############################################################################

NUM_MODELS = 10    # Number of random models to generate for each batch of
                    # configuration settings.
TRAINING_FRAC = .8  # Fraction of rows to use as training data in each model.


# int: number of most common words/bigrams to retain
NUM_TOP_FEATURESES = [ 100 ]

# str: binary, freq, count, or tfidf
METHODS = [ 'count', 'binary' ]

# int: number of "neurons" in our only layer
NUM_NEURONSES = [ 20 ]

# int: number of neural net training epochs
NUM_EPOCHSES = [ 20 ]

# bool: remove stopwords, or leave them? (see "note" below)
REMOVE_STOPWORDSES = [ True ]

# bool: use bigrams, or just unigrams?
USE_BIGRAMSES = [ True, False ]

# float: ignore unigrams/bigrams with document frequency higher than this
MAX_DFS = [ .5,.95 ]

############################################################################



# load and shuffle the training data
df = pd.read_csv("../classifier/training_data.csv")
df = df.sample(frac=1)

## load all threads and labels.
all_threads = df.text
yall = np.where(df.polarized=="yes",1,0)


# Run a suite of NUM_MODELS random models for a particular set of configuration
# settings. Produce a histogram on disk with an appropriate name, and return an
# array of NUM_MODELS accuracies.
def evaluate_settings(
    numTopFeatures,     # int: number of most common words/bigrams to retain
    method,             # str: binary, freq, count, or tfidf
    numNeurons,         # int 
    numEpochs,          # int
    removeStopwords,    # bool: remove stopwords?
    useBigrams,         # bool: use bigrams, or just unigrams?
    maxDf):             # float: ignore items above this document frequency

    vectorizer = create_vectorizer(numTopFeatures, method,
        removeStopwords, useBigrams, maxDf)
    all_vectorized = vectorizer.fit_transform(all_threads).toarray()

    accuracies = np.empty(NUM_MODELS)

    for i in range(NUM_MODELS):
        print("Training model {}/{}...".format(i+1,NUM_MODELS))
        results = validate(all_vectorized, yall)
        accuracies[i] = sum(results)/len(results)*100
    plt.figure()
    pd.Series(accuracies).hist(density=True, bins=range(0,100,4))
    ax = plt.gca()
    plt.axvline(x=accuracies.mean(),color="red")
    plt.text(x=accuracies.mean()+5,y=.9*ax.get_ylim()[1],
        s="{:.2f}%".format(accuracies.mean()),color="red")
    plt.xlabel("Accuracy (%)")
    title = "Raw top {} {} {} {} {}n {}e {}maxDf".format(
        numTopFeatures, "bigrams" if useBigrams else "unigrams",
            method, "removeSW" if removeStopwords else "keepSW",
            numNeurons, numEpochs, maxDf)
    plt.title(title)
    plt.savefig(title.replace(" ","_") + ".png")
    return accuracies



def validate(all_vectorized, yall, numNeurons=20, numEpochs=20):

    # split into training and test set
    training_vthreads, test_vthreads, ytrain, ytest = train_test_split(
        all_vectorized, yall, test_size=(1-TRAINING_FRAC))

    model = create_model(all_vectorized.shape[1], numNeurons)
    histo = model.fit(training_vthreads, ytrain, epochs=numEpochs, verbose=0)

    return model.predict(test_vthreads)[:,0].round() == ytest


numTopFeaturesColumn = np.array([],dtype=int)
methodsColumn = np.array([],dtype="U10")
numNeuronsColumn = np.array([],dtype=int)
numEpochsColumn = np.array([],dtype=int)
removeStopwordsColumn = np.array([],dtype=bool)
useBigramsColumn = np.array([],dtype=bool)
maxDfsColumn = np.array([],dtype=float)
avgAccColumn = np.array([],dtype=float)

curr = 1
total = len(NUM_TOP_FEATURESES) * len(METHODS) * len(NUM_NEURONSES) * \
    len(NUM_EPOCHSES) * len(REMOVE_STOPWORDSES) * len(USE_BIGRAMSES) * \
    len(MAX_DFS)

msg = "({} features, {}, {} neurons, {} epochs, {}, {}, {} maxDf)"

for numTopFeatures in NUM_TOP_FEATURESES:
    for method in METHODS:
        for numNeurons in NUM_NEURONSES:
            for numEpochs in NUM_EPOCHSES:
                for removeStopwords in REMOVE_STOPWORDSES:
                    for useBigrams in USE_BIGRAMSES:
                        for maxDf in MAX_DFS:

                            print("\n\n***** Evaluating configuration " +
                                "{} of {} *****".format(curr,total))
                            print(msg.format(numTopFeatures, method,
                                numNeurons, numEpochs,
                                "removeSW" if removeStopwords else "useSW",
                                "bigrams" if useBigrams else "unigrams",
                                maxDf))

                            acc = evaluate_settings(
                                numTopFeatures,
                                method,
                                numNeurons,
                                numEpochs,
                                removeStopwords,
                                useBigrams,
                                maxDf)
                            numTopFeaturesColumn = np.append(
                                numTopFeaturesColumn, numTopFeatures)
                            methodsColumn = np.append(
                                methodsColumn, method)
                            numNeuronsColumn = np.append(
                                numNeuronsColumn, numNeurons)
                            numEpochsColumn = np.append(
                                numEpochsColumn, numEpochs)
                            removeStopwordsColumn = np.append(
                                removeStopwordsColumn, removeStopwords)
                            useBigramsColumn = np.append(
                                useBigramsColumn, useBigrams)
                            maxDfsColumn = np.append(
                                maxDfsColumn, maxDf)
                            avgAccColumn = np.append(
                                avgAccColumn, acc.mean())

                            curr += 1
                            
results = pd.DataFrame({
    'numTopFeatures':numTopFeaturesColumn,
    'method':methodsColumn,
    'numNeurons':numNeuronsColumn,
    'numEpochs':numEpochsColumn,
    'removeStopwords':removeStopwordsColumn,
    'useBigrams':useBigramsColumn,
    'maxDfs':maxDfsColumn,
    'avgAccuracy':avgAccColumn})

results.to_csv("results_{}.csv".format(
    str(dt.date.today()).replace(" ","_")),
        index=False, encoding="utf-8")

