# parallel_sweep.py: a parallel version of sweep.py.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime as dt
from polarops import create_model, create_vectorizer
import os
import sys
import subprocess


# To run, set the ALL_CAPS variables below. The first is the name of the .csv
# file you would like to contain your results. The next two should be single
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

OUTPUT_CSV_FILE = "parallelResults.csv"

NUM_MODELS = 10    # Number of random models to generate for each batch of
                    # configuration settings.
TRAINING_FRAC = .8  # Fraction of rows to use as training data in each model.


# int: number of most common words/bigrams to retain
NUM_TOP_FEATURESES = [ 500, 3000 ]

# str: binary, freq, count, or tfidf
METHODS = [ 'binary', 'count' ]

# int: number of "neurons" in our only layer
NUM_NEURONSES = [ 20 ]

# int: number of neural net training epochs
NUM_EPOCHSES = [ 20 ]

# bool: remove stopwords, or leave them? (see "note" below)
REMOVE_STOPWORDSES = [ True, False ]

# bool: use bigrams, or just unigrams?
USE_BIGRAMSES = [ True, False ]

# float: ignore unigrams/bigrams with document frequency higher than this
MAX_DFS = [ .95 ]

############################################################################

if os.path.exists(OUTPUT_CSV_FILE):
    sys.exit("{} already exists! Will not overwrite.".format(OUTPUT_CSV_FILE))
else:
    with open(OUTPUT_CSV_FILE, "w", encoding="utf-8") as f:
        print("numTopFeatures,method,numNeurons,numEpochs,removeStopwords,useBigrams,maxDfs,avgAccuracy",file=f)
        f.flush()
        f.close()


# load and shuffle the training data
df = pd.read_csv("hand_tagged_data.csv")
df = df.sample(frac=1)

## load all threads and labels.
all_threads = df.text
yall = np.where(df.polarized=="yes",1,0)


curr = 1
total = len(NUM_TOP_FEATURESES) * len(METHODS) * len(NUM_NEURONSES) * \
    len(NUM_EPOCHSES) * len(REMOVE_STOPWORDSES) * len(USE_BIGRAMSES) * \
    len(MAX_DFS)

msg = "({} features, {}, {} neurons, {} epochs, {}, {}, {} maxDf)"

processes = []

for numTopFeatures in NUM_TOP_FEATURESES:
    for method in METHODS:
        for numNeurons in NUM_NEURONSES:
            for numEpochs in NUM_EPOCHSES:
                for removeStopwords in REMOVE_STOPWORDSES:
                    for useBigrams in USE_BIGRAMSES:
                        for maxDf in MAX_DFS:

                            print("\n\n*** Spawning thread for configuration" +
                                " {} of {} ***".format(curr,total))
                            print(msg.format(numTopFeatures, method,
                                numNeurons, numEpochs,
                                "removeSW" if removeStopwords else "useSW",
                                "bigrams" if useBigrams else "unigrams",
                                maxDf))
                            processes.append(subprocess.Popen(
                                ("./parallel_sweep_helper.py " +
                                    OUTPUT_CSV_FILE + " " + 
                                    str(curr) + " " + 
                                    str(NUM_MODELS) + " " + 
                                    str(TRAINING_FRAC) + " " + 
                                    str(numTopFeatures) + " " +
                                    method + " " + 
                                    str(numNeurons) + " " +
                                    str(numEpochs) + " " +
                                    str(removeStopwords) + " " +
                                    str(useBigrams) + " " +
                                    str(maxDf)).split(" ")))
                            curr += 1

[ process.wait() for process in processes ]
print("All done! Output in {}.".format(OUTPUT_CSV_FILE))
