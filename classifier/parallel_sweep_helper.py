#!/usr/bin/env python3
# parallel_sweep_helper.py: this program is NOT designed to be called directly.
# It will be launched by parallel_sweep.py so that multiple configuration
# settings can be evaluated in parallel. (See parallel_sweep.py for usage.)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import datetime as dt
from polarops import create_model, create_vectorizer, get_features
import sys
import nltk 

if len(sys.argv) != 18:
    sys.exit(
        '''
        Usage: parallel_sweep_helper.py
                    OUTPUT_CSV_FILE
                    CONFIG_NUM
                    NUM_MODELS (int)
                    TRAINING_FRAC (float)
                    NUM_TOP_FEATURES (int)
                    METHOD ('binary', 'freq', 'count', or 'tfidf')
                    NUM_NEURONS (int)
                    NUM_EPOCHS (int)
                    REMOVE_STOPWORDS (true/false)
                    USE_BIGRAMS (true/false)
                    MAX_DFS (float)
                    STEMMING (true/false).
        '''
    )

OUTPUT_CSV_FILE = sys.argv[1]
if not OUTPUT_CSV_FILE.endswith(".csv"):
    OUTPUT_CSV_FILE += ".csv"

CONFIG_NUM = int(sys.argv[2])   # Just for printing progress.

NUM_MODELS = int(sys.argv[3])   # Number of random models to generate for
                                # each batch of configuration settings.

TRAINING_FRAC = float(sys.argv[4])  # Fraction of rows to use as training
                                    # data in each model.

numTopFeatures = int(sys.argv[5]) # number of most common words/bigrams to
                                    # retain.

method = sys.argv[6]

numNeurons = int(sys.argv[7])   # number of "neurons" in our only layer

numEpochs = int(sys.argv[8])    # number of neural net training epochs

removeStopwords = sys.argv[9].lower() == "true"

useBigrams = sys.argv[10].lower() == "true"

comments = sys.argv[11].lower() == "true"

itquotes = sys.argv[12].lower() == "true"

links = sys.argv[13].lower() == "true"

wordLength = sys.argv[14].lower() == "true"

ld = sys.argv[15].lower() == "true"


maxDf = float(sys.argv[17])   # ignore unigrams/bigrams with document
                                # frequency higher than this
stemming = sys.argv[16].lower() == "true"


############################################################################



# load and shuffle the training data
df = pd.read_csv("hand_tagged_data.csv")
df = df.sample(frac=1)

## load all threads and labels.
all_threads = df.text
yall = np.where(df.polarized=="yes",1,0)

def stem(thread):
    stemmer = nltk.PorterStemmer()
    thread=[stemmer.stem(w) for w in thread]
    return thread
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
    comments,           # bool: use number of comments
    itquotes,           # bool: use frequency of in-thread quotes
    links,              # bool: use frequency of links
    wordLength,         # bool: use average word length
    ld,
    maxDf,
    stemming):             # float: ignore items above this document frequency

    vectorizer = create_vectorizer(numTopFeatures, method,
        removeStopwords, useBigrams, maxDf)
    if stemming:
        all_vectorized = vectorizer.fit_transform(stem(all_threads)).toarray()
    else:
       all_vectorized = vectorizer.fit_transform(all_threads).toarray() 
    print(" ")
    print(all_vectorized)
    print(all_threads)
    print(comments)
    print(itquotes)
    print(links)
    print(wordLength)
    print(ld)
    all_vectorized = get_features(all_vectorized, all_threads, comments, itquotes, links, wordLength,ld)

    accuracies = np.empty(NUM_MODELS)

    for i in range(NUM_MODELS):
        print("Configuration {:3d}: training model{:3d}/{:3d}...".
            format(CONFIG_NUM,i+1,NUM_MODELS))
        results = validate(all_vectorized, yall)
        accuracies[i] = sum(results)/len(results)*100
    plt.figure()
    pd.Series(accuracies).hist(density=True, bins=range(0,100,4))
    ax = plt.gca()
    plt.axvline(x=accuracies.mean(),color="red")
    plt.text(x=accuracies.mean()+5,y=.9*ax.get_ylim()[1],
        s="{:.2f}%".format(accuracies.mean()),color="red")
    plt.xlabel("Accuracy (%)")
    title = "Raw top {} {} {} {} {}n {}e {}maxDf {} {} {} {} {} {}".format(
        numTopFeatures, "bigrams" if useBigrams else "unigrams",
            method, "removeSW" if removeStopwords else "keepSW",
            numNeurons, numEpochs, maxDf, "stemmed" if stemming else "not stemmed", 
            "num comments" if comments else "no num comments", "quotes" if itquotes else "no quotes",
            "links" if links else "no links", "wordlength" if wordLength else "no wordlength",
            "ld" if ld else "no ld")
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

acc = evaluate_settings(numTopFeatures, method, numNeurons, numEpochs, removeStopwords, useBigrams,comments, itquotes,links, wordLength,ld, maxDf, stemming)

with open(OUTPUT_CSV_FILE, "a", encoding="utf-8") as f:
    f.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(numTopFeatures, method,
        numNeurons, numEpochs, removeStopwords, useBigrams,comments, itquotes,links,wordLength,ld, maxDf,stemming, acc.mean()))
    f.flush()
    f.close()
    
