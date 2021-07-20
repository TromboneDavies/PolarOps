# validate.py: contains functions to create and train a neural net on
# hand_tagged_data.csv, or some subset thereof, and evaluate its performance.
# Functions that may be of use outside this script include validate(),
# validation_hist(), perform_cross_validation(), get_classifier(), and
# encode_features().

import string
import re
from os import listdir
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polarops import remove_punct, get_features, create_vectorizer
from datetime import datetime

from gensim.scripts.glove2word2vec import KeyedVectors



# Configurable aspects of the model.
NUM_TOP_WORDS = 3000  # Number of most common words to retain.
METHOD = 'binary'     # binary, freq, count, or tfidf
NUM_EPOCHS = 20
NUM_NEURONS = 20
REMOVE_STOPWORDS = False     # Remove stopwords?
EMBEDDINGS = False           # Use word embeddings, or just one-hot words?
#WORD_EMBEDDINGS_FILE = "Set1_TweetDataWithoutSpam_Word.bin"
WORD_EMBEDDINGS_FILE = "glove.6B.100d.w2v.txt"




def tokenize(thread):
    return [ w for w in thread.split(" ") if w not in stop ]
    
def thread_to_tokens(thread, vocab=None):
    tokens = tokenize(remove_punct(thread))
    if vocab:
        tokens = [w for w in tokens if w in vocab]
    return tokens

# Return a list of lists, each list of which is a tokenized thread. The outer
# list will contain all training documents in the DataFrame passed.
def process_threads(df, vocab):
    lines = list()
    for row in df.itertuples():
        lines.append(thread_to_tokens(row.text, vocab))
    return lines

def load_clean_dataset(df, vocab):
    return process_threads(df, vocab), np.where(df.polarized=="yes",1,0)

def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def define_model(numWords, numNeurons=NUM_NEURONS):
    # define network
    model = Sequential()
    model.add(Dense(numNeurons, input_shape=(numWords,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=['accuracy'])
    # summarize defined model
    #model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


def predict_polarized(thread, vocab, tokenizer, model):
    tokens = thread_to_tokens(thread)
    # unneeded?  tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode=METHOD)
    # predict polarization
    yhat = model.predict(encoded, verbose=0)
    # retrieve predicted percentage and label
    percent_polar = yhat[0,0]
    if round(percent_polar) == 0:
        return (1-percent_polar), 'no'
    return percent_polar, 'yes'


def compute_vocab(df, numWords):
    vocab = Counter()
    for row in df.itertuples():
        vocab.update(thread_to_tokens(row.text))
    return [ w for w,c in vocab.most_common(numWords) 
        if w not in ['newcomment','inthreadquote'] ]


# Compute the centroid (Stephen's "dumb idea")
def embed(threads):
    ignored = set()
    mat = np.zeros(shape=(len(threads),wv.vector_size))
    for i,thread in enumerate(threads):
        if METHOD == "binary":
            words = set(thread)
        elif METHOD in [ "freq", "count" ]:
            words = thread
        for word in words:
            if word in wv and word in vocab:
                mat[i,:] += wv[word]
            else:
                ignored |= {word}
        if METHOD == "freq":
            mat[i,:] = mat[i,:] / len(words)
    print("(Ignoring these non-words: {}.)".format(", ".join(ignored)))
    return mat
        


# Given the training data passed, return a neural net on which .predict() can
# be called to make predictions on new data. (The argument to .predict() should
# be a matrix each row of which is a set of features for one thread. This
# matrix can be creatd with encode_features().
def get_classifier(threads, labels, comments=False, itquotes=False,
    links=False, wordLength=False, ld=False):

    model = define_model(numWords+5)   # "5" because of TJ features
    Xtrain = get_features(encode_features(threads), threads, comments,
         itquotes, links, wordLength, ld)
    model.fit(Xtrain, labels, epochs=NUM_EPOCHS, verbose=0)
    return model



# Given a set of threads, return a matrix of features.
def encode_features(threads):
    if EMBEDDINGS:
        return embed(threads)
    else:
        try:
            return tokenizer.texts_to_matrix(threads, mode=METHOD)
        except AttributeError:
            sys.exit("Could not encode_features({})!".format(threads))


# Train the model one time on a randomly-chosen training set from the data set
# passed, and return the correctness of its predictions on all elements of the
# remaining data points (i.e., the test set).
# "all_threads" contains the features, and "yall" contains the labels.
def validate(all_threads, yall, comments=False, itquotes=False,
    links=False, wordLength=False, ld=False, test_frac=.2):

    # split into training and test set
    training_threads, test_threads, ytrain, ytest = train_test_split(
        all_threads, yall, test_size=test_frac)

    # encode separate training and test matrices
    model = get_classifier(training_threads, ytrain)
    Xtest = get_features(encode_features(test_threads), test_threads,
        comments, itquotes, links, wordLength, ld)

    return model.predict(Xtest)[:,0].round() == ytest



# Train the model on numModels randomly-chosen training sets, and return an
# array of the models' accuracies (as a percentage).
def perform_cross_validation(numModels,all_threads,yall,
    comments=False, itquotes=False, links=False, wordLength=False, ld=False,
    test_frac=.2):

    accuracies = np.empty(numModels)
    for i in range(numModels):
        print("\nTraining model {}/{}...".format(i+1,numModels))
        results = validate(all_threads, yall, comments, itquotes,
            links, wordLength, ld, test_frac)
        accuracies[i] = sum(results)/len(results)*100
    return accuracies



# Train the model on numModels randomly-chosen training sets, and plot a
# histogram of the models' accuracies (as a percentage). The filename of the
# plot will be a concatenation of the model parameters, followed by ".png".
def validation_hist(numModels=100,title=""):
    accuracies = perform_cross_validation(numModels,all_threads,yall)
    plt.clf()
    pd.Series(accuracies).hist(density=True, bins=range(0,100,4))
    ax = plt.gca()
    plt.axvline(x=accuracies.mean(),color="red")
    plt.text(x=accuracies.mean()+5,y=.9*ax.get_ylim()[1],
        s="{:.2f}%".format(accuracies.mean()),color="red")
    plt.xlabel("Accuracy (%)")
    if EMBEDDINGS:
        def_title = "{} {} words {} {}".format(
            WORD_EMBEDDINGS_FILE[:4], NUM_TOP_WORDS, METHOD,
                "removing" if REMOVE_STOPWORDS else "keeping")
    else:
        def_title = "Raw {} words {} {}".format(
            NUM_TOP_WORDS, METHOD,
                "removing" if REMOVE_STOPWORDS else "keeping")
    title = title if len(title) > 0 else def_title
    plt.title(title)
    plt.savefig(title.replace(" ","_") + ".png")
        

def validate_one():
    results = validate(all_threads, yall)
    print("\nThe model got {}/{} ({:.2f}%) correct.".format(
        sum(results), len(results), sum(results)/len(results)*100))


# createMythicalGraph: Create a plot called "mythicalGraph.png" in the current
# directory, which contains a time series plot for each subreddit in the list
# of subreddit names passed.
def createMythicalGraph(handTaggedPlusDataFrame, subredditNames):
    mdf = createMythicalDataFrame(handTaggedPlusDataFrame, subredditNames)
    createMythicalGraphHelper(mdf)


def createMythicalGraphHelper(mdf):
    ax = plt.gca()
    for sub in mdf.subreddit.unique():
        mdf[mdf.subreddit==sub].plot(kind='line',
            x='year', y='perc_polar', ax=ax, label=sub)
    plt.ylim(-5,105)
    plt.ylabel("% of polarized threads")
    plt.legend()
    plt.title("Estimated polarization of subreddits over time")
    plt.show()
    plt.savefig("mythicalGraph.png")


# createMythicalDataFrame: produce a DataFrame with three columns: subreddit,
# year, and perc_polar, for the training data and list of subreddit names
# passed.
#
# Example usage:
# mdf = createMythicalDataFrame(pd.read_csv("hand_tagged_plus.csv"),
#    ['congress','politics','bannedfromthe_donald','capitalism'])
#
# handTaggedPlusDataFrame: the DataFrame corresponding to the final training
# set (hand-tagged plus bootstrapped) that we are going to use, baby.
# subredditNames: a list of names of subreddits, each of which should have a
# .csv file in the data_collection directory.
def createMythicalDataFrame(handTaggedPlusDataFrame, subredditNames):
    years = np.array([],dtype=int)
    subredditColumn = np.array([],dtype=object)
    perc_polars = np.array([],dtype=float)
    with open("mythicalDataFrame.csv","w",encoding="utf-8") as f:
        print("subreddit,year,perc_polar",file=f)
        f.flush()
        for subredditName in subredditNames:
            subredditName = subredditName.lower()
            print("Predicting for {}...".format(subredditName))
            if not subredditName.endswith('.csv'):
                subredditFileName = subredditName + '.csv'
            else:
                subredditFileName = subredditName
            subredditFileName = "../data_collection/" + subredditFileName
            subredditDf = pd.read_csv(subredditFileName, encoding='utf-8')
            subredditDf['year'] = \
                subredditDf.date.astype(int).astype("datetime64[s]").dt.year
            for year in sorted(subredditDf.year.unique()):
                print("    year {}...".format(year))
                threads = subredditDf[subredditDf.year == year].text
                results = classify(threads, handTaggedPlusDataFrame.text,
                    handTaggedPlusDataFrame.polarized,
                    6000, "count", False, True, .95, False, False, False, False,
                    True, 20, 20)
                percent_polarized = (results > .5).sum() / len(results) * 100
                years = np.append(years, year)
                subredditColumn = np.append(subredditColumn, subredditName)
                perc_polars = np.append(perc_polars, percent_polarized)
                print("{},{},{}".format(subredditName,year,percent_polarized),
                    file=f)
                f.flush()
    masterDataFrame = pd.DataFrame({'subreddit':subredditColumn,
        'year':years, 'perc_polar':perc_polars})
    return masterDataFrame


# toClassify: A list/Series of strings, each of which is a thread to classify.
# trainingData: A list/Series of strings, each of which is a training thread.
# trainingLabels: A list of the corresponding polarization values for each of
#   the training data, as "yes"/"no" strings
#
# To run this, Alexis, do this:
# 
#   classify(the texts of the threads for a particular year and subreddit,
#       the text column of your hand_tagged_plus.csv dataframe,
#       the polarized column of your hand_tagged_plus.csv dataframe,
#       6000, "count", False, True, .95, False, False, False, False, True,
#       20, 20)
# You will get back a np array of numbers between 0 and 1.
# Use an np.where() on that to convert to "yes" and "nos".
# Divide to get the "percentage polarized."
# Done. :D
def classify(toClassify, trainingData, trainingLabels, numTopFeatures, method,
    removeStopwords, useBigrams, maxDf, useComments, useItQuotes, useLinks,
    useWordLength, useLd, numNeurons, numEpochs):

    # Create a vectorizer which will use only the training data, not the new
    # data.
    training_v = create_vectorizer(numTopFeatures, method, removeStopwords,
        useBigrams, maxDf)

    # Go ahead and vectorize the training data.
    training_vectorized = training_v.fit_transform(trainingData).toarray()

    # Save the vectorizer's vocabulary so it can be used with the new data,
    # which will have different words and frequencies, of course.
    training_vocab = training_v.vocabulary_

    # Add TJ's (mostly worthless, as it turns out) extra features.
    training_vectorizedPlusTJ = get_features(training_vectorized,
        trainingData, useComments, useItQuotes, useLinks, useWordLength, useLd)

    # Now create a new vectorizer (weird, I know) which will use the previous
    # vectorizer's vocabulary. This is so that the new documents are encoded in
    # precisely the same way
    new_v = create_vectorizer(numTopFeatures, method, removeStopwords,
        useBigrams, maxDf, vocabulary=training_v.vocabulary_)
    new_vectorized = new_v.fit_transform(toClassify).toarray()
    new_vectorizedPlusTJ = get_features(new_vectorized, toClassify,
        useComments, useItQuotes, useLinks, useWordLength, useLd)

    trainingLabelsCoded = np.where(np.array(trainingLabels) == "yes", 1, 0)

    model = define_model(numTopFeatures+5, numNeurons) # "5" because of TJ
    model.fit(training_vectorizedPlusTJ, trainingLabelsCoded, epochs=numEpochs,
        verbose=0)

    results = model.predict(new_vectorizedPlusTJ)
    return results


if REMOVE_STOPWORDS:
    stop = [ remove_punct(sw) for sw in stopwords.words('english') ]
else:
    stop = []

if EMBEDDINGS:
    print("Loading word embeddings...")
    binary = False if WORD_EMBEDDINGS_FILE.endswith(".txt") else True
    wv = KeyedVectors.load_word2vec_format(WORD_EMBEDDINGS_FILE, binary=binary)
    print("...done.")


# Load and shuffle the training data
df = pd.read_csv("hand_tagged_data.csv")
df = df.sample(frac=1)
vocab = compute_vocab(df,NUM_TOP_WORDS)


# Load all threads and labels.
all_threads, yall = load_clean_dataset(df, vocab)
if EMBEDDINGS:
    allX = embed(all_threads)
    numWords = allX.shape[1]
else:
    # create the tokenizer, for use in calling .texts_to_matrix().
    tokenizer = create_tokenizer(all_threads)
    # encode data
    allX = tokenizer.texts_to_matrix(all_threads, mode=METHOD)
numWords = allX.shape[1]


#validation_hist(100)
