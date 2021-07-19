# validate.py: contains functions to create and train a neural net on
# hand_tagged_data.csv, or some subset thereof, and evaluate its performance.
# Functions that may be of use outside this script include validate(),
# validation_hist(), perform_cross_validation(), and get_classifier(),
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
from polarops import remove_punct

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

def define_model(numWords):
    # define network
    model = Sequential()
    model.add(Dense(NUM_NEURONS, input_shape=(numWords,), activation='relu'))
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
def get_classifier(threads, labels):
    model = define_model(numWords)
    Xtrain = encode_features(threads)
    model.fit(Xtrain, labels, epochs=NUM_EPOCHS, verbose=0)
    return model



# Given a set of threads, return a matrix of features.
def encode_features(threads):
    if EMBEDDINGS:
        return embed(threads)
    else:
        return tokenizer.texts_to_matrix(threads, mode=METHOD)


# Train the model one time on a randomly-chosen training set from the data set
# passed, and return the correctness of its predictions on all elements of the
# remaining data points (i.e., the test set).
# "all_threads" contains the features, and "yall" contains the labels.
def validate(all_threads, yall, test_frac=.2):

    # split into training and test set
    training_threads, test_threads, ytrain, ytest = train_test_split(
        all_threads, yall, test_size=test_frac)

    # encode separate training and test matrices
    model = get_classifier(training_threads, ytrain)
    Xtest = encode_features(test_threads)

    return model.predict(Xtest)[:,0].round() == ytest



# Train the model on numModels randomly-chosen training sets, and return an
# array of the models' accuracies (as a percentage).
def perform_cross_validation(numModels,all_threads,yall,test_frac=.2):
    accuracies = np.empty(numModels)
    for i in range(numModels):
        print("\nTraining model {}/{}...".format(i+1,numModels))
        results = validate(all_threads, yall, test_frac)
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
