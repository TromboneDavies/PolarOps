# prepare.py -- Perform basic data cleansing and model creation in preparation
# for downstream script like validate.py, cross_validate.py, or interactive.py.
#
# When this script is finished running, the following variables will be
# available to downstream scripts:
#
# - define_model()
# - all_threads
# - yall
# - vocab
# - numWords

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



# Configurable aspects of the model.
NUM_TOP_WORDS = 6000  # Number of most common words to retain.
METHOD = 'binary'
NUM_EPOCHS = 20
NUM_NEURONS = 20

# TJ - Removes punctuation and capitalization from a string
def remove_punct(thread):
    thread = thread.replace("\\n", "")
    thread = thread.replace("\t", "")
    thread = thread.replace(">>", " inthreadquote newcomment ")
    thread = thread.replace(">", " newcomment ")
    punctuation = string.punctuation 
    for element in punctuation:
        thread = thread.replace(element, "")
        thread = thread.replace("’", "")
        thread = thread.replace("—", "")
    thread = thread.replace("“", "")
    thread = thread.replace("”", "")
    return thread.lower()

# SD
def tokenize(thread):
    return thread.split(" ")
    
# orig/SD
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

# orig/SD
def load_clean_dataset(df, vocab):
    return process_threads(df, vocab), np.where(df.polarized=="yes",1,0)

# orig
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# orig - define the model
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

# orig/SD - classify a thread as polarized or not polarized
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

# SD - compute vocabulary
def compute_vocab(df, numWords):
    vocab = Counter()
    for row in df.itertuples():
        vocab.update(thread_to_tokens(row.text))
    return [ w for w,c in vocab.most_common(numWords) 
        if w not in ['newcomment','inthreadquote'] ]
        



# load and shuffle the training data
df = pd.read_csv("../classifier/training_data.csv")
df = df.sample(frac=1)
vocab = compute_vocab(df,NUM_TOP_WORDS)


## load all threads and labels.
all_threads, yall = load_clean_dataset(df, vocab)

## create the tokenizer, for use in calling .texts_to_matrix().
tokenizer = create_tokenizer(all_threads)

## encode data
allX = tokenizer.texts_to_matrix(all_threads, mode=METHOD)


# define neural net
numWords = allX.shape[1]


def validate(all_threads, yall, test_size=.2):

    # split into training and test set
    training_threads, test_threads, ytrain, ytest = train_test_split(
        all_threads, yall, test_size=test_size)

    # encode separate training and test matrices
    Xtrain = tokenizer.texts_to_matrix(training_threads, mode=METHOD)
    Xtest = tokenizer.texts_to_matrix(test_threads, mode=METHOD)

    model = define_model(numWords)
    histo = model.fit(Xtrain, ytrain, epochs=NUM_EPOCHS, verbose=0)

    return model.predict(Xtest)[:,0].round() == ytest



def validation_hist(numModels=100,title=""):
    accuracies = np.empty(numModels)
    for i in range(numModels):
        print("\nTraining model {}/{}...".format(i+1,numModels))
        results = validate(all_threads, yall)
        accuracies[i] = sum(results)/len(results)*100
    plt.clf()
    pd.Series(accuracies).hist(density=True, bins=range(0,100,4))
    ax = plt.gca()
    plt.axvline(x=accuracies.mean(),color="red")
    plt.text(x=accuracies.mean()+5,y=.9*ax.get_ylim()[1],
        s="{:.2f}%".format(accuracies.mean()),color="red")
    plt.xlabel("Accuracy (%)")
    plt.title(title)
    plt.savefig("{}_{}words_{}epochs_{}neurons.png".format(
        NUM_TOP_WORDS, METHOD, NUM_EPOCHS, NUM_NEURONS))
    return accuracies
        

def validate_one():
    results = validate(all_threads, yall)
    print("\nThe model got {}/{} ({:.2f}%) correct.".format(
        sum(results), len(results), sum(results)/len(results)*100))



print(("Validating model using {} top words, {} counting, {} epochs, " +
    " and {} 'neurons'.").format(NUM_TOP_WORDS, METHOD, NUM_EPOCHS,
    NUM_NEURONS))
validation_hist(100,"{} {} words {} epochs {} neurons".format(
    NUM_TOP_WORDS, METHOD, NUM_EPOCHS, NUM_NEURONS))

