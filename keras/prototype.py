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

# The number of most-frequent words to include in analysis.
NUM_TOP_WORDS = 1000


# TJ
# Removes punctuation and capitalization from a string
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

# orig define the model
def define_model(numWords):
    # define network
    model = Sequential()
    model.add(Dense(50, input_shape=(numWords,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile network
    model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=['accuracy'])
    # summarize defined model
    #model.summary()
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

# orig/SD classify a thread as polarized or not polarized
def predict_polarized(thread, vocab, tokenizer, model):
    tokens = thread_to_tokens(thread)
    # unneeded?  tokens = [w for w in tokens if w in vocab]
    line = ' '.join(tokens)
    # encode
    encoded = tokenizer.texts_to_matrix([line], mode='binary')
    # predict polarization
    yhat = model.predict(encoded, verbose=0)
    # retrieve predicted percentage and label
    percent_polar = yhat[0,0]
    if round(percent_polar) == 0:
        return (1-percent_polar), 'no'
    return percent_polar, 'yes'

# SD compute vocabulary
def compute_vocab(df):
    vocab = Counter()
    for row in df.itertuples():
        vocab.update(thread_to_tokens(row.text))
    return [ w for w,c in vocab.most_common(NUM_TOP_WORDS) ]
        



# load the training data
df = pd.read_csv("../data_collection/training_data.csv")
vocab = compute_vocab(df)


## load all threads
training_set = df.sample(frac=.8)
test_set = df.drop(training_set.index)
training_threads, ytrain = load_clean_dataset(training_set, vocab)
test_threads, ytest = load_clean_dataset(test_set, vocab)

## create the tokenizer
tokenizer = create_tokenizer(training_threads + test_threads)
## encode data
Xtrain = tokenizer.texts_to_matrix(training_threads, mode='binary')
Xtest = tokenizer.texts_to_matrix(test_threads, mode='binary')



# define network
numWords = Xtrain.shape[1]
model = define_model(numWords)
# fit network
model.fit(Xtrain, ytrain, epochs=10, verbose=0)

results = model.predict(Xtest)[:,0].round() == ytest
print("\nThe model got {}/{} ({:.2f}%) correct.".format(
    sum(results), len(results), sum(results)/len(results)*100))

# test polarized text
text = 'I hate you libtards and i will never believe anything you say!'
percent, sentiment = predict_polarized(text, vocab, tokenizer, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
# test negative text
text = 'Thats a really good point. I think Im changing my mind.'
text = "Can't we for once have a bipartisan solution where everybody wins?"
percent, sentiment = predict_polarized(text, vocab, tokenizer, model)
print('Review: [%s]\nSentiment: %s (%.3f%%)' % (text, sentiment, percent*100))
