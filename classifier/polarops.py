# polarops.py: common functions, to be imported in other .py scripts.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
import tensorflow.compat.v1.logging
import string


# Suppress annoying (and red herring, apparently) warning message from TF.
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)


# lines: a list/Series of strings, each of which represents a thread.
# numTopFeatures: the number of most commonly occurring words/bigrams
#   which we are using in the clasifier.
# method: "binary", "count", "tfidf"
# removeStopWords: should we remove them (but see note below)
# useBigrams: if True, use both bigrams and unigrams. If False, unigrams only.
# maxDf: ignore unigrams/bigrams with document frequency higher than this
def create_vectorizer(numTopFeatures, method, removeStopwords, useBigrams,
    maxDf):

    if method == "tfidf":
        vectorizer = TfidfVectorizer(
            lowercase=True,
            # note: default stopword list evidently has disadvantages
            stop_words = 'english' if removeStopwords else None,  
            token_pattern=r"(?u)\b\w\w+\b",
            analyzer="word",
            max_df=maxDf,
            max_features=numTopFeatures,
            binary=False,   # experiment?
            ngram_range=(1,2 if useBigrams else 1))
    else:
        vectorizer = CountVectorizer(
            lowercase=True,
            # note: default stopword list evidently has disadvantages
            stop_words = 'english' if removeStopwords else None,  
            token_pattern=r"(?u)\b\w\w+\b",
            analyzer="word",
            max_df=maxDf,
            max_features=numTopFeatures,
            binary=(method=='binary'),
            ngram_range=(1,2 if useBigrams else 1))
    return vectorizer


# create an "empty" neural net of the right dimensions.
def create_model(numWords, numNeurons):
    model = Sequential()
    model.add(Dense(numNeurons, input_shape=(numWords,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=['accuracy'])
    return model


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
