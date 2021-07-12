# polarops.py: common functions, to be imported in other .py scripts.

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
import tensorflow.compat.v1.logging


# Suppress annoying (and red herring, apparently) warning message from TF.
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)


def create_tokenizer(lines, numTopFeatures, method, removeStopwords, 
    useBigrams, maxDf):

    if method == "tfidf":
        tokenizer = TfidfVectorizer(
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
        tokenizer = CountVectorizer(
            lowercase=True,
            # note: default stopword list evidently has disadvantages
            stop_words = 'english' if removeStopwords else None,  
            token_pattern=r"(?u)\b\w\w+\b",
            analyzer="word",
            max_df=maxDf,
            max_features=numTopFeatures,
            binary=(method=='binary'),
            ngram_range=(1,2 if useBigrams else 1))
    tokenizer.fit_transform(lines)
    return tokenizer


# define the model
def create_model(numWords, numNeurons):
    # define network
    model = Sequential()
    model.add(Dense(numNeurons, input_shape=(numWords,), activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
        metrics=['accuracy'])
    return model
