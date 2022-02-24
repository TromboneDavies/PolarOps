# Do the entire process from loading data to building and evaluating the
# classifier. Command-line args may include:
# seed= random (not np.random) number generator seed
# train_frac= fraction of data to use for training
# feats= number of top features to use
# layers= number of neural net layers
# units= number of neural net units
# ngram= the maximum value of "n" to use in n-grams (i.e., 2 means uni + bi)
# dropout= the dropout rate
# min_df= lower bound document frequency to include a word
# max_df= upper bound document frequency to include a word
# stop= the word "None" or "english"
#
# When finished, the program will have written a line to results.csv.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import re
import math
import sys
import json
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout



# See also https://scikit-learn.org/stable/modules/classes.html#text-feature-extraction-ref
from sklearn.feature_extraction.text import TfidfVectorizer

# Defaults; selectively overridden by argument to build_and_eval()
# (or via command-line args if this script is run as a main.)
params = {
    'seed': 12345,
    'train_frac': .8,
    'feats': 2000,
    'layers': 5,
    'units': 96,
    'dropout': .15,
    'ngram': 2,
    'min_df': 0.001,
    'max_df': .998,
    'stop': 'english'
}



def clean(dirty):
    """Given a real-live Reddit comment thread, prepare it for processing by
    converting to lower-case, removing punctuation, etc.
    """
    dirty = re.sub(r"\(https?[^)]*\)","",dirty)
    dirty = dirty.replace("https://en.wikipedia.org/wiki","")
    dirty = dirty.replace("\\n","")
    return dirty.replace("'","")

def vectorize(train_texts, validate_texts=None, vocab=None):
    """Given a list of (cleaned) texts, and optionally a list of validation
    texts, turn them into matrices of features. This involves both the
    traditional "tokenization" step and occurrence counting.

    The reason train_texts and validate_texts are passed separately is that we
    need to .fit_transform() the former but just .transform() the latter.

    Return values:
      - the array of vectorized training texts: one row per text, and one
        column per feature. Each entry is a number, giving the TF/IDF value
        of that feature in that text.
      - if validate_texts is not None, the second return value is the same as
        above, but for the validation texts.
      - a list of the actual words that each column corresponds to.
      - the dictionary generated by the vectorization process, for use in a
        future call to this function, if desired.
    """
    vectorizer = TfidfVectorizer(
        encoding='utf-8',
        lowercase=True,
        stop_words=params['stop'],     # could use None
        ngram_range=(1,params['ngram']),
        analyzer='word',
        min_df=params['min_df'],
        max_df=params['max_df'],
        max_features=None,   # unlimited
        vocabulary=vocab,     # use vocab in the texts
    )
    train_vecs = vectorizer.fit_transform(train_texts).toarray()
    if validate_texts is None:
        return (train_vecs, vectorizer.get_feature_names_out(),
            vectorizer.vocabulary_)
    else:
        validate_vecs = vectorizer.transform(validate_texts).toarray()
        return (train_vecs, validate_vecs, vectorizer.get_feature_names_out(),
            vectorizer.vocabulary_)



def build_and_eval(overridden_params={}):
    """
    overridden_params is an optional dictionary of parameter key/value pairs 
    which will selectively override the defaults at the top of this file.

    In the case where train_frac = 1, all data will be used for training, and
    hence the evaluation step will be omitted. Also, in this case, the "vocab"
    for all the training samples (a dictionary) will be written to vocab.json
    and the already-fit feature selector will be pickled to selector.pickle.

    This function returns the trained model object, as well as saving it to
    the Polarops.model directory structure.
    """

    for p,v in overridden_params.items():
        params[p] = v

    evaluate = not math.isclose(params['train_frac'], 1)

    tf.random.set_seed(params['seed'])
    np.random.seed(params['seed'])
    random.seed(params['seed'])

    ht = pd.read_csv("TheHandTaggedDataBaby.csv", comment="#")
    ht.sample(frac=1)   # Shuffle the data before doing anything else.

    # Eliminate the top 10% longest threads, since they comprise most of a
    # long, probably-not-representative tail. (See email chain 2/18/2022.)
    ht = ht[ht.text.str.len() < ht.text.str.len().quantile(.9)]

    cleaned_texts = np.empty(len(ht), dtype=object)
    for i,row in enumerate(ht.itertuples()):
        cleaned_texts[i] = clean(row.text)
    ht['text'] = cleaned_texts

    if evaluate:
        train = ht.sample(frac=params['train_frac'])
        validate = ht[~ht.index.isin(train.index)]
    else:
        train = ht


    # Build a classifier.
    # TODO: word embeddings

    # The number of "best" features to use, as measured by f_classif.
    NUM_TOP_FEATURES = params['feats']

    # The number of layers in our neural net (including the last
    # layer/activation).
    NUM_LAYERS = params['layers']
    NUM_UNITS = params['units']
    DROPOUT_RATE = params['dropout']
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 500
    BATCH_SIZE = 128


    if evaluate:
        train_vecs, validate_vecs, feature_names,  = \
            vectorize(train.text, validate.text)
    else:
        train_vecs, _, vocab = \
            vectorize(train.text, None)
        with open("vocab.json","w") as f:
            json.dump(vocab, f)

    selector = SelectKBest(f_classif, k=NUM_TOP_FEATURES)
    selector.fit(train_vecs, train.polarized)
    if not evaluate:
        with open("selector.pickle","wb") as f:
            pickle.dump(selector, f)
    x_train = selector.transform(train_vecs).astype('float32')
    if evaluate:
        x_validate = selector.transform(validate_vecs).astype('float32')
        fns = selector.get_feature_names_out(feature_names)


    model = models.Sequential()
    model.add(Dropout(rate=DROPOUT_RATE, input_shape=(NUM_TOP_FEATURES,)))
    for _ in range(NUM_LAYERS-1):
        model.add(Dense(units=NUM_UNITS, activation='relu'))
        model.add(Dropout(rate=DROPOUT_RATE))
    model.add(Dense(units=1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss="binary_crossentropy",
        metrics=['acc'])

    callbacks = []

    if evaluate:
        eval_stuff = (x_validate, validate.polarized)
    else:
        eval_stuff = None
    history = model.fit(
        x_train,
        train.polarized,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        validation_data=eval_stuff,
        verbose=0,  # Logs once per epoch.
        batch_size=BATCH_SIZE)

    if evaluate:
        history = history.history
        print('Validation accuracy: {acc}, loss: {loss}'.format(
            acc=history['val_acc'][-1], loss=history['val_loss'][-1]))
        with open("results.csv","a",encoding="utf-8") as f:
            print(f"""{params['seed']},{params['feats']},{params['layers']},{params['units']},{params['dropout']},{params['ngram']},{params['min_df']},{params['max_df']},{params['stop']},{history['val_acc'][-1]}""", file=f)
    model.save('PolarOps.model')
    return model



if __name__ == "__main__":

    def print_usage():
        print("""
            Usage: enchilada.py [SWEEP]
                seed=[seed]
                train_frac=[train_frac]
                feats=[feats]
                layers=[layers]
                units=[units]
                dropout=[dropout]
                ngram=[ngram]
                min_df=[min_df]
                max_df=[max_df]
                stop=[stop].
        """)

    if len(sys.argv) < 2:
        print_usage()
        sys.exit()
    if len(sys.argv) == 2 and sys.argv[1]=="SWEEP":
        params['feats'] = 2000  # seems good
        params['layers'] = 5  # seems marginally better than fewer
        params['min_df'] = .001  # meh, maybe slightly better if > 0
        params['max_df'] = .998  # meh, ditto
        params['dropout'] = .15  # slightly better
        params['ngram'] = 2  # because this is what the google people say,
                             # not because it really seemed better than 1 or 3
        params['units'] = 96   # this seemed to give us the best
        for units in np.arange(16,128+16,16):
            params['units'] = units
            print(f"Evaluating with units={units}...")
            build_and_eval(params)
    else:
        for param in sys.argv[1:]:
            parts = param.split("=")
            if len(parts) != 2:
                print_usage()
                sys.exit(f"Malformed argument '{param}'.")
            if parts[0] not in params:
                print_usage()
                sys.exit(f"Unknown argument '{parts[0]}'.")
            else:
                params[parts[0]] = parts[1]
                if parts[0] in { 'stop' }:
                    if params[parts[0]] == "None":
                        params[parts[0]] = None
                elif parts[0] in {'train_frac','dropout','min_df','max_df'}:
                    params[parts[0]] = float(params[parts[0]])
                else:
                    params[parts[0]] = int(params[parts[0]])

        build_and_eval(params)
